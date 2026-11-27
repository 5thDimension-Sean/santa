"""Conditional hub voting and a width-eight beam with per-hub-signature diversity."""
from collections import Counter,defaultdict
import ctypes
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from joint_duration import banks,cut,OUT
from joint_near_repair import closure
from shared_device_bench import visit,compile_graph,replay
from joint_chain_check import finish
from hub_vote import forward,exact_two

class Op(ctypes.Structure):
    _fields_=[('op',ctypes.c_char),('a',ctypes.c_int64),('b',ctypes.c_int64)]
class Rule(ctypes.Structure):
    _fields_=[('v',ctypes.c_int64),('r',Op)]
class Patch(ctypes.Structure):
    _fields_=[('n',ctypes.c_int),('a',Rule*4)]

def options(free,t,limit,cap=1000):
    global native
    if 'native' not in globals():native=ctypes.CDLL(str(Path(tempfile.gettempdir())/'santa-hub-paths.dll'))
    vs=np.asarray(sorted(free-{t}),dtype=np.int64)
    buf=(Patch*cap)()
    n=native.patches(ctypes.c_void_p(vs.ctypes.data),len(vs),ctypes.c_int64(t),ctypes.c_int64(limit),buf,cap)
    out=set()
    for i in range(n):
        rules=buf[i]
        out.add(tuple((r.v,(r.r.op.decode(),r.r.a,r.r.b)) for r in rules.a[:rules.n]))
    return sorted(out,key=lambda z:(len(z),z))

def build_library(p,keep,ports):
    initial=closure(p,keep)
    # These are hypothetical inputs for discovery; unmet prerequisites are
    # recorded for every recipe and must be paid for during search.
    hypothetical=set(p.starts)|set(keep)|ports
    bank=defaultdict(set)
    reachable={}
    for t in ports:
        assumed=(ports-{t})|set(p.starts)
        while True:
            extra={v for v,(_,a,b) in keep.items() if a in assumed and b in assumed}
            if extra<=assumed:break
            assumed.update(extra)
        reachable[t]=assumed
    def harvest(pool):
        for t in sorted(ports):
            allowed=pool & (reachable[t]|(pool-hypothetical))
            if len(ports)>12 and len(allowed)>56:
                # Include seeds and keep the operands closest to the target
                # in bit length, alongside small arithmetic constants.
                small=sorted(allowed)[:24]
                near=sorted(allowed-set(small),key=lambda x:(abs(x.bit_length()-t.bit_length()),x))[:32]
                allowed=set(small+near)|set(p.starts)
            bank[t].update(options(allowed,t,p.limit))
    harvest(initial)
    harvest(hypothetical)
    def votes():
        by=defaultdict(dict)
        for t,patterns in bank.items():
            for seq in patterns:
                intermediate={v for v,_ in seq[:-1]}
                deps={x for _,r in seq for x in r[1:]}
                virtual=(deps-hypothetical)-intermediate
                for v in intermediate|virtual:
                    if v in hypothetical:continue
                    cost=len(seq)+(v in virtual)
                    by[v][t]=min(by[v].get(t,99),cost)
        frequency=Counter(t for support in by.values() for t in support)
        return sorted(((sum(1/(c*frequency[t]**0.5) for t,c in support.items()),len(support),v) for v,support in by.items()),key=lambda r:(-r[0],-r[1],r[2]))
    provisional=votes()
    # Enrich with balanced alternatives from each target, including those
    # with only one vote before other intermediates are made available.
    added={v for _,_,v in provisional[:16]}
    for t,patterns in bank.items():
        cheap=sorted((len(seq),v) for seq in patterns for v,_ in seq[:-1] if v not in hypothetical)
        added.update(v for _,v in cheap[:12])
    harvest(hypothetical|added)
    scores=votes()
    return bank,scores

def prepare(bank,keep,ports,p):
    known=set(keep)|ports|set(p.starts)
    preds=defaultdict(set)
    for t,patterns in bank.items():
        for seq in patterns:
            for i,(v,r) in enumerate(seq):
                if v not in known:preds[v].add(seq[:i+1])
    # Include standalone construction of promising hubs; never import an
    # operand from the hypothetical pool unless closure already contains it.
    return preds

def ready(seq,free,existing):
    vals=set();pay={}
    for v,r in seq:
        if v in free or v in vals:continue
        if v in existing or not all(x in vals or x in free for x in r[1:]):return None
        vals.add(v);pay[v]=r
    return pay

def run(p,old,keep,ports,bank,scores,mode,layers=12,monitor=None,leaf=True):
    initial=closure(p,keep)
    pred=prepare(bank,keep,ports,p)
    hubs=[v for _,_,v in scores[:32]]
    priority={v:s for s,_,v in scores}
    plans={}
    for t,patterns in bank.items():
        plans[t]=[]
        for seq in sorted(patterns,key=lambda s:(len(s),s)):
            made=frozenset(v for v,_ in seq)
            external=frozenset(x for _,r in seq for x in r[1:])-made
            plans[t].append((seq,made,external))
    for v in hubs:plans[v]=[(seq,frozenset(x for x,_ in seq),frozenset(y for _,r in seq for y in r[1:])-set(x for x,_ in seq)) for seq in pred[v]]
    def plan_order(item):
        seq,made,external=item
        reward=sum(priority.get(v,0) for v in made|external)
        return len(seq)+len(external-initial)*0.55-reward*6,len(seq),seq
    for choices in plans.values():
        choices.sort(key=plan_order)
        if len(ports)>12:del choices[128:]
    # Compact bitsets for evaluating residual estimates over many siblings.
    allvals=set(initial)|set(keep)|ports|set(hubs)
    for choices in plans.values():
        for _,m,e in choices:allvals.update(m|e)
    bits={v:1<<i for i,v in enumerate(sorted(allvals))}
    def mask(values):
        n=0
        for v in values:n|=bits.get(v,0)
        return n
    estimates={t:list({(mask(m),mask(e)) for _,m,e in plans[t][:48]}) for t in ports}
    score_cache={}
    def predicted(free,exact=None):
        fm=mask(free)
        if exact is None and fm in score_cache:return score_cache[fm]
        total=0;seen=0
        for t in sorted(ports-free):
            cost=9;used=0
            for m,e in estimates[t]:
                req=m&~fm;unpaid=e&~fm
                d=req.bit_count()+2*unpaid.bit_count()
                if d<cost:cost=d;used=req|unpaid
            # Discount an intermediate only when two residuals name it.
            if exact is not None and exact[t]<cost:cost=exact[t];used=0
            total+=cost-.5*(seen&used).bit_count();seen|=used
        if exact is None:score_cache[fm]=total
        return total
    def compact(seed,free):
        # Reprice an existing output once a later helper becomes available.
        # Keep neutral alternatives for one more rewrite to capture shared costs.
        want=ports&free
        original=dict(seed)
        best_patch=original
        variants=[original]
        for turn in range(2):
            alternatives={}
            for variant in variants:
                for t in sorted(want):
                    base={v:r for v,r in variant.items() if v!=t}
                    readyfree=closure(p,keep|base)
                    direct=forward(readyfree,p.limit)
                    for seq,made,ext in plans[t]:
                        missing=ext-readyfree
                        if len(missing)>1 or any(x not in direct or x in keep for x in missing):continue
                        added={x:direct[x] for x in missing}
                        edits=ready(seq,readyfree|missing,keep)
                        if not edits:continue
                        rules=keep|base|added|edits
                        # Already-computed intermediates in the replacement may
                        # vanish when the old calculation is removed.
                        order=visit(want,rules);paid={x:rules[x] for x in order if x not in keep}
                        if len(paid)>len(original) or not want<=closure(p,keep|paid):continue
                        sig=frozenset(paid)
                        alternatives.setdefault(sig,paid)
                        if len(paid)<len(best_patch):best_patch=paid
            variants=sorted(alternatives.values(),key=lambda d:(len(d),-sum(priority.get(v,0) for v in d)))[:8]
            if len(best_patch)<len(original) or not variants:break
        return best_patch
    generated=set();survived=set();evidence=[];best=old;checked=set()
    states=[({},initial)];wins=[];used_finisher=0
    for depth in range(layers):
        if len(best)<len(old):break
        following={}
        for patch,free in states:
            direct=forward(free,p.limit)
            for t in sorted((ports-free)|(set(hubs)-free)):
                choices={}
                for seq,made,ext in plans[t]:
                    missing=ext-free
                    if len(missing)>1 or any(x not in direct or x in keep for x in missing):continue
                    seed={v:direct[v] for v in missing}
                    pay=ready(seq,free|missing,keep)
                    if not pay:continue
                    combined=seed|pay
                    if any(v in r[1:] for v,r in combined.items()):continue
                    key=frozenset(combined)
                    reward=sum(priority.get(v,0) for v in combined)
                    choices.setdefault(key,((len(combined)-5*reward,len(combined),seq),combined))
                for _,pay in sorted(choices.values(),key=lambda item:item[0])[:24 if t in ports else 1]:
                    new=patch|pay;nf=closure(p,keep|new)
                    generated.update(nf-initial)
                    signature=frozenset(nf)
                    if ports<=nf:
                        rules=keep|new;order=visit(p.targets,rules)
                        if len(order)<len(best):
                            c=compile_graph(p,rules,order)
                            if c is not None:
                                replay(p,c);best=c;wins.append(('direct',depth+1,len(c)))
                        continue
                    remaining=ports-nf
                    basic=(len(new)-len(ports&nf),-len(ports&nf),len(new))
                    if mode=='legacy':score=basic
                    else:score=(len(new)+predicted(nf)+3*len(remaining),*basic)
                    if signature not in following or score<following[signature][0]:
                        following[signature]=(score,new,nf)
        if not following:break
        ranked=sorted(following.values(),key=lambda z:z[0])
        if mode!='legacy':
            # Screen by the integer expression estimate, then price exact
            # depth-one/two reaches for both global and signature finalists.
            finalists=list(ranked[:16]);per_group=Counter()
            for entry in ranked:
                key=tuple(v for v in hubs if v in entry[2])
                if per_group[key]<2 and len(finalists)<48:finalists.append(entry)
                per_group[key]+=1
            refined={}
            for _,candidate,free in finalists:
                key=frozenset(free)
                if key in refined:continue
                remaining=ports-free
                dist=exact_two(free,remaining,p.limit)
                value=len(candidate)+predicted(free,dist)+3*len(remaining)
                refined[key]=((value,len(candidate)-len(ports&free),-len(ports&free)),candidate,free)
            ranked=sorted(refined.values(),key=lambda z:z[0])
            groups=defaultdict(list)
            for entry in ranked:groups[tuple(v for v in hubs if v in entry[2])].append(entry)
            # First seat per signature, then a second seat if width is available.
            ranked=[group[i] for i in range(8) for group in groups.values() if len(group)>i]
        states=[(s,f) for _,s,f in ranked[:8]]
        for patch,nf in states:
            survived.update(nf-initial)
            if len(ports-nf)==1:
                item={'depth':depth+1,'missing':sorted(ports-nf),'patch':patch,'cost':len(patch)}
                if monitor is not None:item['observed_hub']=monitor in nf
                evidence.append(item)
        # Only the leading surviving one-missing state of a hub group is
        # eligible for the native four-instruction leaf, at most once per state.
        if leaf:
            groups=set()
            for seat,(patch,nf) in enumerate(states):
                if len(ports-nf)!=1:continue
                key=tuple(v for v in hubs if v in nf)
                if key in groups or frozenset(nf) in checked or used_finisher>=16:continue
                groups.add(key);checked.add(frozenset(nf))
                if mode!='legacy':
                    reduced=compact(patch,nf)
                    if len(reduced)<len(patch):
                        patch=reduced;nf=closure(p,keep|patch)
                        states[seat]=(patch,nf)
                        survived.update(nf-initial)
                        evidence.append({'depth':depth+1,'missing':sorted(ports-nf),'patch':patch,
                                         'cost':len(patch),'repriced':True,'observed_hub':monitor in nf if monitor else None})
                res=finish(p,keep,ports,patch);used_finisher+=1
                if res['program'] and len(res['program'])<len(best):
                    best=res['program'];wins.append(('finisher',depth+1,len(best),monitor in nf if monitor else None))
    return generated,survived,evidence,best,wins,used_finisher


"""Test single- and two-output repairs against the historical old checkpoint."""
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from joint_duration import banks, OUT
from near_reductions import canonical
from optimize import graph, compile_graph, replay


def inverses(t,x,limit):
    if t>x:yield t-x,('+',x,t-x)
    if x>t:yield x-t,('-',x,x-t)
    if t<=limit-x:yield t+x,('-',t+x,x)
    if t%x==0:yield t//x,('*',x,t//x)
    if x%t==0:yield x//t,('/',x,x//t)
    if t<=limit//x:yield t*x,('/',t*x,x)


def ops(a,b,limit):
    if a<b:a,b=b,a
    if a<=limit-b:yield a+b,('+',a,b)
    if a>b:yield a-b,('-',a,b)
    if a<=limit//b:yield a*b,('*',a,b)
    if a%b==0:yield a//b,('/',a,b)


def sig(rules):
    return tuple(sorted((v, *canonical(r)) for v,r in rules.items()))


def run(job):
    p,old,donor,cases=job
    g=graph(p,old); dg=graph(p,donor); seeds=set(p.starts); oldpool=seeds|g.keys()
    original=len(old); bits={v:1<<i for i,v in enumerate(g)}
    @lru_cache(maxsize=200)
    def segments(cut):
        # Collapse old graph regions to bit masks. A cut node is a hole to expand
        # under the proposed recipe. Target dependencies are retained byte-for-byte.
        masks={v:(0,()) for v in seeds}
        for v,(_,a,b) in g.items():
            if v in cut:masks[v]=(0,(v,));continue
            ma,ha=masks[a];mb,hb=masks[b]
            masks[v]=(ma|mb|bits[v],tuple(set(ha+hb)))
        mask=0;holes=set()
        for t in p.targets:
            m,h=masks[t];mask|=m;holes.update(h)
        return masks,mask,holes

    def size(patch):
        masks,mask,holes=segments(frozenset(v for v in patch if v in g))
        seen=set();active=set()
        def expand(v):
            nonlocal mask
            if v in seen:return True
            if v in active:return False
            active.add(v)
            for x in patch[v][1:]:
                if x in patch:
                    if not expand(x):return False
                elif x in masks:
                    m,h=masks[x];mask|=m
                    if any(not expand(z) for z in h):return False
                else:return False
            active.remove(v);seen.add(v);return True
        if all(expand(z) for z in holes):return mask.bit_count()+len(seen)
        return None

    def legal(patch):
        rules=g|patch
        for prioritization in (list(g),list(p.targets),list(dg),list(reversed(p.targets))):
            seen=set(seeds);active=set();order=[]
            def walk(v):
                if v in seen:return True
                if v in active or v not in rules:return False
                active.add(v)
                if not all(walk(x) for x in rules[v][1:]):return False
                active.remove(v);seen.add(v);order.append(v);return True
            if not all(walk(t) for t in p.targets):return None
            live=set(order);order=[];seen=set(seeds)
            for v in prioritization:
                if v in live:walk(v)
            for t in p.targets:walk(t)
            prog=compile_graph(p,rules,order)
            if prog is not None:
                assert replay(p,prog)==size(patch)
                return prog
        return None

    def historical(t):
        patch={}
        def add(v):
            if v in seeds or v in patch:return
            if v!=t and v in g:return
            patch[v]=dg[v]
            for x in dg[v][1:]:add(x)
        add(t)
        return patch

    # Historical partner catalog. Each route stops at a value in the old program.
    historical_by_hub=defaultdict(list)
    for q in p.targets:
        if q not in dg:continue
        r=historical(q)
        for h in r.keys()-oldpool:historical_by_hub[h].append((q,r))
    # Unseeded partner catalog: use a supplied new hub in the first or last
    # operation of a one- or two-instruction partner route. A companion operand
    # can also be produced from two old values. Enumerate all such recipes.
    back=defaultdict(list);one=defaultdict(list)
    oldvals=sorted(oldpool)
    for i,x in enumerate(oldvals):
        for y in oldvals[:i+1]:
            for z,r in ops(x,y,p.limit):
                if z not in oldpool:one[z].append(r)
        for q in p.targets:
            for z,r in inverses(q,x,p.limit):
                if z not in oldpool:back[z].append((q,r))
    @lru_cache(maxsize=4096)
    def partner(h):
        result={}
        def add(q,patch):
            if q not in g or (q in patch[q][1:]) or any(z in r[1:] for z,r in patch.items()):return
            if canonical(patch[q])==canonical(g[q]):return
            result[(q,sig(patch))]=(q,patch)
        for q in p.targets:
            for z,last in inverses(q,h,p.limit):
                if z in oldpool:add(q,{q:last})
                else:
                    for r in one.get(z,()):add(q,{z:r,q:last})
        for x in oldvals:
            for z,r in ops(x,h,p.limit):
                if z in oldpool or z==h:continue
                for q,last in back.get(z,()):add(q,{z:r,q:last})
        return list(result.values())

    def graft(h,r):
        # A partner-only control must pay to construct the shared hub even when
        # the first output's final operation has been removed.
        patch={}
        def add(v):
            if v not in r or v in patch:return
            patch[v]=r[v]
            for x in r[v][1:]:add(x)
        add(h);return patch

    results=[];winners=[]
    for t,reason,raw in cases:
        meter=Counter();best={};best_raw={};best_solo=None
        def test(first,mode):
            nonlocal best_solo
            n=size(first)
            if n is None:return
            meter[mode+'_valid_routes']+=1
            if n<original:
                prog=legal(first)
                if prog and (best_solo is None or n<best_solo):best_solo=n
                return
            fresh=first.keys()-oldpool
            if not fresh:return
            for h in fresh:
                supply=graft(h,first)
                catalogs=[('historical_pair',historical_by_hub[h])] if mode=='historical' else [('generated_pair',partner(h)),('mixed_pair',historical_by_hub[h])]
                for arm,catalog in catalogs:
                    for q,other in catalog:
                        if q==t or q in first or t in other:continue
                        # Check both recipes for common nodes when they differ.
                        overlaps=first.keys()&other.keys()
                        conflicts=[v for v in overlaps if canonical(first[v])!=canonical(other[v])]
                        for choices in product((False,True),repeat=len(conflicts)):
                            combined=first|other
                            for v,choose_first in zip(conflicts,choices):
                                if choose_first:combined[v]=first[v]
                            meter[arm+'_trials']+=1
                            joint=size(combined)
                            if joint is None or joint>=original:continue
                            # Verify that both routes still directly depend on h
                            # after resolving common recipes, independently of the other target.
                            def uses(v,stop):
                                if v==h:return True
                                if v==stop or v not in combined:return False
                                return any(uses(x,stop) for x in combined[v][1:])
                            if not uses(t,q) or not uses(q,t):continue
                            if joint>=best.get(arm,original):continue
                            joint_prog=legal(combined)
                            if joint_prog is None:continue
                            best_raw[arm]=min(best_raw.get(arm,original),joint)
                            # Build solo patches using the exact recipes selected in the joint graph.
                            left=graft(t,combined);right=graft(q,combined)
                            ln=size(left);rn=size(right)
                            if ln is None or rn is None or ln<original or rn<=joint:continue
                            if legal(left) is None or legal(right) is None:continue
                            best[arm]=joint
                            winners.append(dict(id=p.id,target=t,partner=q,hub=h,arm=arm,old=original,
                                                focus_alone=ln,partner_alone=rn,joint=joint,
                                                left=sig(left),right=sig(right),program=joint_prog))
        for row in raw:
            first={v:(op,a,b) for v,op,a,b in row}
            if any(v in r[1:] for v,r in first.items()):continue
            test(first,'generated')
        test(historical(t),'historical')
        results.append(dict(id=p.id,target=t,reason=reason,old=original,raw_routes=len(raw),
                            best_solo=best_solo,**meter,
                            **{a:original-n for a,n in best.items()},
                            **{a+'_raw_saved':original-n for a,n in best_raw.items()}))
    return results,winners


def main():
    p,old,new=banks();groups=defaultdict(list)
    for line in (OUT/'shared-route-candidates.jsonl').open():
        k,t,reason,raw=json.loads(line);groups[k].append((t,reason,raw))
    arms=['generated_pair','mixed_pair','historical_pair']
    fields=['id','target','reason','old','raw_routes','best_solo','generated_valid_routes','historical_valid_routes',
            *[a+'_trials' for a in arms],*arms,*[a+'_raw_saved' for a in arms]]
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(7) as pool,(OUT/'shared-route-test.csv').open('w',newline='') as f,(OUT/'shared-route-wins.jsonl').open('w') as wins:
        w=csv.DictWriter(f,fieldnames=fields,restval=0);w.writeheader()
        jobs=[pool.submit(run,(p[k],old[k],new[k],v)) for k,v in groups.items()]
        for i,job in enumerate(as_completed(jobs),1):
            rows,ws=job.result()
            w.writerows(rows);f.flush()
            for item in ws:print(json.dumps(item,separators=(',',':')),file=wins)
            wins.flush()
            if i%10==0:print(i,'/',len(jobs),flush=True)
if __name__=='__main__':main()

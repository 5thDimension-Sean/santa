"""Jointly rebuild a deleted set using only surviving values and boundary outputs."""
import csv
import time
import numpy as np
from near_rank_audit import OUT
from shared_device_bench import Engine, load_puzzles, read_bank, graph, visit, compile_graph, replay
from near_reductions import canonical,rank,unfocused_rows,convergents


def closure(p,rules):
    free=set(p.starts)
    while True:
        more={v for v,(_,a,b) in rules.items() if a in free and b in free}
        if more<=free:return free
        free.update(more)


def paths(engine,p,free,t):
    live=np.array(sorted(free),dtype=np.int64);n=len(live)
    raw=engine.run('make_values',n*n,live,np.int32(n),np.int64(p.limit),out=np.empty(4*n*n,dtype=np.int64))
    vals,idx=np.unique(np.concatenate((live,raw)),return_index=True)
    ok=vals>0;vals,idx=vals[ok],idx[ok]; cost=(idx>=n).astype(np.int32)
    def one(v):
        pos=int(vals.searchsorted(v))
        if pos==len(vals) or vals[pos]!=v:return None
        if not cost[pos]:return {}
        xy,op=divmod(int(idx[pos])-n,4);i,j=divmod(xy,n)
        a,b=int(live[i]),int(live[j])
        return {v:('+-*/'[op],b,a) if op in (1,3) else ('+-*/'[op],a,b)}
    direct=one(t)
    if direct is not None:yield direct
    ranks=rank(engine,t,live,vals,cost,unfocused_rows(cost),p.limit,{t})
    probes=[(t,None,None)]+[(v,k,r) for _,v,k,r,_,d in ranks[:8] if d<127]
    # Reapply the quotient/remainder reduction to the selected near-targets.
    # Materialize every coefficient, correction and denominator in the DAG.
    for _,v,k,outer,_,_ in ranks[:8]:
        payment=one(k)
        if payment is None: continue
        for b in live:
            b=int(b)
            for q,d in {((v+b//2)//b,1),*convergents(v,b,8)}:
                if not q or q>p.limit//b or v>p.limit//d: continue
                product=q*b
                numerator=v*d
                e=abs(numerator-product)
                needed=([q] if product not in free else [])+([d] if d!=1 else [])+([e] if e and numerator not in free else [])
                paid=[one(x) for x in needed]
                if any(x is None for x in paid):continue
                patch=dict(payment)
                for recipes in paid: patch.update(recipes)
                if product not in free:
                    patch[product]=('*',q,b)
                if e and numerator not in free:
                    patch[numerator]=('+',product,e) if product<numerator else ('-',product,e)
                if d!=1 and v not in free: patch[v]=('/',numerator,d)
                patch[t]=outer
                # All operands must be constructible without the deleted roots.
                reached=set(free)
                for _ in range(len(patch)):
                    reached.update(z for z,(_,a,b) in patch.items() if a in reached and b in reached)
                if set(patch)<=reached: yield patch
    roots=np.array([v for v,_,_ in probes],dtype=np.int64)
    m=len(roots)
    out=engine.run('links',m*len(vals),vals,cost,np.int32(len(vals)),roots,np.int32(m),
                   np.int64(p.limit),out=np.empty((len(vals),m),dtype=np.int32))
    for col,(v,k,r) in enumerate(probes):
        for row in np.flatnonzero(out[:,col]>=0)[:30]:
            qi,op=divmod(int(out[row,col]),6)
            h,x=int(vals[row]),int(vals[qi])
            if v in (h,x) or t in (h,x,k):continue
            patch=one(h)|one(x)
            a,b=(x,h) if op in (2,5) else (h,x)
            patch[v]=('+--*//'[op],a,b)
            if r:
                payment=one(k)
                if payment is None or t in payment:continue
                patch.update(payment);patch[t]=r
            yield patch


def rebuild(engine,p,old,deleted,seconds):
    g=graph(p,old);keep={v:r for v,r in g.items() if v not in deleted}
    ports=(set(p.targets)|{x for r in keep.values() for x in r[1:]}) & deleted
    start=time.perf_counter();deadline=start+seconds
    best=None;max_ports=0;expanded=0
    states=[{}]
    while states and time.perf_counter()<deadline:
        following={}
        for patch in states:
            rules=keep|patch;free=closure(p,rules)
            remaining=ports-free;max_ports=max(max_ports,len(ports)-len(remaining))
            if not remaining:
                order=visit(p.targets,rules)
                if len(order)<len(old) and (best is None or len(order)<len(best)):
                    candidate=compile_graph(p,rules,order)
                    if candidate is not None: replay(p,candidate);best=candidate
                continue
            for t in sorted(remaining):
                if time.perf_counter()>=deadline:break
                expanded+=1
                for ops in paths(engine,p,free,t):
                    if any(v in rules and v not in free for v in ops):continue
                    new=patch|{v:r for v,r in ops.items() if v not in free}
                    if len(new)>=len(deleted):continue
                    done=closure(p,keep|new)&ports
                    max_ports=max(max_ports,len(done))
                    if done==ports:
                        order=visit(p.targets,keep|new)
                        if len(order)<len(old) and (best is None or len(order)<len(best)):
                            cand=compile_graph(p,keep|new,order)
                            if cand is not None: replay(p,cand);best=cand
                    score=(len(new)-len(done),-len(done),len(new))
                    following.setdefault(frozenset(new),(score,new))
        states=[patch for _,patch in sorted(following.values(),key=lambda pair:pair[0])[:8]]
    return best,(len(deleted),len(ports),max_ports,expanded,time.perf_counter()-start)


def main():
    import json
    P=load_puzzles('puzzles.csv');current=read_bank('submission.csv',P)
    older=read_bank('public/ishankarnick/submission.csv',P);engine=Engine(False)
    gaps=sorted(((len(current[k])-len(set(p.targets)-set(p.starts)),k) for k,p in P.items()),reverse=True)
    with (OUT/'near-miss-headroom.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','length','missing_target_lower_bound','gap_ceiling'])
        for gap,k in gaps:w.writerow([k,len(current[k]),len(current[k])-gap,gap])
    rows=[]
    for k,kind in [('scored_0379','checkpoint-difference')]+[(k,'shared-descendants') for _,k in gaps[:3]]:
        p=P[k];base=older[k] if kind=='checkpoint-difference' else current[k]
        g=graph(p,base);witness=''
        if kind=='checkpoint-difference':
            h=graph(p,current[k]);deleted={v for v in g if v not in h or canonical(g[v])!=canonical(h[v])}
            intact={v:r for v,r in g.items() if v not in deleted}
            rules=intact|{v:r for v,r in h.items() if v not in intact}
            order=visit(p.targets,rules)
            cand=compile_graph(p,rules,order)
            witness=replay(p,cand)
        else:
            options=[]
            for pivot in g:
                desc={pivot}
                for v,(_,a,b) in g.items():
                    if a in desc or b in desc:desc.add(v)
                if len(desc&set(p.targets))>=2:options.append(desc)
            deleted=min(options,key=lambda s:abs(len(s)-12))
        cand,metrics=rebuild(engine,p,base,deleted,10)
        if cand:
            (OUT/f'near-miss-joint-{k}.json').write_text(json.dumps(cand))
        rows.append([k,kind,len(base),witness,*metrics,len(cand) if cand else '',len(base)-len(cand) if cand else 0])
        print(rows[-1],flush=True)
    with (OUT/'near-miss-joint.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','cut','old','injected_score','deleted','required_outputs','max_rebuilt','expanded','seconds','score','savings']);w.writerows(rows)
if __name__=='__main__':main()


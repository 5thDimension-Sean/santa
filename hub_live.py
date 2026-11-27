"""Time-matched streaming hub vote over constructible proposals."""
import csv,json,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from joint_duration import banks,cut,OUT,worker_init
import joint_duration as jd
from joint_near_repair import paths,closure
from shared_device_bench import visit,compile_graph,replay
from hub_beam import options
import hub_beam
import ctypes,tempfile
from pathlib import Path
from hub_vote import exact_two
from joint_chain_check import finish


def search_live(engine,p,old,keep,ports,seconds,enhanced):
    indexed=enhanced=='indexed'
    if indexed:hub_beam.native=ctypes.CDLL(str(Path(tempfile.gettempdir())/'santa-hub-index.dll'))
    start=time.perf_counter();end=start+seconds;initial=closure(p,keep)
    generated=set();retained=set();states=[({},initial)]
    library=defaultdict(dict);hub_votes=defaultdict(dict);cached={}
    best=old;leaves=0;locked=set();rounds=0
    while states and time.perf_counter()<end:
        following={}
        for patch,free in states:
            prefixes={}
            empty_queries=0
            for t in sorted(ports-free):
                if time.perf_counter()>=end:break
                key=(frozenset(free),t)
                if key not in cached:
                    # Preserve original recipes. Add independently derived
                    # reductions using the actual surviving values. Cache both.
                    opts={}
                    if enhanced:
                        candidates=options(free,t,p.limit,96)
                        for seq in candidates:
                            unique=frozenset(seq)
                            opts[unique]=dict(seq)
                    # Prefer direct native joins for this arm. Fall back to the
                    # older generator when the join finds no usable proposal.
                    if indexed and not opts:empty_queries+=1
                    expansions=() if indexed and (opts or empty_queries>2) else paths(engine,p,free,t)
                    for ops in expansions:
                        opts.setdefault(frozenset(ops.items()),ops)
                        if time.perf_counter()>=end:break
                    cached[key]=list(opts.values())
                for ops in cached[key]:
                    if any(v in keep and v not in free for v in ops):continue
                    pay={v:r for v,r in ops.items() if v not in free}
                    if not pay or t not in pay:continue
                    new=patch|pay;nf=closure(p,keep|new)
                    if t not in nf:continue
                    generated.update(nf-initial)
                    covered=ports&nf
                    if len(covered)==len(ports):
                        order=visit(p.targets,keep|new)
                        if len(order)<len(best):
                            c=compile_graph(p,keep|new,order)
                            if c is not None:replay(p,c);best=c
                        continue
                    signature=frozenset(new.items())
                    if signature in following:continue
                    following[signature]=(new,nf)
                    if enhanced:
                        helpers=frozenset(pay)-ports
                        external=frozenset(x for r in pay.values() for x in r[1:])-set(pay)
                        library[t].setdefault((frozenset(pay),external),None)
                        if indexed:
                            for h in helpers:
                                bundle={v:pay[v] for v in visit([h],pay)}
                                if h not in prefixes or len(bundle)<len(prefixes[h]):prefixes[h]=bundle
                        for h in helpers|(external-initial-ports):
                            weights=hub_votes[h];weights[t]=max(weights.get(t,0),1/len(pay))
                if time.perf_counter()>=end:break
            if indexed:
                shared=sorted((h for h in prefixes if len(hub_votes[h])>=2),
                              key=lambda h:(-sum(hub_votes[h].values())/len(prefixes[h]),h))
                for h in shared[:12]:
                    new=patch|prefixes[h];nf=closure(p,keep|new)
                    if h not in nf:continue
                    generated.update(nf-initial)
                    following.setdefault(frozenset(new.items()),(new,nf))
            if time.perf_counter()>=end:break
        if not following:break
        def basic(item):
            patch,nf=item
            return (len(patch)-len(ports&nf),-len(ports&nf),len(patch))
        items=sorted(following.values(),key=basic)
        if enhanced:
            votes={h:sum(by_target.values()) for h,by_target in hub_votes.items()}
            top=sorted(votes,key=lambda h:(-votes[h],h))[:32]
            # Global and per-signature finalists both get a cost-to-go estimate.
            groups=defaultdict(list)
            for item in items:
                key=tuple(h for h in top if h in item[1])
                groups[key].append(item)
            def utility(group):
                return sum(votes[h] for h in group)
            keys=sorted(groups,key=lambda k:(basic(groups[k][0])[0]-.5*utility(k),k))[:32]
            pool=items[:16]+[item for k in keys for item in groups[k][:2]]
            selected={};estimates={t:sorted(library[t],key=lambda x:len(x[0]))[:24] for t in ports}
            for patch,nf in pool:
                sig=frozenset(nf)
                if sig in selected:continue
                rem=ports-nf;exact=exact_two(nf,rem,p.limit)
                expense=0;used=set()
                for t in sorted(rem):
                    smallest=min(8,exact[t]);needed=set()
                    for made,deps in estimates[t]:
                        unpaid=made-nf;extra=deps-nf
                        cost=len(unpaid)+2*len(extra)
                        if cost<smallest:smallest=cost;needed=unpaid|extra
                    expense+=smallest-.5*len(used&needed);used.update(needed)
                key=tuple(h for h in top if h in nf)
                score=len(patch)+expense-.25*utility(key)
                selected[sig]=(score,key,patch,nf)
            bins=defaultdict(list)
            for value in sorted(selected.values(),key=lambda x:x[0]):bins[value[1]].append(value)
            ranked=[b[i] for i in range(8) for b in bins.values() if len(b)>i]
            items=[(patch,nf) for _,_,patch,nf in ranked]
        states=items[:8];rounds+=1
        for patch,nf in states:
            # Count only admissions finalized before the wall-clock deadline.
            if time.perf_counter()>end:break
            retained.update(nf-initial)
            if enhanced and len(ports-nf)==1 and leaves<2 and time.perf_counter()+.25<end:
                sig=tuple(h for h in top if h in nf)
                if sig in locked:continue
                locked.add(sig);res=finish(p,keep,ports,patch);leaves+=1
                if time.perf_counter()<end and res['program'] and len(res['program'])<len(best):best=res['program']
    if len(best)<len(old):
        (OUT/f'hub-live-winner-{enhanced}-{p.id}.json').write_text(json.dumps(best))
    return dict(id=p.id,generated=sorted(generated),survived=sorted(retained),score=len(best),
                leaves=leaves,seconds=time.perf_counter()-start,rounds=rounds)

def worker(args):
    p,old,keep,ports,enhanced=args
    return search_live(jd.ENGINE,p,old,keep,ports,10,enhanced)

def main(indexed=False):
    P,O,N=banks();keys=[k for k in P if len(O[k])>len(N[k]) and k!='scored_0379']
    tasks=[]
    for i,k in enumerate(keys):
        for arm in (['indexed'] if indexed else ([False,True] if i%2 else [True,False])):
            keep,ports,_=cut(P[k],O[k],N[k]);tasks.append((P[k],O[k],keep,ports,arm))
    with (OUT/('hub-heldout-indexed.jsonl' if indexed else 'hub-heldout-live.jsonl')).open('w') as f,ProcessPoolExecutor(max_workers=6,initializer=worker_init,initargs=('cpu',)) as pool:
        futures={pool.submit(worker,a):a[-1] for a in tasks}
        for i,future in enumerate(as_completed(futures),1):
            row=future.result();row['enhanced']=futures[future]
            print(json.dumps(row),file=f,flush=True)
            if i%30==0:print(i,'/180',flush=True)
if __name__=='__main__':
    import sys
    main('--indexed' in sys.argv)

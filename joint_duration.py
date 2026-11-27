"""Prefix-timed joint repair. The newer bank supplies cut locations only."""
import os
os.environ['OMP_THREAD_LIMIT']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from joint_near_repair import paths,closure
from shared_device_bench import Engine,graph,visit,compile_graph,replay,load_puzzles,read_bank
from near_reductions import canonical

OUT=Path('experiments')
BUDGETS=(1,10,60)


def cut(p,old,new):
    g=graph(p,old);h=graph(p,new)
    gone={v for v in g if v not in h or canonical(g[v])!=canonical(h[v])}
    keep={v:r for v,r in g.items() if v not in gone}
    ports=(set(p.targets)|{x for r in keep.values() for x in r[1:]})&gone
    return keep,ports,gone


def search(engine,p,old,keep,ports,deleted,seconds,seed=None,detail=False,strict=False,diverse=False,pathgen=paths,tracker=None):
    start=time.perf_counter()
    events=[];wins=[];logs={};best=len(old);max_ports=-1
    expanded=0;pruned=0;above=0;cache={};states=[seed or {}]
    def note(patch,free,now):
        nonlocal max_ports,best
        elapsed=now-start
        if elapsed>seconds:return
        reached=ports&free
        if len(reached)>max_ports:
            max_ports=len(reached);events.append((elapsed,max_ports))
        if detail and len(reached)>=len(ports)-1:
            key=tuple(sorted(patch.items()))
            if key not in logs:
                logs[key]={'elapsed':elapsed,'missing':sorted(ports-free),'patch':patch,
                           'retained_values':sorted(free),'new_nodes':len(patch)}
        if reached!=ports:return
        rules=keep|patch; order=visit(p.targets,rules)
        if len(order)>=best:return
        candidate=compile_graph(p,rules,order)
        if candidate is None:return
        length=replay(p,candidate);elapsed=time.perf_counter()-start
        if elapsed<=seconds:
            best=length;wins.append((elapsed,candidate))
    note(states[0],closure(p,keep|states[0]),time.perf_counter())
    while states and time.perf_counter()-start<seconds:
        following={}
        for patch in states:
            rules=keep|patch;free=closure(p,rules)
            # Reverse magnitude is a deterministic ordering alternative. The
            # diversity arm also reserves seats for different unfinished sets.
            for t in sorted(ports-free, reverse=diverse):
                if time.perf_counter()-start>=seconds:break
                expanded+=1
                key=(frozenset(free),t)
                if key not in cache:
                    found=[]
                    for ops in pathgen(engine,p,free,t):
                        if time.perf_counter()-start>=seconds:break
                        found.append(ops)
                    if len(cache)>4096:cache.clear()
                    cache[key]=found
                for ops in cache[key]:
                    now=time.perf_counter()
                    if now-start>=seconds:break
                    if any(v in rules and v not in free for v in ops):continue
                    new=patch|{v:r for v,r in ops.items() if v not in free}
                    if len(new)>=len(deleted):
                        above+=1
                        if strict:pruned+=1;continue
                    done=closure(p,keep|new)
                    if tracker is not None:tracker[0].update(done)
                    note(new,done,time.perf_counter())
                    if ports<=done:continue
                    n=len(done&ports)
                    signature=tuple(sorted(new.items()))
                    following.setdefault(signature,((len(new)-n,-n,len(new)),new,frozenset(ports-done)))
                if time.perf_counter()-start>=seconds:break
            if time.perf_counter()-start>=seconds:break
        # Preserve distinct recipes with identical value sets.
        ranked=sorted(following.values(),key=lambda item:item[0])
        if diverse:
            first={};rest=[]
            for item in ranked:
                if item[2] in first:rest.append(item)
                else:first[item[2]]=item
            ranked=[*first.values(),*rest]
        states=[v for _,v,_ in ranked[:8]]
        if tracker is not None:
            for survivor in states:tracker[1].update(closure(p,keep|survivor))
    elapsed=time.perf_counter()-start
    return {'elapsed':elapsed,'exhausted':not states and elapsed<seconds,'expanded':expanded,
            'over_old_cap':above,'pruned_by_cost':pruned,'progress':events,
            'wins':wins,'six_states':list(logs.values())}


def worker_init(device):
    global ENGINE
    ENGINE=Engine(device=='gpu')
    # Exclude device initialization and compilation from per-puzzle budgets.
    import numpy as np
    ENGINE.run('make_values',1,np.array([1],dtype=np.int64),np.int32(1),np.int64(9),out=np.empty(4,dtype=np.int64))


def worker(task):
    k,p,old,keep,ports,gone,device=task
    result=search(ENGINE,p,old,keep,ports,gone,60)
    rows=[]
    for cap in BUDGETS:
        found=[(t,c) for t,c in result['wins'] if t<=cap]
        candidate=found[-1][1] if found else old
        length=replay(p,candidate)
        count=max((n for t,n in result['progress'] if t<=cap),default=0)
        rows.append([device,k,cap,len(old),length,len(old)-length,len(ports),count,min(cap,result['elapsed']),result['exhausted'] and result['elapsed']<=cap])
    if result['wins']:
        (OUT/f'joint-duration-hit-{device}-{k}.json').write_text(json.dumps(result['wins']))
    return rows


def banks():
    p=load_puzzles('puzzles.csv')
    return p,read_bank('public/ishankarnick/submission.csv',p),read_bank('submission.csv',p,strict=True)


def main():
    p,old,new=banks()
    changed=[k for k in p if len(old[k])>len(new[k])]
    assert len(changed)==91 and sum(len(old[k])-len(new[k]) for k in changed)==130
    for device in ('cpu','gpu'):
        with (OUT/f'joint-duration-{device}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['device','id','budget','old','verified','saved','ports','max_ports','active_seconds','exhausted'])
            with ProcessPoolExecutor(max_workers=8,initializer=worker_init,initargs=(device,)) as pool:
                futures=[pool.submit(worker,(k,p[k],old[k],*cut(p[k],old[k],new[k]),device)) for k in changed]
                for i,future in enumerate(as_completed(futures),1):
                    w.writerows(future.result());f.flush()
                    if i%15==0 or i==91:print(device,i,flush=True)

if __name__=='__main__':main()

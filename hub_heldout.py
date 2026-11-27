"""Measure retention on unidentified historical values, joining identities after each search."""
import csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from joint_duration import banks,cut,search,OUT,worker_init
import joint_duration as jd
from joint_near_repair import closure
from hub_beam import build_library,run
import time

def execute(task):
    p,old,keep,ports,gone,arm=task;start=time.perf_counter()
    if arm=='original':
        tracked=[set(),set()]
        r=search(jd.ENGINE,p,old,keep,ports,gone,10,tracker=tracked)
        gen,sur=tracked;wins=r['wins'];score=min([len(old),*[len(c) for _,c in wins]])
        leaves=0
    else:
        bank,ranks=build_library(p,keep,ports)
        if arm=='paired':
            results={}
            preparation=time.perf_counter()-start
            for name in ('legacy','new'):
                before=time.perf_counter()
                gen,sur,_,c,_,leaves=run(p,old,keep,ports,bank,ranks,name,layers=3)
                results[name]=dict(generated=sorted(gen),survived=sorted(sur),score=len(c),
                                   leaves=leaves,seconds=time.perf_counter()-before)
            return dict(id=p.id,preparation=preparation,**results)
        gen,sur,_,c,_,leaves=run(p,old,keep,ports,bank,ranks,arm,layers=3)
        score=len(c)
    free=closure(p,keep)
    return dict(id=p.id,arm=arm,generated=sorted(gen-free),survived=sorted(sur-free),score=score,leaves=leaves,seconds=time.perf_counter()-start)

def main(arm):
    P,O,N=banks();keys=[k for k in P if len(O[k])>len(N[k]) and k!='scored_0379']
    with (OUT/f'hub-heldout-{arm}.jsonl').open('w') as f,ProcessPoolExecutor(max_workers=6,initializer=worker_init,initargs=('cpu',)) as pool:
        futures=[pool.submit(execute,(P[k],O[k],*cut(P[k],O[k],N[k]),arm)) for k in keys]
        for i,j in enumerate(as_completed(futures),1):
            print(json.dumps(j.result()),file=f,flush=True)
            if i%15==0:print(arm,i,flush=True)
if __name__=='__main__':
    import sys
    main(sys.argv[1])

"""Same pre-rebuild pool for both rankers; inject the requested output identity only."""
import csv
from concurrent.futures import ProcessPoolExecutor
import random
import numpy as np
import joint_duration as jd
from joint_duration import banks,cut,OUT,worker_init
from joint_near_repair import closure
from near_reductions import rank,unfocused_rows,canonical
from shared_device_bench import sample_suffixes

def evaluate(job):
    p,free,blocked,records=job;engine=jd.ENGINE
    live=np.array(sorted(free),dtype=np.int64);n=len(live)
    raw=engine.run('make_values',n*n,live,np.int32(n),np.int64(p.limit),out=np.empty(4*n*n,dtype=np.int64))
    base=np.unique(np.concatenate((live,raw)));base=base[(base>0)&~np.isin(base,list(blocked))]
    rows=[]
    for step,t,op,a,b,lost in records:
        vals=base[base!=t];cost=(~np.isin(vals,live)).astype(np.int32);rr=unfocused_rows(cost)
        def exact(indices):
            helpers=vals[indices]
            if not len(indices):return []
            out=engine.run('frontier_links',len(indices),vals,cost,np.int32(len(vals)),helpers,np.int32(len(indices)),np.array([t],dtype=np.int64),np.int32(1),np.int64(p.limit),out=np.empty(len(indices),dtype=np.int32))
            seq=[]
            for j in np.flatnonzero(out>=0):
                qi,o=divmod(int(out[j]),6);h,k=int(helpers[j]),int(vals[qi])
                x,y=(k,h) if o in (2,5) else (h,k)
                seq.append(canonical(('+--*//'[o],x,y)))
            return seq
        truth=canonical((op,a,b))
        def position(seq):
            seq=list(dict.fromkeys(seq))
            return seq.index(truth)+1 if truth in seq else ''
        before=exact(np.arange(len(vals)))
        before.extend(canonical(s[0][2]) for s in sample_suffixes(t,free,p.limit,random.Random(781+step+int(p.id[-4:]))) if s and s[-1][1] not in blocked)
        after=exact(rr)
        after.extend(canonical(x[3]) for x in rank(engine,t,live,vals,cost,rr,p.limit,blocked|{t}))
        rows.append([p.id,step,t,lost,position(before),position(after)])
    return rows

def main():
    P,O,N=banks();groups={}
    for r in csv.DictReader((OUT/'value-ablation.csv').open()):
        if r['changed']=='1':
            groups.setdefault(r['id'],[]).append((int(r['step']),int(r['value']),r['op'],int(r['a']),int(r['b']),r['any_saving']))
    jobs=[]
    for k,rs in groups.items():
        keep,_,_=cut(P[k],O[k],N[k]);free=closure(P[k],keep)
        jobs.append((P[k],free,set(keep)-free,rs))
    with (OUT/'value-ablation-cut-ranks.csv').open('w',newline='') as f,ProcessPoolExecutor(max_workers=3,initializer=worker_init,initargs=('cpu',)) as pool:
        w=csv.writer(f);w.writerow(['id','step','value','any_saving','legacy_rank','new_rank'])
        for result in pool.map(evaluate,jobs):w.writerows(result);f.flush()
if __name__=='__main__':main()

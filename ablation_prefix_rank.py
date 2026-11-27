"""Rank canonical, changed, saving-critical recipes on their unmodified incoming live registers."""
import os
os.environ['OMP_THREAD_LIMIT']='1'
from concurrent.futures import ProcessPoolExecutor
import csv
import numpy as np
from joint_duration import banks,OUT,worker_init
import joint_duration as jd
from shared_device_bench import sample_suffixes
from near_reductions import canonical,rank,unfocused_rows
from value_ablation import apply
import random

def job(args):
    p,program,slots=args;engine=jd.ENGINE;regs=[*p.starts,*([0]*(p.registers-len(p.starts)))]
    output=[]
    for i,(op,a,b,d) in enumerate(program):
        x,y=regs[a],regs[b];t=apply(op,x,y)
        if i in slots:
            live=np.array(sorted(set(regs)-{0}),dtype=np.int64);n=len(live)
            raw=engine.run('make_values',n*n,live,np.int32(n),np.int64(p.limit),out=np.empty(4*n*n,dtype=np.int64))
            vals=np.unique(np.concatenate((raw,live))); vals=vals[(vals>0)&(vals!=t)]
            cost=(~np.isin(vals,live)).astype(np.int32)
            def exact(rows):
                helpers=vals[rows]
                hits=engine.run('frontier_links',len(rows),vals,cost,np.int32(len(vals)),helpers,np.int32(len(rows)),
                                np.array([t],dtype=np.int64),np.int32(1),np.int64(p.limit),out=np.empty(len(rows),dtype=np.int32))
                items=[]
                for j in np.flatnonzero(hits>=0):
                    qi,operator=divmod(int(hits[j]),6)
                    h,k=int(helpers[j]),int(vals[qi])
                    v,w=(k,h) if operator in (2,5) else (h,k)
                    items.append(canonical(('+--*//'[operator],v,w)))
                return items
            truth=canonical((op,x,y))
            def position(seq):
                recipes=list(dict.fromkeys(seq))
                return recipes.index(truth)+1 if truth in recipes else ''
            base=exact(np.arange(len(vals)))
            base.extend(canonical(s[0][2]) for s in sample_suffixes(t,set(map(int,live)),p.limit,random.Random(781+i+int(p.id[-4:]))) if s)
            rr=unfocused_rows(cost);modern=exact(rr)
            modern.extend(canonical(r[3]) for r in rank(engine,t,live,vals,cost,rr,p.limit,{t}))
            output.append([p.id,i,t,position(base),position(modern)])
        regs[d]=t
    return output

def main():
    P,_,N=banks();slots={}
    for r in csv.DictReader((OUT/'value-ablation.csv').open()):
        if r['changed']=='1' and r['any_saving']=='lost':
            slots.setdefault(r['id'],set()).add(int(r['step']))
    with (OUT/'value-ablation-prefix-ranks.csv').open('w',newline='') as f,ProcessPoolExecutor(max_workers=3,initializer=worker_init,initargs=('cpu',)) as pool:
        w=csv.writer(f);w.writerow(['id','step','value','legacy_prefix_rank','new_prefix_rank'])
        for r in pool.map(job,[(P[k],N[k],v) for k,v in slots.items()]):w.writerows(r);f.flush()
if __name__=='__main__':main()

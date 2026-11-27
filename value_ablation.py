"""Ablate the mathematical recipe in each instruction slot of each improved program."""
import csv
import json
import os
import subprocess
import tempfile
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shared_device_bench import load_puzzles,read_bank,replay,graph
from near_reductions import canonical
OUT=Path('experiments')


def apply(op,a,b):
    if op=='+':return a+b
    if op=='-':return a-b
    if op=='*':return a*b
    return a//b


def test(k,p,old,new):
    g=graph(p,old);delta=len(old)-len(new)
    regs=[*p.starts,*([0]*(p.registers-len(p.starts)))]
    live=set();writable={}
    for i in reversed(range(len(new))):
        op,a,b,d=new[i]
        writable[i]=sorted((set(range(p.registers))-live)|{d})
        live.discard(d);live.update((a,b))
    cases=[];inputs=[]
    for i,(op,a,b,d) in enumerate(new):
        x,y=regs[a],regs[b];v=apply(op,x,y)
        changed=v not in g or canonical((op,x,y))!=canonical(g[v])
        try:
            replay(p,new[:i]+new[i+1:]);zero=True
        except ValueError:zero=False
        record=dict(id=k,step=i,value=v,op=op,a=x,b=y,changed=int(changed),
                    target=int(v in p.targets),historical_gain=delta,minimum_splice='',excluded_through=0,
                    any_saving='',full_saving='',replacement='')
        if zero:
            record.update(minimum_splice=0,any_saving='survives',full_saving='survives',replacement='[]')
        else:
            cap=max(4,delta)
            inputs.append(' '.join(map(str,[i,len(regs),len(writable[i]),cap,p.limit,v,d,op,x,y,*regs,*writable[i]])))
        cases.append(record);regs[d]=v
    exe=Path(tempfile.gettempdir())/'santa-ablate-slot.exe'
    result=subprocess.run([exe],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True)
    for line in result.stdout.splitlines():
        s=line.split();i,d,n=map(int,s[:3]);r=cases[i];r['excluded_through']=d
        if n:
            ins=[[s[j],*map(int,s[j+1:j+4])] for j in range(3,len(s),4)]
            assert len(ins)==n and d==n-1
            length=replay(p,new[:i]+ins+new[i+1:])
            r.update(minimum_splice=n,any_saving='survives' if length<len(old) else 'lost',
                     full_saving='survives' if length<=len(new) else 'lost',replacement=json.dumps(ins,separators=(',',':')))
        else:
            r['any_saving']='lost' if d>=delta else 'unresolved'
            r['full_saving']='lost' if d>=1 else 'unresolved'
    # Deep equal-length plateaus can leave only the destination writable.
    from single_scratch import one_scratch
    regs=[*p.starts,*([0]*(p.registers-len(p.starts)))]
    for i,(op,a,b,dst) in enumerate(new):
        r=cases[i]
        if r['minimum_splice']=='' and op=='*' and writable[i]==[dst]:
            ins,depth=one_scratch(regs,dst,r['value'],p.limit,(op,regs[a],regs[b]))
            if ins:
                length=replay(p,new[:i]+ins+new[i+1:])
                r.update(minimum_splice=len(ins),excluded_through=len(ins)-1,
                         any_saving='survives' if length<len(old) else 'lost',
                         full_saving='survives' if length<=len(new) else 'lost',
                         replacement=json.dumps(ins,separators=(',',':')))
            else:r['excluded_through']=max(depth,r['excluded_through'])
        regs[dst]=r['value']
    assert all(r['any_saving'] for r in cases)
    return cases


def main():
    P=load_puzzles('puzzles.csv');old=read_bank('public/ishankarnick/submission.csv',P);new=read_bank('submission.csv',P)
    keys=[k for k in P if len(old[k])>len(new[k])]
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,group in enumerate(pool.map(lambda k:test(k,P[k],old[k],new[k]),keys),1):
            rows.extend(group)
            if i%20==0:print('ablated',i,flush=True)
    with (OUT/'value-ablation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('all',len(rows),Counter(r['any_saving'] for r in rows))
    print('changed',Counter(r['any_saving'] for r in rows if r['changed']))
    original=list(csv.DictReader((OUT/'near-miss-truth-ranks.csv').open()))
    valid={(r['id'],r['value']) for r in rows if r['changed'] and r['any_saving']=='lost'}
    subset=[r for r in original if (r['id'],int(r['target'])) in valid]
    print('target-ending subset',len(subset),'old',sum(bool(r['old_exact_then_backward_rank']) for r in subset),'new',sum(bool(r['new_exact_then_backward_rank']) for r in subset))
if __name__=='__main__':main()



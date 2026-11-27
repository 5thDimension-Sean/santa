"""Check whether a locally decisive recipe already has a whole-program counterexample."""
import csv
from collections import Counter
from joint_duration import OUT,banks
from shared_device_bench import graph,replay,visit,compile_graph,read_bank
from near_reductions import canonical
from optimize import SOURCES

def graft(p,old,new,t):
    # Resolve target dependencies preferring the improved program. Fall back to
    # the older program where the ablated producer or its inputs are unavailable.
    chosen={};required=set(p.starts);active=set()
    def resolve(v):
        if v in required or v in chosen:return True
        if v in active:return False
        active.add(v)
        for r in dict.fromkeys([new.get(v),old.get(v)]):
            if r is None or (v==t and canonical(r)==canonical(new[t])):continue
            if all(resolve(x) for x in r[1:]):
                chosen[v]=r
                active.remove(v);return True
        active.remove(v);return False
    if not all(resolve(v) for v in p.targets):return
    order=visit(p.targets,chosen)
    cand=compile_graph(p,chosen,order)
    if cand is not None:replay(p,cand)
    return cand

def main():
    P,O,N=banks();R=list(csv.DictReader((OUT/'value-ablation.csv').open()))
    keys={x['id'] for x in R};alternatives={k:[] for k in keys}
    for file in SOURCES:
        B=read_bank(file,P)
        for k in keys:
            if len(B[k])>=len(O[k]):continue
            alt={v:canonical(r) for v,r in graph(P[k],B[k]).items()}
            if all(alt!=g for _,_,g in alternatives[k]):alternatives[k].append((file,len(B[k]),alt))
    graphs={k:(graph(P[k],O[k]),graph(P[k],N[k])) for k in keys}
    output=[]
    for row in R:
        if row['changed']!='1' or row['any_saving']!='lost':continue
        k=row['id'];v=int(row['value']);old,new=graphs[k]
        alternatives_here=[(n,file) for file,n,g in alternatives[k] if g.get(v)!=canonical(new[v])]
        hybrid=graft(P[k],old,new,v)
        if hybrid is not None:
            check=graph(P[k],hybrid)
            assert v not in check or canonical(check[v])!=canonical(new[v])
            if len(hybrid)<len(O[k]):alternatives_here.append((len(hybrid),'two-checkpoint graft'))
        n,file=min(alternatives_here,default=('', ''))
        output.append([k,row['step'],v,row['target'],file,n])
    with (OUT/'value-ablation-broader-check.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','step','value','target','counterexample','verified_score']);w.writerows(output)
    print('positive bypasses',sum(bool(x[-2]) for x in output),'of',len(output))
if __name__=='__main__':main()

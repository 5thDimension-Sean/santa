"""Collect the original oracle's arithmetic candidates with its per-output budget disabled."""
import csv,json,random
from collections import Counter
from joint_duration import banks,OUT
from near_rank_audit import available_table
from near_reductions import canonical
from shared_device_bench import Engine,graph,repairs,visit

def main():
    p,old,new=banks()
    eng=Engine(False)
    rows=[r for r in csv.DictReader((OUT/'reroute-oracle-failures.csv').open())
          if r['case_reason'] in ('budget','patch over allowance')]
    assert len(rows)==705
    hist=Counter()
    with (OUT/'shared-route-candidates.jsonl').open('w') as f:
        for row in rows:
            k=row['id'];t=int(row['target']);g=graph(p[k],old[k])
            truth=graph(p[k],new[k])[t]
            _,_,ban,_,vals,_=available_table(eng,p[k],g,t)
            pool=set(map(int,vals));found=set()
            def collect(patch):
                # Keep only nodes on the forced root's own route.
                order=visit([t],patch)
                sig=tuple(sorted((v,*canonical(patch[v])) for v in order))
                found.add(sig)
            for v,x in ((truth[1],truth[2]),(truth[2],truth[1])):
                if x not in pool or v in ban:continue
                list(repairs(eng,p[k],old[k],g,(t,),float('inf'),[0,0],random.Random(1),
                             ((t,v,truth),),collect_patch=collect))
            print(json.dumps([k,t,row['case_reason'],sorted(found)],separators=(',',':')),file=f)
            hist['cases']+=1;hist['cases_with_candidates']+=bool(found);hist['routes']+=len(found)
    print(hist)
if __name__=='__main__':main()

"""Recount actual calls as well as eligible target cases in the injection audit."""
from joint_duration import banks,OUT
from near_rank_audit import available_table
from shared_device_bench import Engine,graph,repairs,replay
from near_reductions import canonical
import random,csv

def main():
    P,O,N=banks();e=Engine(False)
    previous={(r['id'],int(r['target'])):r for r in csv.DictReader((OUT/'near-miss-truth-ranks.csv').open())}
    with (OUT/'oracle-injection-calls.csv').open('w',newline='') as f:
        out=csv.writer(f);out.writerow(['id','target','calls','accepted','gross_savings'])
        for k,p in P.items():
            if len(O[k])<=len(N[k]):continue
            g=graph(p,O[k]);h=graph(p,N[k])
            for t in p.targets:
                if (k,t) not in previous:continue
                _,free,forbidden,_,vals,_=available_table(e,p,g,t)
                pool=set(map(int,vals));r=h[t];calls=0;accepted=False;gain=0
                for v,x in ((r[1],r[2]),(r[2],r[1])):
                    if x not in pool or v in forbidden:continue
                    calls+=1
                    cand=next(repairs(e,p,O[k],g,(t,),float('inf'),[0,0],random.Random(1),((t,v,r),)),None)
                    if cand is not None:
                        gain=len(O[k])-replay(p,cand);accepted=True;break
                assert accepted==(previous[k,t]['oracle_result']=='replayed')
                out.writerow([k,t,calls,int(accepted),gain])
    result=list(csv.DictReader((OUT/'oracle-injection-calls.csv').open()))
    print('calls',sum(int(r['calls']) for r in result),'eligible',sum(int(r['calls'])>0 for r in result),'accepted',sum(int(r['accepted']) for r in result),'positive',sum(int(r['gross_savings'])>0 for r in result))
if __name__=='__main__':main()

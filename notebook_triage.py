"""Replay the notebook's top-25 hypothesis against matched local controls."""
import csv
import json
import math
from pathlib import Path
import random
import re
import sys
from itertools import combinations

from optimize import SOURCES, graph, load_puzzles, read_bank
from region_milp import ancestors, proposals

def main(notebook):
    cells=json.loads(Path(notebook).read_text(encoding='utf-8-sig'))['cells']
    table=''.join(''.join(o.get('data',{}).get('text/html',[])) for o in cells[11]['outputs'])
    prioritized=list(dict.fromkeys(re.findall(r'scored_\d{4}',table)))
    if len(prioritized)!=25: raise ValueError('Expected the saved top-25 table')
    puzzles=load_puzzles('puzzles.csv')
    best=read_bank('submission.csv',puzzles,strict=True)
    older=read_bank('public/ishankarnick/submission.csv',puzzles)
    banks=[read_bank(s,puzzles) for s in SOURCES]
    with open('public/goodpjw2008/families.csv',newline='') as f:
        labels={r['id']:r['fam'] for r in csv.DictReader(f)}
    reserved=set(prioritized)
    trials=[]
    for i,k in enumerate(prioritized):
        candidates=[j for j in puzzles if j not in reserved and labels[j]==labels[k]]
        control=min(candidates,key=lambda j:(abs(len(best[j])-len(best[k])),abs(len(puzzles[j].targets)-len(puzzles[k].targets)),j))
        reserved.add(control)
        trials.extend([(i+1,'priority',k),(i+1,'control',control)])
    rows=[]
    for rank,arm,k in trials:
        p=puzzles[k]; g=graph(p,best[k]); old=len(best[k])
        ancestry={v:ancestors([v],g) for v in p.targets}
        def score(pair):
            a,b=pair; d=math.gcd(a,b)
            close=max(0,b.bit_length()-(b-a).bit_length())
            weight=max(d.bit_length()-.3*(a//d).bit_length()-.3*(b//d).bit_length(),close)
            if a in ancestry[b] or b in ancestry[a] or d in ancestry[a]&ancestry[b]: weight-=20
            return weight
        pairs=sorted(combinations(p.targets,2),key=score,reverse=True)
        donors=[]; seen=set()
        for bank in banks:
            sig=tuple(map(tuple,bank[k]))
            if sig not in seen: donors.append(bank[k]);seen.add(sig)
        for idx,(a,b) in enumerate(pairs[:4]):
            group={a,b}
            for u,v in pairs:
                if (u in group) != (v in group):
                    group.update((u,v))
                    if len(group)>=4: break
            rng=random.Random(96521+rank*11+idx)
            for candidate in proposals(p,best[k],donors,rng,group):
                best[k]=candidate
                dest=Path('experiments')/f'region-triage-{k}.json'
                pending=dest.with_suffix('.pending')
                pending.write_text(json.dumps({'id':k,'program':candidate,'origin':f'local notebook triage rank {rank}, {arm}'}))
                pending.replace(dest)
        rows.append([rank,arm,k,old,len(older[k])-old,old-len(best[k])])
        if len(rows)%10==0: print('Tested',len(rows),'of',len(trials),flush=True)
    with open('experiments/notebook-triage-test.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['rank','arm','id','start_steps','historical_savings','new_savings']);w.writerows(rows)
    for arm in ('priority','control'):
        group=[r for r in rows if r[1]==arm]
        print(arm,len(group),'puzzles','historical',sum(r[4] for r in group),'new',sum(r[5] for r in group))

if __name__=='__main__': main(sys.argv[1])

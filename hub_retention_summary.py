import csv,json
from collections import defaultdict
from joint_duration import banks,cut,OUT
from joint_near_repair import closure
from statistics import median
def summarize():
    P,O,N=banks();labels=defaultdict(set)
    for r in csv.DictReader((OUT/'value-ablation-broader-check.csv').open()):
        if not r['counterexample'] and r['target']=='0' and r['id']!='scored_0379':
            labels[r['id']].add(int(r['value']))
    original={r['id']:r for l in (OUT/'hub-heldout-original.jsonl').read_text().splitlines() if (r:=json.loads(l))}
    paired={r['id']:r for l in (OUT/'hub-heldout-paired.jsonl').read_text().splitlines() if (r:=json.loads(l))}
    path=OUT/'hub-heldout-final.jsonl'
    final={r['id']:r for l in path.read_text().splitlines() if (r:=json.loads(l))} if path.exists() else {}
    assert len(original)==len(paired)==90
    rows=[]
    for k in original:
        p=P[k];keep,ports,_=cut(p,O[k],N[k]);start=closure(p,keep)
        eligible=labels[k]-start
        row={'id':k,'eligible':len(eligible),'initial_excluded':len(start&labels[k])}
        for name,arm in [('original',original[k]),('matched',paired[k]['legacy']),('screen',paired[k]['new']),('final',final.get(k))]:
            if not arm:continue
            for field in ['generated','survived']:
                row[name+'_'+field]=len(eligible&set(arm[field]))
            row[name+'_score']=arm['score']
        rows.append(row)
    print('denom',sum(r['eligible'] for r in rows),'puzzles',len(rows),flush=True)
    for n in ['original','matched','screen','final']:
        if n+'_survived' not in rows[0] or any(n+'_survived' not in r for r in rows):continue
        print(n,*(sum(r[n+'_'+f] for r in rows) for f in ['generated','survived']),sum(r[n+'_survived']>0 for r in rows),'puzzles with retention; wins',sum(len(O[r['id']])-r[n+'_score'] for r in rows))
    if len(final)==90:
        keys=list(rows[0])
        with (OUT/'hub-retention-summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(sorted(rows,key=lambda r:r['id']))
        print('paired signs',[(c,sum((r['final_survived']-r['matched_survived'])*c>0 for r in rows)) for c in (1,-1)])
if __name__=='__main__':summarize()

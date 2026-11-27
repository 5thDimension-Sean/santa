"""Verify focal replay and join blind search traces to ablation labels."""
import csv,json
from statistics import median
from hub_retention_summary import summarize
from joint_duration import banks,cut,OUT
from shared_device_bench import replay

def main():
    summarize()
    P,O,N=banks();labels={}
    for row in csv.DictReader((OUT/'value-ablation-broader-check.csv').open()):
        if row['target']=='0' and not row['counterexample']:
            labels.setdefault(row['id'],set()).add(int(row['value']))
    ent={r['id']:r for r in csv.DictReader((OUT/'hub-retention-summary.csv').open())}
    records=[json.loads(x) for x in (OUT/'hub-heldout-live.jsonl').read_text().splitlines()]
    assert len(records)==180 and len({(r['id'],r['enhanced']) for r in records})==180
    indexed=OUT/'hub-heldout-indexed.jsonl'
    if indexed.exists():
        extra=[json.loads(x) for x in indexed.read_text().splitlines()]
        assert len(extra)==90
        records+=extra
    stat={}
    for r in records:
        k=r['id']; truth=labels.get(k,set()); row=ent[k]
        name={False:'time_original',True:'time_augmented','indexed':'time_indexed'}[r['enhanced']]
        assert len(truth)==int(row['eligible']) and set(r['survived'])<=set(r['generated'])
        for col in ['generated','survived']:row[name+'_'+col]=len(truth&set(r[col]))
        row[name+'_score']=r['score'];row[name+'_seconds']=round(r['seconds'],4)
        winner=OUT/f"hub-live-winner-{r['enhanced']}-{k}.json"
        program=json.loads(winner.read_text()) if r['score']<len(O[k]) else O[k]
        assert r['score']==replay(P[k],program)
        stat.setdefault(name,[]).append(row)
    names=list(stat)
    for name,rs in stat.items():
        assert len(rs)==90
        print(name,'generated',sum(r[name+'_generated'] for r in rs),'survived',sum(r[name+'_survived'] for r in rs),
              'median wall',median(r[name+'_seconds'] for r in rs))
    with (OUT/'hub-retention-comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(next(iter(ent.values()))));w.writeheader();w.writerows(ent.values())
    focal=json.loads((OUT/'hub-focal-verified.json').read_text())
    k='scored_0379';assert replay(P[k],focal['program'])==len(N[k])==108<len(O[k])==109
    assert focal['witness']['score']==108 and len(focal['witness']['missing'])==1
    assert sum(replay(P[k],N[k]) for k in P)==127311
    print('confirmed baseline and focal program')
if __name__=='__main__':main()

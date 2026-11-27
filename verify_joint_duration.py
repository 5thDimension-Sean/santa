"""Independently replay splice witnesses and verify benchmark coverage and provenance."""
import csv,json,statistics
from pathlib import Path
from joint_duration import banks,OUT
from shared_device_bench import replay
from near_reductions import canonical
from value_ablation import apply

def read(name):return list(csv.DictReader((OUT/name).open()))

def main():
    P,O,N=banks();expected={k for k in P if len(O[k])>len(N[k])}
    ablations=read('value-ablation.csv')
    assert len(ablations)==sum(len(N[k]) for k in expected)
    seen=set()
    for row in ablations:
        k=row['id'];i=int(row['step']);assert (k,i) not in seen;seen.add((k,i))
        ins=json.loads(row['replacement']); assert len(ins)==int(row['minimum_splice'])
        prog=N[k];reg=list(P[k].starts)+[0]*(P[k].registers-len(P[k].starts))
        for op,a,b,d in prog[:i]:reg[d]=apply(op,reg[a],reg[b])
        ban=canonical((row['op'],int(row['a']),int(row['b'])))
        for op,a,b,d in ins:
            assert canonical((op,reg[a],reg[b]))!=ban
            reg[d]=apply(op,reg[a],reg[b])
        length=replay(P[k],prog[:i]+ins+prog[i+1:])
        assert (length<len(O[k]))==(row['any_saving']=='survives')
    provenance={r['id']:r for r in csv.DictReader(Path('submission.sources.csv').open())}
    assert set(provenance)==set(P) and all(int(provenance[k]['instructions'])==len(N[k]) for k in P)
    rows=[]
    for dev in ('cpu','gpu'):
        records=read(f'joint-duration-{dev}.csv')
        for cap in (1,10,60):
            part=[r for r in records if int(r['budget'])==cap]
            assert len(part)==91 and {r['id'] for r in part}==expected
            for r in part:
                k=r['id'];prog=O[k];path=OUT/f'joint-duration-hit-{dev}-{k}.json'
                if path.exists():
                    for when,c in json.loads(path.read_text()):
                        if when<=cap:prog=c
                assert replay(P[k],prog)==int(r['verified']) and len(O[k])-len(prog)==int(r['saved'])
            rows.append([dev,cap,len(part),sum(int(r['saved']) for r in part),
                         sum(r['max_ports']==r['ports'] for r in part),
                         sum(r['exhausted']=='True' for r in part),
                         sum(int(r['max_ports']) for r in part),sum(int(r['ports']) for r in part)])
    with (OUT/'joint-duration-summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['device','seconds','puzzles','verified_saved','all_outputs','frontier_exhausted','total_max_outputs','total_required_outputs']);w.writerows(rows)
    print(rows)
if __name__=='__main__':main()

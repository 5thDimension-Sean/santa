"""Independently reconcile the finished comparisons, winning witnesses, and provenance."""
import csv
import json
from collections import Counter
from pathlib import Path

from joint_duration import banks, OUT
from optimize import graph, read_bank, replay

def main():
    puzzles,old,merged=banks()
    assert sum(map(len,merged.values()))==127311
    final=read_bank(OUT/'whole-hub-submission.csv',puzzles,strict=True)
    report={}
    historical={k for k in puzzles if len(old[k])>len(merged[k])}
    assert len(historical)==91
    for arm,expected in [('focal',{'scored_0379'}),('heldout',historical-{'scored_0379'}),('baseline',set(puzzles))]:
        source=merged if arm=='baseline' else old
        rows=list(csv.DictReader((OUT/f'whole-hub-{arm}.csv').open()))
        assert len(rows)==len(expected) and {r['id'] for r in rows}==expected
        path=OUT/f'whole-hub-{arm}-wins.jsonl'
        wins=dict(json.loads(s) for s in path.read_text().splitlines()) if path.exists() else {}
        lengths=[]
        for r in rows:
            k=r['id']; saved=int(r['saved'])
            assert len(source[k])==int(r['original'])
            assert (k in wins)==bool(saved)
            prog=wins.get(k,source[k])
            n=replay(puzzles[k],prog)
            assert n==int(r['final']) and len(source[k])-n==saved
            if arm=='baseline':assert prog==final[k]
            else:assert int(r['recovered'])==min(saved,len(old[k])-len(merged[k]))
            lengths.append(n)
        if arm=='focal':
            k='scored_0379';g=graph(puzzles[k],wins[k])
            assert 787994 in g and 787994 not in graph(puzzles[k],old[k]) and lengths==[108]
        report[arm]=dict(puzzles=len(rows),improved=len(wins),steps=sum(int(r['saved']) for r in rows),
                         recovered=sum(int(r['recovered']) for r in rows),
                         historical_gap=sum(int(r['historical_gap']) for r in rows),
                         cpu_hours=round(sum(float(r['cpu_seconds']) for r in rows)/3600,4),
                         max_cpu_seconds=max(float(r['cpu_seconds']) for r in rows),
                         score=sum(lengths))
    source_file=OUT/'whole-hub-submission.sources.csv'
    origins=list(csv.DictReader(source_file.open()))
    assert len(origins)==1000
    for r in origins:assert int(r['instructions'])==len(final[r['id']])
    failures=list(csv.DictReader((OUT/'reroute-oracle-failures.csv').open()))
    report['eligible_failures']=dict(Counter(r['case_reason'] for r in failures))
    assert len(failures)==1018
    print(json.dumps(report,indent=2))
    (OUT/'whole-hub-results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

"""Benchmark the same whole-program move on each bank. History is used only for evaluation."""
import argparse
import csv
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from joint_duration import banks, OUT
from optimize import graph, compile_graph, replay, save


def search(p, original, budget):
    program = original
    used = singles = pairs = proposed = pressure = passes = finishers = 0
    witnesses = []
    while used + 0.1 < budget:
        tick = time.thread_time()
        rules = graph(p, program)
        seeds = list(dict.fromkeys(p.starts))
        lines = [f'{budget-used} {p.limit} {len(seeds)} {len(p.targets)} {len(rules)}',
                 ' '.join(map(str, seeds)), ' '.join(map(str, p.targets))]
        lines.extend(f'{v} {op} {x} {y}' for v,(op,x,y) in rules.items())
        exe = Path(tempfile.gettempdir()) / 'santa-whole-hub.exe'
        result = subprocess.run([exe],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
        rows = iter(result.stdout.splitlines())
        best = program
        chosen = None
        native = None
        for row in rows:
            fields = row.split()
            if fields[0]=='T':
                native = float(fields[1])
            elif fields[0]=='S':
                singles += int(fields[4])
                pairs += int(fields[5])
            elif fields[0]=='F':
                finishers += int(fields[1])
            elif fields[0]=='C':
                n,a,b = map(int,fields[1:])
                proposal = {}
                for _ in range(n):
                    v,op,x,y = next(rows).split()
                    proposal[int(v)] = (op,int(x),int(y))
                proposed += 1
                if n>=len(best):continue
                candidate = compile_graph(p,proposal,list(proposal))
                if candidate is None:
                    pressure += 1
                    continue
                assert replay(p,candidate)==n
                best = candidate
                chosen = [a,b,sorted(set(proposal)-rules.keys())]
        assert native is not None
        passes += 1
        used += native + time.thread_time()-tick
        if chosen is None:break
        witnesses.append(chosen)
        program = best
    return program,[len(original),len(program),len(original)-len(program),round(used,3),passes,singles,pairs,finishers,proposed,pressure,json.dumps(witnesses)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('arm',choices=['focal','heldout','baseline'])
    parser.add_argument('--workers',type=int,default=10)
    args=parser.parse_args()
    p,old,baseline=banks()
    keys=([k for k in p if len(old[k])>len(baseline[k]) and k!='scored_0379']
          if args.arm=='heldout' else list(p) if args.arm=='baseline' else ['scored_0379'])
    source = baseline if args.arm=='baseline' else old
    best=dict(source)
    origins={r['id']:r['origin'] for r in csv.DictReader(open('submission.sources.csv'))}
    filename=OUT/f'whole-hub-{args.arm}.csv'
    with filename.open('w',newline='') as stream, ThreadPoolExecutor(args.workers) as pool:
        w=csv.writer(stream)
        w.writerow('id original final saved cpu_seconds passes singles pairs four_step_leaves candidates allocation_failures accepted_hubs_and_new_values historical_gap recovered'.split())
        futures={pool.submit(search,p[k],source[k],180):k for k in keys}
        for index,f in enumerate(as_completed(futures),1):
            k=futures[f]
            prog,row=f.result()
            best[k]=prog
            gap=len(old[k])-len(baseline[k])
            w.writerow([k,*row,gap,min(gap,row[2]) if args.arm!='baseline' else 0])
            stream.flush()
            if row[2]:
                assert replay(p[k],prog)==len(prog)
                with (OUT/f'whole-hub-{args.arm}-wins.jsonl').open('a') as file:
                    print(json.dumps([k,prog]),file=file)
                origins[k]+=' + whole-program hub reroute'
            if index%100==0 or args.arm=='focal':print(index,k,row[:10],flush=True)
    if args.arm=='baseline':
        score=save(OUT/'whole-hub-submission.csv',p,best,origins)
        print('verified',score,flush=True)
if __name__=='__main__':main()

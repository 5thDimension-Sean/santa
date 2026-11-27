"""Exhaustive snapshot audit and full-program replay for the 127311 baseline."""
import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from audit_cones import reachable
from optimize import load_puzzles, read_bank, replay
from elf import write_submission


def calc(op, a, b):
    if op == '+': return a+b
    if op == '-': return a-b
    if op == '*': return a*b
    assert a % b == 0
    return a//b


def trace(p, prog):
    regs = [-i-1 for i in range(len(p.starts))] + [None]*(p.registers-len(p.starts))
    values = {-i-1: v for i, v in enumerate(p.starts)}
    missing = set(p.targets)-set(p.starts)
    snaps, ops, roots = [], [], []
    uses = Counter()
    for k, (op, i, j, d) in enumerate(prog):
        snaps.append(tuple(regs))
        x, y = regs[i], regs[j]
        ops.append((op, x, y))
        uses.update({x,y})
        v = values[k] = calc(op, values[x], values[y])
        regs[d] = k
        if v in missing:
            missing.remove(v)
            roots.append(k)
    branches = []
    targets = set(p.targets)
    for root in roots:
        removed = Counter()
        stack, cone = [root], set()
        while stack:
            k = stack.pop()
            cone.add(k)
            for dep in set(ops[k][1:]):
                if dep < 0: continue
                removed[dep] += 1
                if removed[dep] == uses[dep] and values[dep] not in targets:
                    stack.append(dep)
        if len(cone) >= 4:
            snapshot = {values[w]: w for w in snaps[min(cone)] if w is not None}
            branches.append((root, cone, snapshot))
    return values, ops, branches


def schedule(p, ops, patches):
    replaced = set()
    insert = {}
    for root, cone, snapshot, witness in patches:
        assert not replaced.intersection(cone)
        replaced.update(cone)
        live = dict(snapshot)
        block = []
        for j, (op, a, b) in enumerate(witness):
            k = root if j == len(witness)-1 else (root, j)
            block.append((k, op, live[a], live[b]))
            live[calc(op,a,b)] = k
        insert[min(cone)] = block
    ordered = []
    for k, (op, a, b) in enumerate(ops):
        ordered.extend(insert.get(k, ()))
        if k not in replaced:
            ordered.append((k,op,a,b))
    uses = Counter(x for _,_,a,b in ordered for x in (a,b))
    slots = {-i-1:i for i in range(len(p.starts)) if uses[-i-1]}
    free = [i for i in range(p.registers) if i not in slots.values()]
    program = []
    for k,op,a,b in ordered:
        if a not in slots or b not in slots: return None
        i,j = slots[a], slots[b]
        for x in (a,b):
            uses[x] -= 1
            if not uses[x]: free.append(slots.pop(x))
        if not free: return None
        d = free.pop()
        program.append([op,i,j,d])
        if uses[k]: slots[k] = d
        else: free.append(d)
    replay(p,program)
    return program


def main(exe, baseline, depth=3, workers=1):
    label = {3:'three', 4:'four'}[depth]
    puzzles = load_puzzles('puzzles.csv')
    progs = read_bank(baseline,puzzles,strict=True)
    assert sum(map(len,progs.values())) == 127311
    cases, requests, traces = [], [], {}
    for key,p in puzzles.items():
        values, ops, branches = trace(p,progs[key])
        traces[key] = ops
        for root, cone, snap in branches:
            if len(cone) <= depth: continue
            t = values[root]
            assert reachable(snap,t,p.limit) == 3
            idx = len(cases)
            requests.append(' '.join(map(str,[idx,p.limit,t,len(snap),*snap])))
            cases.append((key,t,root,cone,snap))
    assert len(cases) == {3:4866, 4:2312}[depth]
    if depth == 4:
        with open('experiments/three-step-audit.csv', newline='') as stream:
            prior = {(r['id'], int(r['root_index']), int(r['snapshot_index']), int(r['target']))
                     for r in csv.DictReader(stream)
                     if int(r['old_steps']) > 4 and int(r['witness_count']) == 0}
        assert prior == {(key, root, min(cone), t) for key,t,root,cone,_ in cases}
    def run(batch):
        return subprocess.run([exe], input='\n'.join(batch)+'\n', text=True,
                              capture_output=True, check=True).stdout
    batches = [requests[i:i+16] for i in range(0,len(requests),16)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        output = []
        for i,result in enumerate(pool.map(run,batches)):
            output.append(result)
            if (i+1)%16 == 0: print('searched',16*(i+1),flush=True)
    raw = ''.join(output)
    witnesses = defaultdict(set)
    for line in raw.splitlines():
        idx,*parts = line.split()
        witness = tuple((parts[i],int(parts[i+1]),int(parts[i+2])) for i in range(0,3*depth,3))
        witnesses[int(idx)].add(witness)
    accepted = defaultdict(list)
    best = dict(progs)
    rows = []
    for idx in sorted(range(len(cases)),key=lambda i:-len(cases[i][3])):
        key,t,root,cone,snap = cases[idx]
        p = puzzles[key]
        standalone = combined = 0
        for w in sorted(witnesses[idx]):
            available = set(snap)
            for op,a,b in w:
                assert a in available and b in available
                v = calc(op,a,b)
                assert 1 <= v <= p.limit
                available.add(v)
            assert v == t
            patch = (root,cone,snap,w)
            isolated = schedule(p,traces[key],[patch])
            if isolated is None: continue
            standalone = len(progs[key])-len(isolated)
            together = schedule(p,traces[key],accepted[key]+[patch])
            if together is not None:
                combined = len(best[key])-len(together)
                assert combined == standalone == len(cone)-depth
                best[key] = together
                accepted[key].append(patch)
                break
        rows.append((idx,key,root,min(cone),len(cone),t,len(snap),len(witnesses[idx]),
                     standalone,combined))
    dest = Path(f'experiments/{label}-step-audit.csv')
    with dest.open('w',newline='') as stream:
        out = csv.writer(stream)
        out.writerow(['case','id','root_index','snapshot_index','old_steps','target',
                      'snapshot_values','witness_count','isolated_savings','combined_savings'])
        out.writerows(sorted(rows))
    score = sum(replay(puzzles[k],v) for k,v in best.items())
    if score < 127311:
        dest = Path(f'experiments/{label}-step-submission.csv')
        write_submission(dest,puzzles,best)
        assert sum(map(len,read_bank(dest,puzzles,strict=True).values())) == score
    print('baseline_sha256',hashlib.sha256(Path(baseline).read_bytes()).hexdigest())
    print('cases',len(rows),'hits',sum(r[-3]>0 for r in rows),'isolated',
          sum(r[-2]>0 for r in rows),'combined',sum(r[-1]>0 for r in rows),
          'saved',sum(r[-1] for r in rows),'score',score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('executable')
    parser.add_argument('--baseline',default='submission.csv')
    parser.add_argument('--depth',type=int,choices=(3,4),default=3)
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args()
    main(args.executable,args.baseline,args.depth,args.workers)

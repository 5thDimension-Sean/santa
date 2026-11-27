"""Merge public programs, splice arithmetic graphs, and replay the scored CSV."""
import argparse
import csv
import json
import os
import subprocess
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sys import path
path.insert(0, str(Path(__file__).parent / 'public/goodpjw2008'))
from elf import load_puzzles, replay
from pfmt import encode, parse_results

SOURCES = [
    'public/frontier-nov27/submission.csv', 'public/viktorbarbarich/best_programs.csv',
    'public/b05204047elias/submission.csv',
    'public/ishankarnick/submission.csv', 'public/goodpjw2008/our_programs.csv',
    'public/goodpjw2008/public_frontier.csv', 'public/alexchilton/rudolph_programs.csv',
    'public/asterios/best_programs.csv', 'public/farshid/submission.csv',
    'public/asterios/public_day_ago.csv', 'public/goodpjw2008/frontier_22nov.csv',
    'public/goodpjw2008/frontier_20nov.csv',
]

def read_bank(filename, puzzles, strict=False):
    programs = {}
    with open(filename, newline='', encoding='utf-8-sig') as stream:
        rows = csv.DictReader(stream)
        if strict and rows.fieldnames != ['id', 'program']:
            raise ValueError('Incorrect submission header')
        for r in rows:
            key = r['id']
            if key in programs or key not in puzzles or None in r:
                raise ValueError(f'Invalid CSV row for {key}')
            if len(r['program'].encode()) > 131072:
                raise ValueError('Oversize JSON cell')
            prog = json.loads(r['program'])
            if not isinstance(prog, list):
                raise ValueError('Expected JSON array')
            replay(puzzles[key], prog)
            programs[key] = prog
    if programs.keys() != puzzles.keys():
        raise ValueError(f'Incomplete program bank {filename}')
    return programs

def save(output, puzzles, best, origins):
    dest = Path(output)
    temp = dest.with_suffix('.pending.csv')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        w = csv.writer(stream)
        w.writerow(['id', 'program'])
        for key, prog in best.items():
            w.writerow([key, json.dumps(prog, separators=(',', ':'))])
    checked = read_bank(temp, puzzles, strict=True)
    os.replace(temp, dest)
    with dest.with_suffix('.sources.csv').open('w', newline='', encoding='utf-8') as stream:
        w = csv.writer(stream)
        w.writerow(['id', 'instructions', 'origin'])
        for key in best:
            w.writerow([key, len(best[key]), origins[key]])
    return sum(map(len, checked.values()))

def graph(p, prog):
    regs = list(p.starts) + [None] * (p.registers-len(p.starts))
    nodes = dict.fromkeys(p.starts)
    for op, i, j, d in prog:
        x, y = regs[i], regs[j]
        if op == '+': v = x + y
        elif op == '-': v = x - y
        elif op == '*': v = x * y
        else: v = x // y
        nodes.setdefault(v, (op,x,y))
        regs[d] = v
    return {v:r for v,r in nodes.items() if r is not None}

def compile_graph(p, graph, order):
    remaining = Counter(x for v in order for x in graph[v][1:])
    regs = {v:i for i,v in enumerate(p.starts) if remaining[v]}
    free = [i for i in range(p.registers) if i not in regs.values()]
    prog = []
    for v in order:
        op, x, y = graph[v]
        a,b = regs[x],regs[y]
        for operand in (x,y):
            remaining[operand] -= 1
            if not remaining[operand]:
                free.append(regs.pop(operand))
        if not free: return None
        d = free.pop()
        prog.append([op,a,b,d])
        if remaining[v]: regs[v] = d
        else: free.append(d)
    return prog

def graft(p, prog, donors):
    current = graph(p,prog)
    starts = set(p.starts)
    for label, donor_prog in donors:
        donor = graph(p,donor_prog)
        roots = [v for v in current if v in donor and donor[v] != current[v]]
        for root in reversed(roots):
            if root not in current: continue
            for depth in (1, 3, 10000):
                patch = {}
                stack = [(root,depth)]
                while stack:
                    v, level = stack.pop()
                    if v in patch or v in starts or (level <= 0 and v in current): continue
                    patch[v] = donor[v]
                    stack.extend((x,level-1) for x in donor[v][1:])
                nodes = current | patch
                seen, active, order = set(starts), set(), []
                def visit(v):
                    if v in seen: return True
                    if v in active or len(order) >= len(prog)-1: return False
                    active.add(v)
                    if not all(visit(x) for x in nodes[v][1:]): return False
                    active.remove(v)
                    seen.add(v)
                    order.append(v)
                    return True
                if not all(visit(v) for v in p.targets) or len(order) >= len(prog): continue
                candidate = compile_graph(p,nodes,order)
                if candidate is None: continue
                replay(p,candidate)
                prog = candidate
                current = {v:nodes[v] for v in order}
                yield prog,label
                break

def run_native(exe, seconds, seed, p, prog):
    run = subprocess.run([exe, '--time', str(seconds), '--seed', str(seed),
                          '--cap','2','--T','0.2','--rb','0.7','--deep','0.2'],
                         input=encode(p,prog), capture_output=True, text=True, check=True)
    if p.id+' ' not in run.stderr or ' skipped' in run.stderr or 'closure-fail' in run.stderr:
        raise RuntimeError(run.stderr[-300:])
    return parse_results(run.stdout).get(p.id)

def run_cross(exe, seconds, seed, p, programs, pack=False):
    programs = list(dict.fromkeys(tuple(map(tuple,prog)) for prog in programs))
    lines = [f'{p.limit} {len(p.starts)} {len(p.targets)} {len(programs)}',
             ' '.join(map(str,p.starts)), ' '.join(map(str,p.targets))]
    for prog in programs:
        lines.append(str(len(prog)))
        lines.extend('%s %d %d %d' % row for row in prog)
    cmd = [exe,str(seconds),str(seed+int(p.id[-4:]))]
    if pack: cmd.append('--pack')
    run = subprocess.run(cmd, input='\n'.join(lines)+'\n',
                         capture_output=True,text=True,check=True)
    prog = []
    for line in run.stdout.splitlines():
        op,a,b,d = line.split()
        prog.append([op,int(a),int(b),int(d)])
    return prog or None

def exact_small(p):
    # With at most 16 possible values, all distinct values fit in registers.
    # Rewriting a value already seen cannot shorten any program in this search.
    initial = sum(1<<x for x in set(p.starts))
    needed = sum(1<<x for x in set(p.targets))
    queue, parent = deque([initial]), {initial:None}
    while queue:
        mask = queue.popleft()
        if mask & needed == needed:
            steps = []
            while parent[mask] is not None:
                mask,op,x,y,z = parent[mask]
                steps.append((op,x,y,z))
            regs = {x:i for i,x in enumerate(p.starts)}
            program = []
            for op,x,y,z in reversed(steps):
                dest = len(p.starts)+len(program)
                program.append([op,regs[x],regs[y],dest])
                regs[z] = dest
            replay(p,program)
            return program
        values = [v for v in range(1,p.limit+1) if mask & (1<<v)]
        for i,x in enumerate(values):
            for y in values[:i+1]:
                for op,z in (('+',x+y),('-',x-y),('*',x*y),('/',x//y if x%y==0 else 0)):
                    if not 1<=z<=p.limit: continue
                    new = mask | (1<<z)
                    if new not in parent:
                        parent[new] = (mask,op,x,y,z)
                        queue.append(new)
    raise ValueError('Unreachable targets')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='submission.csv')
    ap.add_argument('--graft', action='store_true')
    ap.add_argument('--engine')
    ap.add_argument('--alternate-slack',type=int,help='Search distinct public value sets within this many steps of best')
    ap.add_argument('--cross')
    ap.add_argument('--pack',help='Optimize the instruction ordering of programs that recompute values')
    ap.add_argument('--include',action='append',default=[])
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--resume', action='store_true',help='Require an existing checkpoint (one is always preserved if present)')
    args = ap.parse_args()
    puzzles = load_puzzles('puzzles.csv')
    if set(puzzles) != {f'scored_{i:04}' for i in range(1,1001)}:
        raise ValueError('Wrong scored puzzle IDs')
    banks = {s:read_bank(s,puzzles) for s in SOURCES}
    best, origins = {}, {}
    for key in sorted(puzzles):
        source = min(banks, key=lambda s:len(banks[s][key]))
        best[key], origins[key] = banks[source][key], source
    print('Public merge',sum(map(len,best.values())),flush=True)
    for filename in args.include:
        extra = read_bank(filename,puzzles,strict=True)
        with Path(filename).with_suffix('.sources.csv').open(newline='') as f:
            labels = {r['id']:r['origin'] for r in csv.DictReader(f)}
        for k,row in extra.items():
            if len(row)<len(best[k]): best[k],origins[k] = row,labels[k]
        banks[filename] = extra
    if args.resume or Path(args.output).exists():
        old = read_bank(args.output,puzzles,strict=True)
        with Path(args.output).with_suffix('.sources.csv').open(newline='') as f:
            labels = {r['id']:r['origin'] for r in csv.DictReader(f)}
        for k,row in old.items():
            if len(row)<len(best[k]): best[k],origins[k] = row,labels[k]
    proven = set()
    for k,p in puzzles.items():
        if p.limit<=16 and len(p.starts)+p.limit<=p.registers:
            optimum = exact_small(p)
            proven.add(k)
            if len(optimum) < len(best[k]):
                best[k],origins[k] = optimum,'exact breadth-first search'
    print('Proved small optima',len(proven),'instructions',sum(len(best[k]) for k in proven),flush=True)
    print('Starting score',save(args.output,puzzles,best,origins),flush=True)
    if args.pack:
        count = 0
        for k,p in puzzles.items():
            seen = set()
            for source,bank in banks.items():
                prog = bank[k]
                if len(graph(p,prog))>=len(best[k]): continue
                signature = tuple(map(tuple,prog))
                if signature in seen: continue
                seen.add(signature)
                count += 1
                candidate = run_cross(args.pack,args.seconds,args.seed,p,[prog],pack=True)
                if candidate is not None and len(candidate)<len(best[k]):
                    replay(p,candidate)
                    print('Reordered',k,len(best[k]),'->',len(candidate),flush=True)
                    best[k],origins[k] = candidate,source+' + register scheduling'
        print('Reordering runs',count,'score',save(args.output,puzzles,best,origins),flush=True)
    if args.graft:
        for i,(k,p) in enumerate(puzzles.items(),1):
            unique, donors = set(), []
            for src,b in banks.items():
                if b[k] == best[k]: continue
                signature = tuple(map(tuple,b[k]))
                if signature not in unique:
                    donors.append((src,b[k]))
                    unique.add(signature)
            for prog,src in graft(p,best[k],donors):
                best[k] = prog
                origins[k] += ' + graft ' + src
            if i%200 == 0:
                print('Grafted', i, 'score',sum(map(len,best.values())),flush=True)
        print('After grafts',save(args.output,puzzles,best,origins),flush=True)
    if args.engine or args.cross:
        keys = sorted((k for k,p in puzzles.items() if k not in proven and len(best[k])>len(set(p.targets)-set(p.starts))),key=lambda k:-len(best[k]))
        tasks = [(k,best[k],origins[k]) for k in keys]
        if args.alternate_slack is not None:
            tasks = []
            if not args.engine: raise ValueError('Alternate programs require --engine')
            for k in keys:
                base = frozenset(graph(puzzles[k],best[k]))
                seen = {base}
                for source,bank in banks.items():
                    alt = bank[k]
                    if len(alt)>len(best[k])+args.alternate_slack: continue
                    values = frozenset(graph(puzzles[k],alt))
                    if values not in seen and len(base^values)>=4:
                        seen.add(values)
                        tasks.append((k,alt,source))
        def run(task):
            k,program,origin = task
            if args.cross:
                inputs = [program]+[b[k] for b in banks.values()]
                return k,origin,run_cross(args.cross,args.seconds,args.seed,puzzles[k],inputs)
            return k,origin,run_native(args.engine,args.seconds,args.seed,puzzles[k],program)
        print('Searching',len(tasks),'candidate programs',flush=True)
        with ThreadPoolExecutor(args.jobs) as pool:
            for i,(k,origin,prog) in enumerate(pool.map(run,tasks),1):
                if prog is not None and len(prog)<len(best[k]):
                    replay(puzzles[k],prog)
                    best[k] = prog
                    origins[k] = origin + f' + {"cross-bank recombination" if args.cross else "triwalk"} seed {args.seed}'
                if i%150 == 0:
                    print('Searched', i, 'of',len(tasks),'score',save(args.output,puzzles,best,origins),flush=True)
        print('After native search',save(args.output,puzzles,best,origins),flush=True)

if __name__ == '__main__':
    main()

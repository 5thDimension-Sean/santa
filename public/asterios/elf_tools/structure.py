"""Structural plan: build targets in increasing order; each target from the pool
{start values, targets already built} with the exact DAG search (<=4 steps). Reports per-target cost
(None = not reachable in <=4) and compares with the public program length."""
import sys, json, csv
from multiprocessing import Pool
sys.path.insert(0, 'src')
from verify import load_puzzles, load_sub
from dag4 import dag_search

def plan(args):
    pid, p, base = args
    pool = {}
    for k, v in enumerate(p['start']):
        pool.setdefault(v, k)
    reg = len(p['start']); costs = []
    for t in sorted(set(p['targets']) - set(p['start'])):
        st = dag_search(t, pool, p['limit'], 4)
        costs.append(len(st) if st else None)
        pool[t] = reg; reg += 1
    return pid, base, costs

if __name__ == '__main__':
    which, n, procs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    P = load_puzzles(f'data/{which}.csv')
    S = load_sub('best/merged_public_v2.csv') if which == 'puzzles' else {}
    rows = list(csv.DictReader(open(f'data/{which}.csv')))
    fam = {r['id']: r.get('family', '?') for r in rows}
    ids = sorted(P)[:n]
    out = []
    with Pool(procs) as pool:
        for pid, base, costs in pool.imap_unordered(plan, [(i, P[i], len(json.loads(S[i])) if S else 0) for i in ids]):
            out.append((pid, base, costs))
    json.dump(out, open(f'runs/structure_{which}_{n}.json', 'w'))
    from collections import Counter
    C = Counter(); byfam = {}
    for pid, base, costs in out:
        for c in costs: C[c] += 1
        f = fam[pid]; d = byfam.setdefault(f, Counter())
        for c in costs: d[c] += 1
    print('cost histogram (None = needs >=5 from starts+earlier targets):', dict(sorted(C.items(), key=lambda x: (x[0] is None, x[0] or 0))))
    for f, d in byfam.items(): print(f, dict(sorted(d.items(), key=lambda x: (x[0] is None, x[0] or 0))))

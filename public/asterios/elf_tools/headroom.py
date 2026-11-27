"""Engine-headroom diagnostic. At the state after the K-th covered target of the public program,
estimate for every remaining target the cheapest independent build with shapes
   t = A*B + r  or  t = A*B - r,   A, B, r in D = pool(0) | S1(1) | F2(2)   (cost = cA+cB+cr+1+[r>0])
and also the exact <=4 search. Compare with the public cost per remaining target."""
import sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, 'src')
from verify import load_puzzles, load_sub
from resynth import apply
from dag4 import dag_search
from deep4 import _combine
from dag5 import forward2

def table(pool_vals, L):
    V = np.array(pool_vals, dtype=np.int64)
    s1 = _combine(V, V, L)[0]
    F, _ = forward2(V, L)
    vals = np.concatenate([V, s1, F]); cost = np.concatenate([np.zeros(len(V), np.int8), np.ones(len(s1), np.int8),
                                                              np.full(len(F), 2, np.int8)])
    o = np.lexsort((cost, vals)); vals, cost = vals[o], cost[o]
    keep = np.ones(len(vals), bool); keep[1:] = vals[1:] != vals[:-1]
    return vals[keep], cost[keep]

def lookup(vals, cost, q):
    pos = np.searchsorted(vals, q); pos = np.minimum(pos, len(vals) - 1)
    hit = vals[pos] == q
    c = np.where(hit, cost[pos], 99).astype(np.int16)
    return c

def best_mulshape(t, vals, cost):
    A = vals[vals <= t]; cA = cost[vals <= t].astype(np.int16)
    B0 = t // A
    best = 99
    for B, sign in ((B0, 1), (B0 + 1, -1)):
        ok = B >= 1
        cB = lookup(vals, cost, B)
        with np.errstate(over='ignore'):
            r = (t - A * B) * sign
        cr = np.where(r == 0, 0, lookup(vals, cost, np.maximum(r, 1)))
        tot = cA + cB + cr + 1 + (r != 0)
        tot = np.where(ok & (r >= 0), tot, 99)
        best = min(best, int(tot.min()))
    return best

def run(args):
    pid, p, prog, K = args
    L = p['limit']; T = set(p['targets']); seen = set(p['start'])
    R = [None] * 48
    for k, v in enumerate(p['start']): R[k] = v
    covered = 0; cut = None
    for n, (op, i, j, d) in enumerate(prog):
        R[d] = apply(op, R[i], R[j], L)
        if R[d] in T and R[d] not in seen:
            covered += 1
            if covered == K: cut = n + 1; Rcut = R[:]; seencut = set(seen) | {R[d]}
        seen.add(R[d])
    if cut is None: return None
    rem = sorted(T - seencut)
    pool = {}
    for r, v in enumerate(Rcut):
        if v is not None and v not in pool: pool[v] = r
    vals, cost = table(list(pool), L)
    res = []
    for t in rem:
        st = dag_search(t, pool, L, 4)
        c = len(st) if st else best_mulshape(t, vals, cost)
        res.append((int(t).bit_length(), c))
    return pid, (len(prog) - cut) / len(rem), res

if __name__ == '__main__':
    n, K, procs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    P = load_puzzles('data/puzzles.csv'); S = load_sub('best/merged_public_v2.csv')
    ids = [i for i in sorted(P) if len(P[i]['targets']) >= K + 8][:n]
    out = []
    with Pool(procs) as pool:
        for r in pool.imap_unordered(run, [(i, P[i], json.loads(S[i]), K) for i in ids]):
            if r: out.append(r)
    from collections import defaultdict
    pub = np.mean([o[1] for o in out])
    allc = [c for o in out for b, c in o[2]]
    print(f'puzzles {len(out)}  public cost per remaining target {pub:.2f}  independent-build estimate {np.mean([min(c, 12) for c in allc]):.2f}')
    byb = defaultdict(list)
    for o in out:
        for b, c in o[2]: byb[b // 5 * 5].append(c)
    for b in sorted(byb):
        cs = byb[b]; print(f'bits {b:2d}-{b+4:<2d} n={len(cs):4d} mean={np.mean([min(c,12) for c in cs]):.2f} hist={dict(sorted(__import__("collections").Counter(cs).items()))}')
    json.dump(out, open(f'runs/headroom_{n}_{K}.json', 'w'))

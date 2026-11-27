"""CP-SAT large-neighbourhood search over value DAGs (Santa 2026).

Modes:  exp0 (bank-only full rebuild), exp1 (positive control), lns (priority run).
Never writes submission.csv or banks; outputs go to experiments/ and submission-cpsat*.csv.
"""
import argparse, csv, json, os, pickle, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "public" / "goodpjw2008"))
from elf import load_puzzles, replay  # noqa: E402

EXP = ROOT / "experiments"
WORK = EXP / "cpsat-lns-work"
BASE = EXP / "cp-sat-rebuild.csv"
BANK_FILES = [
    "experiments/submission-notebook-merge.csv", "experiments/native-candidate.csv",
    "experiments/schedule-candidate.csv", "experiments/whole-hub-submission.csv",
    "experiments/submission-focal-108.csv", "public/basin-lineage-v10/submission.csv",
    "public/bounds-triage-v10/submission.csv", "public/frontier-nov27/submission.csv",
    "public/viktorbarbarich/best_programs.csv", "public/b05204047elias/submission.csv",
    "public/ishankarnick/submission.csv", "public/goodpjw2008/our_programs.csv",
    "public/goodpjw2008/public_frontier.csv", "public/alexchilton/rudolph_programs.csv",
    "public/asterios/best_programs.csv", "public/farshid/submission.csv",
    "public/asterios/public_day_ago.csv", "public/goodpjw2008/frontier_22nov.csv",
    "public/goodpjw2008/frontier_20nov.csv",
]
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
       "/": lambda a, b: a // b}


# ---------------------------------------------------------------- DAG utils
def prog_graph(p, prog):
    """value -> (op,a,b) for non-start values (first recipe wins), plus build order."""
    regs = list(p.starts) + [None] * (p.registers - len(p.starts))
    g, order, starts = {}, [], set(p.starts)
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]
        v = OPS[op](a, b)
        if v not in g and v not in starts:
            g[v] = (op, a, b); order.append(v)
        regs[d] = v
    return g, order


def dce(p, g):
    """Drop values not needed for any target."""
    starts, need, st = set(p.starts), set(), [t for t in p.targets if t not in set(p.starts)]
    while st:
        v = st.pop()
        if v in need or v in starts: continue
        need.add(v); st.extend(g[v][1:])
    return {v: r for v, r in g.items() if v in need}


def consumers_of(g):
    c = defaultdict(set)
    for v, (_, a, b) in g.items():
        c[a].add(v); c[b].add(v)
    return c


def descendants(c, v):
    seen, st = set(), [v]
    while st:
        x = st.pop()
        for y in c.get(x, ()):
            if y not in seen: seen.add(y); st.append(y)
    return seen


def close_exclusive(p, g, D, cons):
    """Add values whose consumers all lie in D (they would become dead)."""
    targets, starts = set(p.targets), set(p.starts)
    D = set(D); changed = True
    while changed:
        changed = False
        for v in list(D):
            for u in g[v][1:]:
                if u in D or u in starts or u in targets: continue
                if cons[u] <= D:
                    D.add(u); changed = True
    return D


def choose_D(p, g, strategy, rng, bank_rec, size=(10, 25)):
    cons = consumers_of(g)
    vals = list(g)
    targets = set(p.targets)
    if strategy == "all":
        return set(vals)
    if strategy == "output":
        t = rng.choice([v for v in vals if v in targets] or vals)
        D = {t}
    elif strategy == "hub":
        hubs = [v for v in vals if len(cons[v]) >= 2 and v not in targets] or vals
        h = rng.choice(hubs)
        D = {h} | descendants(cons, h)
        if len(D) > 60:
            D = set(rng.sample(sorted(D - {h}), 59)) | {h}
    elif strategy == "disagree":
        dis = [v for v in vals if len(bank_rec.get(v, ())) > 1 and any(r != g[v] for r in bank_rec[v])]
        if not dis: return choose_D(p, g, "region", rng, bank_rec, size)
        D = {rng.choice(dis)}
        target = rng.randint(*size)
        return grow(g, cons, D, target, rng, prefer=set(dis))
    else:  # region
        D = {rng.choice(vals)}
        return grow(g, cons, D, rng.randint(*size), rng)
    return D


def grow(g, cons, D, target, rng, prefer=()):
    D = set(D)
    while len(D) < target:
        frontier = set()
        for v in D:
            frontier.update(u for u in g[v][1:] if u in g)
            frontier.update(cons[v])
        frontier -= D
        if not frontier: break
        pref = [v for v in frontier if v in prefer]
        D.add(rng.choice(pref if pref and rng.random() < 0.7 else sorted(frontier)))
    return D


# ---------------------------------------------------------- candidate generation
class Cands:
    """Candidate value -> {recipe: source-tag}."""
    def __init__(self, limit, starts_kept):
        self.rec = defaultdict(dict)
        self.limit = limit
        self.known = starts_kept  # K: kept values + starts

    def add(self, v, op, a, b, tag):
        if v in self.known or not (1 <= v <= self.limit) or v == a or v == b: return
        if op in "+*" and a > b: a, b = b, a
        self.rec[v].setdefault((op, a, b), tag)


def one_step_hits(X, A, Usorted, limit):
    """For each x in X, all (op,a,b) with a in A and b in U with x = a op b (vectorised)."""
    X = np.asarray(X, dtype=np.int64); out = []
    Ua = np.asarray(Usorted, dtype=np.int64)
    if len(Ua) == 0 or len(X) == 0: return out
    for a in A:
        a = np.int64(a)
        tests = (("+", X - a), ("-", a - X), ("-", X + a))  # x=a+b ; x=a-b -> b=a-x ; x=b-a -> b=x+a
        for kind, (op, B) in enumerate(tests):
            m = B >= 1
            idx = np.searchsorted(Ua, B[m]); idx[idx >= len(Ua)] = len(Ua) - 1
            hit = Ua[idx] == B[m]
            for xi in np.nonzero(m)[0][hit]:
                b = int(B[xi])
                if kind == 0: out.append((int(X[xi]), "+", int(a), b))
                elif kind == 1: out.append((int(X[xi]), "-", int(a), b))
                else: out.append((int(X[xi]), "-", b, int(a)))
        # multiplicative: x = a*b -> b = x/a ; x = b/a -> b = x*a ; x = a/b -> b = a/x
        ok = (X % a == 0)
        B = X[ok] // a
        idx = np.searchsorted(Ua, B); idx[idx >= len(Ua)] = len(Ua) - 1
        for xi, b in zip(X[ok][Ua[idx] == B], B[Ua[idx] == B]):
            out.append((int(xi), "*", int(a), int(b)))
        ok = X * a <= limit
        B = X[ok] * a
        idx = np.searchsorted(Ua, B); idx[idx >= len(Ua)] = len(Ua) - 1
        for xi, b in zip(X[ok][Ua[idx] == B], B[Ua[idx] == B]):
            out.append((int(xi), "/", int(b), int(a)))
        ok = (a % X == 0)
        B = a // X[ok]
        idx = np.searchsorted(Ua, B); idx[idx >= len(Ua)] = len(Ua) - 1
        for xi, b in zip(X[ok][Ua[idx] == B], B[Ua[idx] == B]):
            out.append((int(xi), "/", int(a), int(b)))
    return out


def pair_closure(A, limit, cap=400_000):
    """One-step values from pairs of A (list) with recipes."""
    res = {}
    A = sorted(set(A))
    for i, a in enumerate(A):
        for b in A[: i + 1]:
            s = a + b
            if s <= limit: res.setdefault(s, ("+", a, b))
            if a * b <= limit: res.setdefault(a * b, ("*", b, a))
            if a != b: res.setdefault(a - b, ("-", a, b))
            if a % b == 0 and a != b: res.setdefault(a // b, ("/", a, b))
        if len(res) > cap: break
    return res


def generate(p, K, R, bank_vals_rec, orig_rec, args, rng):
    """Return Cands and source counters. K: dict kept+starts value->recipe|None."""
    limit = p.limit
    C = Cands(limit, set(K))
    src = Counter()
    # 1: original recipes of D, 2: bank recipes of the whole puzzle
    for v, r in orig_rec.items():
        C.add(v, *r[:1], r[1], r[2], "orig")
    for v, recs in bank_vals_rec.items():
        for r in recs:
            C.add(v, r[0], r[1], r[2], "bank")
    Kl = sorted(K)
    # pool U = K + current candidate values
    def U():
        return sorted(set(K) | set(C.rec))
    Rl = sorted(R)
    # 4: hubs -- values one step from K voted by how many r in R have a cheap route through them
    if args.hubs:
        one = pair_closure(Kl if len(Kl) <= 250 else rng.sample(Kl, 250), limit)
        Uset = set(U())
        hubs = np.array([h for h in one if h not in Uset or h in C.rec], dtype=np.int64)
        votes = Counter()
        if len(hubs):
            Us = np.asarray(sorted(Uset), dtype=np.int64)
            for r in Rl:
                for B in (r - hubs, hubs - r, r + hubs):
                    m = B >= 1
                    idx = np.searchsorted(Us, B[m]); idx[idx >= len(Us)] = len(Us) - 1
                    for h in hubs[m][Us[idx] == B[m]]: votes[int(h)] += 1
                if r > 1:
                    m = (r % hubs == 0)
                    B = r // hubs[m]
                    idx = np.searchsorted(Us, B); idx[idx >= len(Us)] = len(Us) - 1
                    for h in hubs[m][Us[idx] == B]: votes[int(h)] += 1
        for h, n in votes.most_common(args.hubs):
            op, a, b = one[h]
            C.add(h, op, a, b, "hub"); src["hub_values"] += 1
    # 5: residual intermediates x with r = x op h, h in K or candidate (cheap), x one step from U
    if args.residual:
        Hs = sorted(set(Kl) | {v for v, rr in C.rec.items() if any(t == "hub" for t in rr.values())})
        Hs = Hs if len(Hs) <= 400 else rng.sample(Hs, 400)
        newx = []
        for r in Rl:
            for h in Hs:
                for x, op, a, b in ((r - h, "+", "x", h), (r + h, "-", "x", h), (h - r, "-", h, "x"),
                                    (r // h if h > 1 and r % h == 0 else 0, "*", "x", h),
                                    (r * h if r * h <= limit else 0, "/", "x", h),
                                    (h // r if r > 0 and h % r == 0 and h != r else 0, "/", h, "x")):
                    if x >= 1 and x not in K and x != r:
                        newx.append((x, r, op, a, b))
        xs = sorted({t[0] for t in newx})
        if xs:
            Ul = U()
            hits = one_step_hits(xs, Hs, Ul, limit)
            good = defaultdict(list)
            for x, op, a, b in hits: good[x].append((op, a, b))
            budget = args.max_residual
            for x, r, op, a, b in newx:
                if x in good and budget > 0:
                    aa = x if a == "x" else a; bb = x if b == "x" else b
                    C.add(r, op, aa, bb, "residual")
                    for rec in good[x][:6]:
                        C.add(x, *rec, "residual"); src["residual_values"] += 0
                    budget -= 1
    # direct one-step recipes into R and candidates from K-side operands
    Ul = U()
    tgt = sorted(set(Rl) | set(C.rec))
    for x, op, a, b in one_step_hits(tgt, Kl if len(Kl) <= 300 else rng.sample(Kl, 300), Ul, limit):
        C.add(x, op, a, b, "closure")
    # 6: near-ratio relations between R values
    if args.ratio:
        for i, t1 in enumerate(Rl):
            for t2 in Rl[i + 1:]:
                hi, lo = (t2, t1) if t2 > t1 else (t1, t2)
                for bq in range(1, 51):
                    a = round(hi * bq / lo)
                    if a < 1: continue
                    # hi*bq = a*lo + c  ->  hi = (a*lo + c)/bq ; c small
                    c = hi * bq - a * lo
                    if abs(c) > 10_000: continue
                    # emit only if cheap pieces exist later via closure; recorded as no-op tag
    return C


# ---------------------------------------------------------------- CP-SAT model
def solve_model(p, K, cands, R, hint, tl, nogoods=(), log=None):
    """K: kept value->recipe (None for starts). cands: value->{recipe:tag}. Returns (selected recipes, status, obj)."""
    from ortools.sat.python import cp_model
    Kset = set(K)
    # prune: forward-feasible
    avail = set(Kset); rec = {v: dict(r) for v, r in cands.items()}
    changed = True
    while changed:
        changed = False
        for v, rs in rec.items():
            if v in avail: continue
            if any(a in avail and b in avail for (_, a, b) in rs):
                avail.add(v); changed = True
    rec = {v: {r: t for r, t in rs.items() if r[1] in avail and r[2] in avail} for v, rs in rec.items() if v in avail}
    # backward-reachable from R
    need, st = set(), [r for r in R]
    while st:
        v = st.pop()
        if v in need or v in Kset: continue
        if v not in rec: return None, "UNREACHABLE_R", None
        need.add(v)
        for (_, a, b) in rec[v]: st.extend((a, b))
    rec = {v: rs for v, rs in rec.items() if v in need}
    m = cp_model.CpModel()
    x = {v: m.NewBoolVar("") for v in rec}
    big = len(rec) + len(K) + 1
    lvl = {v: m.NewIntVar(1, big, "") for v in rec}
    # kept values' levels: fixed depth when independent of cands, else variable
    klvl = {}
    def kl(v):
        if v in klvl: return klvl[v]
        r = K[v]
        if r is None: klvl[v] = 0; return 0
        subs = [kl(u) if u in K else (lvl[u] if u in lvl else 0) for u in r[1:]]
        if all(isinstance(s, int) for s in subs): klvl[v] = 1 + max(subs)
        else:
            nv = m.NewIntVar(1, big, ""); klvl[v] = nv
            for s in subs: m.Add(nv >= s + 1)
        return klvl[v]
    sys.setrecursionlimit(10000)
    for v in K: kl(v)
    ys = {}
    for v, rs in rec.items():
        yl = []
        for r in rs:
            y = m.NewBoolVar(""); ys[(v, r)] = y; yl.append(y)
            for o in {r[1], r[2]}:
                if o in x:
                    m.AddImplication(y, x[o])
                    m.Add(lvl[v] >= lvl[o] + 1).OnlyEnforceIf(y)
                elif o in K:
                    s = kl(o)
                    m.Add(lvl[v] >= (s + 1 if isinstance(s, int) else 0)).OnlyEnforceIf(y) if isinstance(s, int) else m.Add(lvl[v] >= s + 1).OnlyEnforceIf(y)
        m.Add(sum(yl) == x[v])
    for r in R:
        if r in x: m.Add(x[r] == 1)
    # kept values that consume cand values need those built (they are in R by construction)
    for v, r in K.items():
        if r is None: continue
        for u in r[1:]:
            if u not in K:
                if u not in x: return None, "KEPT_NEEDS_MISSING", None
                m.Add(x[u] == 1)
                m.Add(klvl[v] >= lvl[u] + 1) if not isinstance(klvl[v], int) else None
    m.Minimize(1000 * sum(x.values()) + sum(y for (v, r), y in ys.items() if hint.get(v) != r))
    for ng in nogoods:
        lits = [ys[k] for k in ng if k in ys]
        if len(lits) == len(ng): m.AddBoolOr([l.Not() for l in lits])
    for v, r in hint.items():
        if v in x:
            m.AddHint(x[v], 1)
            for rr in rec[v]:
                m.AddHint(ys[(v, rr)], 1 if rr == r else 0)
    for v in x:
        if v not in hint: m.AddHint(x[v], 0)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = 1
    st = s.Solve(m)
    name = s.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None, name, None
    chosen = {}
    for v in x:
        if s.Value(x[v]):
            for r in rec[v]:
                if s.Value(ys[(v, r)]): chosen[v] = (r, rec[v][r])
    return chosen, name, len(chosen)


# --------------------------------------------------------------- scheduling
def schedule(p, chosen_rec, rank):
    """chosen_rec: value -> (op,a,b). Returns program or None (<=registers)."""
    starts = set(p.starts)
    users = defaultdict(list)
    for v, (_, a, b) in chosen_rec.items():
        users[a].append(v)
        if b != a: users[b].append(v)
    for mode in range(5):
        remaining = Counter(x for _, a, b in chosen_rec.values() for x in (a, b))
        indeg = {v: len({x for x in chosen_rec[v][1:] if x not in starts}) for v in chosen_rec}
        ready = [v for v in chosen_rec if indeg[v] == 0]
        order, done, lvl = [], set(), {}
        while ready:
            def pri(v):
                _, a, b = chosen_rec[v]
                freed = (remaining[a] == 1) + (b != a and remaining[b] == 1)
                pos = rank.get(v, 10**9)
                if mode == 0: return (pos, -freed, v)
                if mode == 1: return (-freed, pos, v)
                if mode == 2: return (pos, freed == 0, v)
                if mode == 3: return (-freed, -pos, v)
                return (-pos, -freed, v)
            ready.sort(key=pri)
            v = ready.pop(0)
            order.append(v); done.add(v)
            _, a, b = chosen_rec[v]
            remaining[a] -= 1; remaining[b] -= 1
            for u in users[v]:
                indeg[u] -= 1
                if indeg[u] == 0 and u not in done and u not in ready: ready.append(u)
        if len(order) != len(chosen_rec): continue
        uses = Counter(x for v in order for x in chosen_rec[v][1:])
        regs = {v: i for i, v in enumerate(p.starts) if uses[v]}
        free = [i for i in range(p.registers) if i not in regs.values()]
        prog, ok = [], True
        for v in order:
            op, a, b = chosen_rec[v]
            if a not in regs or b not in regs: ok = False; break
            ra, rb = regs[a], regs[b]
            for o in (a, b):
                uses[o] -= 1
                if uses[o] == 0 and o in regs: free.append(regs.pop(o))
            if not free: ok = False; break
            d = free.pop(); prog.append([op, ra, rb, d])
            if uses[v]: regs[v] = d
            else: free.append(d)
        if ok:
            try:
                replay(p, prog); return prog
            except ValueError:
                pass
    return None


# ----------------------------------------------------------- one neighbourhood
def neighbourhood(p, prog, strategy, bank_rec, args, rng, seconds, row):
    """Try to improve prog. Returns (new_prog or None). Fills row dict."""
    g, order = prog_graph(p, prog)
    g = dce(p, g)
    rank = {v: i for i, v in enumerate(order)}
    cons = consumers_of(g)
    D = choose_D(p, g, strategy, rng, bank_rec, size=(args.dmin, args.dmax))
    D = close_exclusive(p, g, D, cons)
    targets = set(p.targets)
    R = {v for v in D if v in targets or (cons[v] - D)}
    K = {v: None for v in p.starts}
    K.update({v: r for v, r in g.items() if v not in D})
    row.update(strategy=strategy, D=len(D), R=len(R))
    orig_rec = {v: g[v] for v in D}
    t0 = time.perf_counter()
    C = generate(p, K, R, bank_rec, orig_rec, args, rng)
    gen_t = time.perf_counter() - t0
    srcs = Counter(t for rs in C.rec.values() for t in rs.values())
    row.update(cands=len(C.rec), recipes=sum(map(len, C.rec.values())),
               src="|".join(f"{k}:{v}" for k, v in sorted(srcs.items())), gen_s=round(gen_t, 1))
    # cap
    if row["recipes"] > args.max_recipes:
        row["reason"] = "too_many_recipes"; return None, None
    keptn = len(K) - len(p.starts)
    nogoods = []
    best = None
    for attempt in range(4):
        chosen, status, obj = solve_model(p, K, C.rec, R, orig_rec, max(1.0, seconds - gen_t), nogoods)
        row.update(status=status, obj=obj)
        if chosen is None:
            row["reason"] = status; return None, None
        total = keptn + obj
        if total >= len(prog):
            row["reason"] = "no_saving" ; return None, chosen
        rec = {v: r for v, r in g.items() if v not in D}
        rec.update({v: r for v, (r, t) in chosen.items()})
        new = schedule(p, rec, rank)
        if new is not None and len(new) < len(prog):
            tags = sorted({t for v, (r, t) in chosen.items() if v not in orig_rec or orig_rec[v] != r})
            row.update(accepted=1, new_len=len(new), reason="", used_sources="|".join(tags))
            return new, chosen
        row["reason"] = "unscheduled"
        nogoods.append([(v, r) for v, (r, t) in chosen.items()])
    return None, None


# ------------------------------------------------------------------ driver
CACHE = Path(os.environ.get("CPSAT_CACHE", str(WORK / "bank_cache.pkl")))


def load_banks(puzzles):
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    from optimize import read_bank
    rec = {k: defaultdict(set) for k in puzzles}
    for f in BANK_FILES + ["experiments/cp-sat-rebuild.csv"]:
        bank = read_bank(str(ROOT / f), puzzles)
        for k, prog in bank.items():
            g, _ = prog_graph(puzzles[k], prog)
            for v, r in g.items(): rec[k][v].add(r)
    rec = {k: dict(v) for k, v in rec.items()}
    CACHE.parent.mkdir(exist_ok=True, parents=True)
    pickle.dump(rec, open(CACHE, "wb"))
    return rec


def read_progs(path, puzzles):
    from optimize import read_bank
    return read_bank(str(path), puzzles, strict=True)


def full_rebuild(p, prog, bank_rec, args, rng, seconds, row):
    """Experiment 0/1: D = everything, candidates from bank (and optionally generators)."""
    g, order = prog_graph(p, prog); g = dce(p, g)
    rank = {v: i for i, v in enumerate(order)}
    K = {v: None for v in p.starts}
    R = set(t for t in p.targets if t not in K)
    C = Cands(p.limit, set(K))
    for v, rs in bank_rec.items():
        for r in rs: C.add(v, r[0], r[1], r[2], "bank")
    for v, r in g.items(): C.add(v, *r, "orig")
    row.update(strategy="all", D=len(g), R=len(R), cands=len(C.rec), recipes=sum(map(len, C.rec.values())))
    nogoods = []
    for _ in range(4):
        chosen, status, obj = solve_model(p, K, C.rec, R, g, seconds, nogoods)
        row.update(status=status, obj=obj)
        if chosen is None: return None
        new = schedule(p, {v: r for v, (r, t) in chosen.items()}, rank)
        if new is not None: return new
        row["reason"] = "unscheduled"
        nogoods.append([(v, r) for v, (r, t) in chosen.items()])
    return None


def exp0_job(a):
    key, seconds = a
    puzzles = load_puzzles(str(ROOT / "puzzles.csv"))
    bank_rec = load_banks(puzzles)[key]
    base = read_progs_cache()[key]
    p = puzzles[key]
    row = dict(id=key, base=len(base)); t0 = time.perf_counter()
    new = full_rebuild(p, base, bank_rec, argparse.Namespace(), random.Random(1), seconds, row)
    row["time"] = round(time.perf_counter() - t0, 1)
    row["new_len"] = len(new) if new else None
    return row, new


_BASE = None
def read_progs_cache():
    global _BASE
    if _BASE is None:
        puzzles = load_puzzles(str(ROOT / "puzzles.csv"))
        _BASE = read_progs(BASE, puzzles)
    return _BASE


def pin(i):
    try: os.sched_setaffinity(0, {i % os.cpu_count()})
    except Exception: pass


def init_worker(cores):
    import multiprocessing
    idx = multiprocessing.current_process()._identity[0] - 1
    try: os.sched_setaffinity(0, {cores[idx % len(cores)]})
    except Exception: pass


def lns_job(a):
    key, budget, args, base_prog = a
    puzzles = load_puzzles(str(ROOT / "puzzles.csv"))
    p = puzzles[key]
    bank_rec = load_banks(puzzles)[key]
    rng = random.Random(hash((key, args.seed)) & 0xFFFFFFFF)
    prog = base_prog; rows, imps = [], []
    strategies = ["output", "hub", "region", "disagree"]
    t_start = time.perf_counter(); it = 0
    while time.perf_counter() - t_start < budget:
        strat = strategies[it % len(strategies)]; it += 1
        row = dict(id=key, strategy=strat, accepted=0)
        t0 = time.perf_counter()
        try:
            new, _ = neighbourhood(p, prog, strat, bank_rec, args, rng, args.nb_seconds, row)
        except Exception as e:  # keep the job alive; log the failure
            new = None; row["reason"] = f"error:{type(e).__name__}:{e}"[:120]
        row["time"] = round(time.perf_counter() - t0, 1)
        if new is not None:
            replay(p, new)
            imps.append(dict(id=key, old=len(prog), new=len(new), strategy=strat, sources=row.get("used_sources", "")))
            prog = new
        rows.append(row)
    return key, prog, rows, imps


def write_csv(path, rows, fields=None):
    if not rows: return
    fields = fields or sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exp0", "exp1", "lns"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cores", default="0-7")
    ap.add_argument("--ids", default="")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--nb-seconds", type=float, default=30, dest="nb_seconds")
    ap.add_argument("--hubs", type=int, default=200)
    ap.add_argument("--residual", type=int, default=1)
    ap.add_argument("--ratio", type=int, default=0)
    ap.add_argument("--max-residual", type=int, default=3000, dest="max_residual")
    ap.add_argument("--max-recipes", type=int, default=50000, dest="max_recipes")
    ap.add_argument("--dmin", type=int, default=10)
    ap.add_argument("--dmax", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    lo, _, hi = args.cores.partition("-"); cores = list(range(int(lo), int(hi or lo) + 1))
    puzzles = load_puzzles(str(ROOT / "puzzles.csv"))
    WORK.mkdir(exist_ok=True, parents=True)
    load_banks(puzzles)
    base = read_progs_cache()
    ids = args.ids.split(",") if args.ids else list(puzzles)
    if args.mode == "exp0":
        out = WORK / "exp0.jsonl"
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        todo = [(k, args.seconds) for k in ids if k not in done]
        with Pool(args.workers, init_worker, (cores,)) as pool, open(out, "a") as f:
            for row, new in pool.imap_unordered(exp0_job, todo):
                f.write(json.dumps(row) + "\n"); f.flush()
        rows = [json.loads(l) for l in open(out)]
        ok = sum(1 for r in rows if r["new_len"] == r["base"]); better = sum(1 for r in rows if r["new_len"] and r["new_len"] < r["base"])
        print(f"exp0: {len(rows)} puzzles, matched {ok}, better {better}, worse/none {len(rows)-ok-better}")
        return
    ranking = sorted(((len(base[k]) - len(set(puzzles[k].targets) - set(puzzles[k].starts)), k) for k in puzzles), reverse=True)
    if args.ids: order = ids
    else: order = [k for _, k in ranking][args.skip:args.skip + args.top]
    ck = WORK / f"{args.tag}-progs.json"
    cur = {k: base[k] for k in puzzles}
    if ck.exists(): cur.update(json.load(open(ck)))
    logf, impf = EXP / "cpsat-lns-log.csv", EXP / "cpsat-lns-improvements.csv"
    jobs = [(k, args.budget, args, cur[k]) for k in order]
    allrows, allimps = [], []
    with Pool(args.workers, init_worker, (cores,)) as pool:
        for key, prog, rows, imps in pool.imap_unordered(lns_job, jobs):
            for r in rows: r["tag"] = args.tag
            allrows += rows; allimps += imps
            if imps:
                cur[key] = prog
                json.dump({k: v for k, v in cur.items() if len(v) < len(base[k])}, open(ck, "w"))
            write_csv(EXP / f"cpsat-lns-log-{args.tag}.csv", allrows)
            write_csv(EXP / f"cpsat-lns-improvements-{args.tag}.csv", allimps)
            print(key, "rows", len(rows), "improvements", len(imps), flush=True)
    saved = sum(len(base[k]) - len(cur[k]) for k in puzzles)
    print("total saved", saved)


if __name__ == "__main__":
    main()

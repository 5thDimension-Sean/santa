"""CP-SAT large-neighbourhood search over value DAGs (Santa 2026).

Modes: exp0 through exp6, suite (all gates in order), publish (dual replay).
Never writes submission.csv or banks; outputs go to experiments/ and submission-cpsat*.csv.
"""
import pickle, sys, time
import subprocess
import hashlib
from near_reductions import canonical, convergents
from collections import Counter, defaultdict, deque
from pathlib import Path


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
    if strategy in ("all", "global"):
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
    def __init__(self, limit, starts_kept, max_values=5000, max_recipes=50000):
        self.max_values = max_values
        self.max_recipes = max_recipes
        self.count = 0
        self.rec = defaultdict(dict)
        self.limit = limit
        self.known = starts_kept  # K: kept values + starts

    def add(self, v, op, a, b, tag):
        if not (0 < a <= self.limit and 0 < b <= self.limit and 0 < v <= self.limit):
            return
        if op not in OPS or (op == "/" and a % b) or OPS[op](a, b) != v:
            return
        if v in self.known or v in (a, b) or self.count >= self.max_recipes:
            return
        if v not in self.rec and len(self.rec) >= self.max_values:
            return
        recipe = canonical((op, a, b))
        if recipe not in self.rec[v]:
            self.rec[v][recipe] = tag
            self.count += 1

    def prune(self, required):
        """Keep two layers of alternatives plus the support of their concrete chains."""
        queue = deque((v, 0) for v in required)
        kept = {}
        while queue:
            v, depth = queue.popleft()
            if v in kept or v not in self.rec:
                continue
            rules = self.rec[v]
            if depth > 2:
                rules = {r: tag for r, tag in rules.items() if tag != "closure"}
            kept[v] = rules
            for _, left, right in rules:
                queue.extend((operand,depth+1) for operand in {left,right} if operand not in kept)
        self.rec = kept
        self.count = sum(map(len,kept.values()))


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
        ok = X <= limit // a
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
            if a % b == 0: res.setdefault(a // b, ("/", a, b))
        if len(res) > cap: break
    return res


def generate(p, K, R, bank_vals_rec, orig_rec, args, rng, global_search=False):
    """Return Cands and source counters. K: dict kept+starts value->recipe|None."""
    limit = p.limit
    C = Cands(limit, set(K), args.max_values, args.max_recipes)
    # 1: original recipes of D, 2: bank recipes of the whole puzzle
    for v, r in orig_rec.items():
        C.add(v, *r[:1], r[1], r[2], "orig")
    for v, recs in bank_vals_rec.items():
        for r in recs:
            C.add(v, r[0], r[1], r[2], "bank")
    Kl = sorted(set(K) | set(orig_rec)) if global_search else sorted(K)
    # pool U = K + current candidate values
    def U():
        return sorted(set(K) | set(C.rec))
    Rl = sorted(R)
    # 4: hubs -- values one step from K voted by how many r in R have a cheap route through them
    one = pair_closure(Kl, limit)
    if args.hubs:
        Uset = set(U())
        hubs = np.array([h for h in one if h not in Uset or h in C.rec], dtype=np.int64)
        votes = Counter()
        if len(hubs):
            Us = np.asarray(sorted(Uset), dtype=np.int64)
            cheap_steps = np.asarray(sorted(set(one) | Uset), dtype=np.int64)
            for r in Rl:
                matched = np.zeros(len(hubs), dtype=bool)
                def present(values, pool):
                    idx = np.minimum(np.searchsorted(pool, values), len(pool)-1)
                    return pool[idx] == values
                for B in (r - hubs, hubs - r, r + hubs):
                    matched |= present(B, Us)
                matched |= (r % hubs == 0) & present(r // hubs, Us)
                # Count each port once, including inexpensive scaled-offset routes.
                q = (r + hubs//2)//hubs
                valid = (q >= 1) & (q <= limit//hubs)
                scaled = q[valid]*hubs[valid]
                matched[valid] |= present(q[valid], Us) & present(abs(r-scaled), cheap_steps)
                votes.update(map(int, hubs[matched]))
        for h, n in votes.most_common(args.hubs):
            op, a, b = one[h]
            C.add(h, op, a, b, "hub")
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
                    if 1 <= x <= limit and x not in K and x != r:
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
                        C.add(x, *rec, "residual")
                    budget -= 1
    # For a reduced fraction n/d, build d*r = n*h +/- correction.
    cheap = set(K) | C.rec.keys() | one.keys()
    def relation(r, h, n, d, tag):
        scaled, product = r*d, n*h
        correction = abs(scaled - product)
        if n < 1 or max(scaled, product) > limit:
            return
        parts = [v for v in (n, d) if v != 1]
        if correction:
            parts.append(correction)
        if any(v not in cheap for v in parts):
            return
        for v in parts:
            if v in one:
                C.add(v, *one[v], tag)
        if n != 1:
            C.add(product, "*", h, n, tag)
        if correction:
            C.add(scaled, "+" if product < scaled else "-", product, correction, tag)
        if d != 1:
            C.add(r, "/", scaled, d, tag)

    for r in Rl:
        if args.ratio:
            for h in sorted(set(p.targets) & (set(K) | R) - {r}):
                for d in range(1, 51):
                    relation(r, h, (r*d + h//2)//h, d, "ratio")
        if args.residual:
            for h in Hs:
                relation(r, h, (r + h//2)//h, 1, "residual")
                for n, d in convergents(r, h, 12):
                    relation(r, h, n, d, "residual")

    if args.native_source:
        native_chains(C, args, Kl, Rl)
    Ul = U()
    for x, op, a, b in one_step_hits(Rl, Kl, Ul, limit):
        C.add(x, op, a, b, "closure")
    for x, op, a, b in one_step_hits(sorted(C.rec), Kl, Ul, limit):
        C.add(x, op, a, b, "closure")
    C.prune(R)
    return C


def native_chains(C, args, kept, targets):
    if not targets:
        return
    payload = " ".join(map(str, [args.native_source, C.limit, len(kept), len(targets),
                                 args.chain_cap, *kept, *targets]))
    try:
        run = subprocess.run([args.helper], input=payload.encode(), capture_output=True,
                             timeout=args.native_seconds, check=True)
        output = run.stdout
    except subprocess.TimeoutExpired as exc:
        # The helper flushes completed witnesses individually.
        output = exc.stdout or b""
        args.native_timeout = True
    for line in output.split(b"\n")[:-1]:
        words = line.decode().split()
        for i in range(0, len(words), 4):
            value, op, a, b = words[i:i+4]
            C.add(int(value), op, int(a), int(b), args.native_source)


# ---------------------------------------------------------------- CP-SAT model
def solve_model(p, K, cands, R, hint, tl, nogoods=(), log=None):
    """K: kept value->recipe (None for starts). cands: value->{recipe:tag}. Returns (selected recipes, status, obj)."""
    from ortools.sat.python import cp_model
    hint = {v: canonical(r) for v, r in hint.items()}
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
                    m.Add(lvl[v] >= s + 1).OnlyEnforceIf(y)
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
    m.Minimize((1 + len(x)) * sum(x.values()) + sum(y for (v, r), y in ys.items() if hint.get(v) != r))
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
def repeated_value_schedule(p, rules, reference):
    """Split live intervals at repeated definitions in the incumbent's order."""
    reference_regs = list(p.starts) + [None] * (p.registers-len(p.starts))
    latest = {v: i for i, v in enumerate(p.starts)}
    steps = []
    for op, a, b, dst in reference:
        value = OPS[op](reference_regs[a], reference_regs[b])
        reference_regs[dst] = value
        if value not in rules:
            continue
        selected, left, right = rules[value]
        if left not in latest or right not in latest:
            return None
        steps.append((selected, latest[left], latest[right]))
        latest[value] = len(p.starts) + len(steps)-1
    if not rules.keys() <= latest.keys():
        return None
    uses = Counter(slot for _, left, right in steps for slot in (left, right))
    registers = {i:i for i in range(len(p.starts)) if uses[i]}
    free = [i for i in range(p.registers) if i not in registers]
    program = []
    for i, (op, left, right) in enumerate(steps, len(p.starts)):
        a, b = registers[left], registers[right]
        for slot in (left, right):
            uses[slot] -= 1
            if not uses[slot]:
                free.append(registers.pop(slot))
        if not free:
            return None
        d = free.pop()
        program.append([op, a, b, d])
        if uses[i]:
            registers[i] = d
        else:
            free.append(d)
    replay(p, program)
    return program


def schedule(p, chosen_rec, rank, reference=None):
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
    if reference is not None:
        return repeated_value_schedule(p, chosen_rec, reference)
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
    orig_rec = {v: canonical(g[v]) for v in D}
    t0 = time.perf_counter()
    C = generate(p, K, R, bank_rec, orig_rec, args, rng, strategy == "global")
    gen_t = time.perf_counter() - t0
    srcs = Counter(t for rs in C.rec.values() for t in set(rs.values()))
    row.update(cands=len(C.rec), recipes=sum(map(len, C.rec.values())),
               src="|".join(f"{k}:{v}" for k, v in sorted(srcs.items())), gen_s=round(gen_t, 1))
    keptn = len(K) - len(set(p.starts))
    nogoods = []
    for attempt in range(4):
        chosen, status, obj = solve_model(p, K, C.rec, R, orig_rec, max(0.01, seconds - (time.perf_counter() - t0)), nogoods)
        row.update(status=status, obj=obj)
        if chosen is None:
            row["reason"] = status; return None, None
        row["selected_sources"] = dict(Counter(tag for _, tag in chosen.values()))
        total = keptn + obj
        if total >= len(prog):
            row["reason"] = "no_saving" ; return None, chosen
        rec = {v: r for v, r in g.items() if v not in D}
        rec.update({v: r for v, (r, t) in chosen.items()})
        new = schedule(p, rec, rank, prog)
        if new is not None and len(new) < len(prog):
            row["selected_values"] = {str(v): t for v, (r, t) in chosen.items()
                                      if v not in orig_rec or orig_rec[v] != r}
            tags = sorted({t for v, (r, t) in chosen.items() if v not in orig_rec or orig_rec[v] != r})
            row.update(accepted=1, new_len=len(new), reason="", used_sources="|".join(tags))
            return new, chosen
        row["reason"] = "unscheduled"
        nogoods.append([(v, r) for v, (r, t) in chosen.items()])
    return None, None


# ------------------------------------------------------------------ driver
def prepare_banks(puzzles):
    files = [ROOT / name for name in BANK_FILES] + [BASE]
    fingerprint = hashlib.sha256()
    for path in files:
        fingerprint.update(path.read_bytes())
    cache = WORK / ("banks-" + fingerprint.hexdigest()[:16])
    cache.mkdir(exist_ok=True, parents=True)
    ready = cache / "complete"
    if ready.exists():
        return cache
    from optimize import read_bank
    recipes = {k: defaultdict(set) for k in puzzles}
    for path in files:
        for k, prog in read_bank(path, puzzles).items():
            graph, _ = prog_graph(puzzles[k], prog)
            for v, rule in graph.items():
                recipes[k][v].add(canonical(rule))
    for k, rules in recipes.items():
        with (cache / (k + ".pkl")).open("wb") as stream:
            pickle.dump(dict(rules), stream)
    ready.touch()
    return cache


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
    row.update(src="bank:" + str(len(C.rec)), strategy="all", D=len(g), R=len(R), cands=len(C.rec), recipes=sum(map(len, C.rec.values())))
    deadline = time.monotonic() + seconds
    nogoods = []
    for _ in range(4):
        chosen, status, obj = solve_model(p, K, C.rec, R, g, max(.01, deadline - time.monotonic()), nogoods)
        row.update(status=status, obj=obj)
        if chosen is None: return None
        new = schedule(p, {v: r for v, (r, t) in chosen.items()}, rank, prog)
        if new is not None and len(new) <= len(prog):
            row.pop("reason", None)
            return new
        row["reason"] = "unscheduled" if new is None else "rematerialization_cost"
        nogoods.append([(v, r) for v, (r, t) in chosen.items()])
    return None


if __name__ == "__main__":
    from cpsat_experiments import main
    main()

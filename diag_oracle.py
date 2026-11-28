"""Oracle diagnostic: destroy exactly the values where the 109 and the known 108 differ, then report
which 108-only values/recipes reach the candidate set. Read-only; writes nothing but stdout."""
import random, sys, time
from collections import Counter
import cpsat_lns as L
from cpsat_experiments import parser
from near_reductions import canonical

SOLVE = 300.0
ID = "scored_0379"
puzzles = L.load_puzzles(str(L.ROOT / "puzzles.csv"))
p = puzzles[ID]
p109 = L.read_progs(L.ROOT / "public/ishankarnick/submission.csv", puzzles)[ID]
p108 = L.read_progs(L.ROOT / "submission.csv", puzzles)[ID]
assert len(p109) == 109 and len(p108) == 108
g109, order = L.prog_graph(p, p109); g109 = L.dce(p, g109)
g108, _ = L.prog_graph(p, p108); g108 = L.dce(p, g108)
c109 = {v: canonical(r) for v, r in g109.items()}
c108 = {v: canonical(r) for v, r in g108.items()}
D = {v for v in c109 if c108.get(v) != c109[v]}
new_vals = {v: c108[v] for v in c108 if c109.get(v) != c108[v]}
print("109 values", len(c109), "108 values", len(c108), "|D| (old values changed/removed)", len(D),
      "108-only or changed values", len(new_vals))
print("108 values missing from 109:", sorted(v for v in c108 if v not in c109))

args = parser().parse_args(["exp1"])
# usage: diag_oracle.py [ratio] [mitm|finisher] [hubs=N] [solve=SECONDS]
opts = sys.argv[1:]
args.ratio = "ratio" in opts
args.native_source = next((o for o in opts if o in ("mitm", "finisher")), "")
args.native_timeout = False; args.native_seconds = 30
args.deep = "deep" in opts; args.deep_budget = next((int(o[7:]) for o in opts if o.startswith("budget=")), 2000); args.max_recipes = next((int(o[5:]) for o in opts if o.startswith("recs=")), args.max_recipes); args.max_values = next((int(o[4:]) for o in opts if o.startswith("max=")), args.max_values); args.deep_pairs = "nopairs" not in opts; args.hubs = next((int(o[5:]) for o in opts if o.startswith("hubs=")), args.hubs); args.residual = "nores" not in opts; args.deep_force = next((int(o[6:]) for o in opts if o.startswith("force=")), None); args.deep_routes = next((int(o[7:]) for o in opts if o.startswith("routes=")), 40); args.deep_hubs = next((int(o[5:]) for o in opts if o.startswith("hubn=")), 60)
args.deep_auto = next((int(o[5:]) for o in opts if o.startswith("auto=")), 0); args.deep_hub_budget = next((int(o[5:]) for o in opts if o.startswith("hubb=")), None)
for o in opts:
    if o.startswith("hubs="): args.hubs = int(o[5:])
    if o.startswith("solve="): SOLVE = float(o[6:])
if args.native_source:
    from cpsat_experiments import build_helper
    args.helper = str(build_helper())

cons = L.consumers_of(g109)
rank = {v: i for i, v in enumerate(order)}
targets = set(p.targets)
for closed in ((False,) if "open" in opts else (False, True)):
    Dc = L.close_exclusive(p, g109, D, cons) if closed else set(D)
    R = {v for v in Dc if v in targets or (cons[v] - Dc)}
    K = {v: None for v in p.starts}; K.update({v: r for v, r in g109.items() if v not in Dc})
    orig = {v: canonical(g109[v]) for v in Dc}
    # candidates before prune, to see what pruning removes
    keep_prune = L.Cands.prune
    pre = {}
    def spy(self, required):
        pre.update({v: dict(rs) for v, rs in self.rec.items()}); keep_prune(self, required)
    L.Cands.prune = spy
    t = time.perf_counter()
    C = L.generate(p, K, R, {}, orig, args, random.Random(1), False)
    L.Cands.prune = keep_prune
    print(f"\n== closed={closed} |D|={len(Dc)} |R|={len(R)} cands pre-prune={len(pre)} post-prune={len(C.rec)} "
          f"gen {time.perf_counter()-t:.1f}s")
    need = {v: r for v, r in new_vals.items() if v in Dc or v not in c109}
    for v, r in sorted(need.items(), key=lambda kv: rank.get(kv[0], 0)):
        where = ("kept" if v in K else
                 "FINAL-recipe-present" if r in C.rec.get(v, {}) else
                 "value-present-other-recipe" if v in C.rec else
                 "pruned" if r in pre.get(v, {}) else
                 "value-pre-prune-only" if v in pre else "NEVER-GENERATED")
        tag = C.rec.get(v, {}).get(r) or pre.get(v, {}).get(r) or ""
        print(f"  {v:>14} {r[0]} {r[1]} {r[2]}  -> {where} {tag}")
    t = time.perf_counter()
    print("  deep route sizes:", sorted(getattr(args, "deep_stats", {}).items())[:12])
    vt = getattr(args, "deep_votes", {})
    ranks = {v: (sorted(vt.values(), reverse=True).index(vt[v]) if v in vt else None, vt.get(v)) for v in (787994, 11827, 93610705, 92822711, 1707449, 1712137)}
    print("  votes (rank,votes):", ranks)
    allr = getattr(args, "deep_all", {})
    for v in (93610705, 92822711, 1707449, 1712137):
        n = sum(1 for recs in allr.values() if any(x == v for x, *_ in recs))
        print(f"  routes containing {v}: {n}")
    both = [recs for recs in allr.values() if any(x == 93610705 for x, *_ in recs) and any(x == 787994 for x, *_ in recs)]
    print("  routes with 93610705 and 787994:", [(len({x for x, *_ in r} - set(K)), r) for r in both][:3])
    print("  routes containing 787994:", sum(1 for r in allr.values() if any(x == 787994 for x, *_ in r)))
    vt2 = args.deep_votes
    hubs_dbg = sorted(vt2, key=lambda v: (-vt2[v], v))[:args.deep_hubs]
    print("  787994 in hubs:", 787994 in hubs_dbg, "deep_hubs", args.deep_hubs, "routes", args.deep_routes, "budget", args.deep_budget, "max_values", args.max_values, "max_recipes", args.max_recipes, "C.count", C.count)
    from collections import Counter as _C
    print("  vote histogram:", sorted(_C(vt2.values()).items(), reverse=True))
    for recs in both:
        print("  true route rank:", args.deep_rank.get(frozenset(recs)))
    present = sum(1 for v, r in need.items() if v in K or r in C.rec.get(v, {}))
    print(f"  FINAL recipes present: {present}/{len(need)}")
    if not closed and "open" not in opts:
        continue
    for o in opts:
        if o.startswith("sample="):
            rng2 = random.Random(5)
            n = int(o[7:])
            truth = {v: {r: "true"} for v, r in new_vals.items()}
            others = [v for v in C.rec if v not in truth and v not in orig]
            keep = set(rng2.sample(others, min(n, len(others))))
            C.rec = {v: dict(rs) for v, rs in C.rec.items() if v in keep or v in orig}
            for v, rs in truth.items():
                C.rec.setdefault(v, {}).update(rs)
            for v, r in orig.items():
                C.rec.setdefault(v, {})[r] = "orig"
            print("  sampled candidates:", len(C.rec))
    chosen, status, obj = L.solve_model(p, K, C.rec, R, orig, SOLVE, [])
    keptn = len(K) - len(set(p.starts))
    print("  solve:", status, "obj", obj, "total", None if chosen is None else keptn + obj,
          f"{time.perf_counter()-t:.1f}s",
          dict(Counter(tg for _, tg in chosen.values())) if chosen else "")

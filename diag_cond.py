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
for closed in (True,):
    Dc = L.close_exclusive(p, g109, D, cons) if closed else set(D)
    R = {v for v in Dc if v in targets or (cons[v] - Dc)}
    K = {v: None for v in p.starts}; K.update({v: r for v, r in g109.items() if v not in Dc})
    orig = {v: canonical(g109[v]) for v in Dc}
    HUB, TARGET = 11827, 787994
    print("recipe of", HUB, c108.get(HUB), "in K?", HUB in K, "in Dc?", HUB in Dc, "recipe of", TARGET, c108.get(TARGET))
    def rank_report(tag, K_, R_):
        args.deep_force = None; args.deep_auto = 0
        t = time.perf_counter()
        L.generate(p, K_, R_, {}, orig, args, random.Random(1), False)
        vt = args.deep_votes
        order_ = sorted(vt, key=lambda v: (-vt[v], v))
        pos = order_.index(TARGET) if TARGET in vt else None
        print(f"{tag}: |K|={len(K_)} |R|={len(R_)} routes={len(args.deep_all)} candidates-voted={len(vt)} "
              f"{TARGET} rank={pos} votes={vt.get(TARGET)} top5={[(v, vt[v]) for v in order_[:5]]} {time.perf_counter()-t:.0f}s", flush=True)
    rank_report("baseline", K, R)
    K2 = dict(K); stack = [HUB]
    while stack:
        v = stack.pop()
        if v in K2 or v in p.starts: continue
        K2[v] = tuple(c108[v]); stack += [x for x in c108[v][1:] if isinstance(x, int)]
    print("hub chain added:", [(v, K2[v]) for v in K2 if v not in K])
    rank_report("with-hub", K2, R - (set(K2) - set(K)))

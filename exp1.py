import os, sys, random, time, json, argparse
os.environ.setdefault("CPSAT_CACHE", os.path.expanduser("~/.tmp/bank_cache.pkl"))
os.sched_setaffinity(0, {108})
import cpsat_lns as L
from elf import load_puzzles
P = load_puzzles("puzzles.csv"); p = P["scored_0379"]
banks = L.load_banks(P)["scored_0379"]
old = L.read_progs("public/ishankarnick/submission.csv", P)["scored_0379"]
print("old", len(old))
args = argparse.Namespace(hubs=200, residual=1, ratio=0, max_residual=3000, max_recipes=50000, dmin=10, dmax=25)
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
t0 = time.time(); it = 0
while time.time() - t0 < 900:
    for s in ["hub", "output", "region", "disagree"]:
        row = {}; it += 1
        # positive control: no seeding of the known hub -> drop this puzzle's banks except what the generators produce
        new, ch = L.neighbourhood(p, old, s, banks if "--bank" in sys.argv else {}, args, rng, 30, row)
        print(it, round(time.time() - t0), row, flush=True)
        if new: print("FOUND", len(new)); json.dump(new, open("experiments/cpsat-lns-work/exp1-prog.json", "w")); sys.exit()

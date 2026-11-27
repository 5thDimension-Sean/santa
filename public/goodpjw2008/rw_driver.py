"""Run Ross Wade's slack-annealing engine (third_party/rosswade, CC0) on our best programs, with our plateau archive and the
current public programs as its variant pool. Every returned program is replayed by our referee; only strictly shorter ones
are merged into work/best.csv.
usage: python src/rw_driver.py --only ids.txt --secs 60 --jobs 10 [--slack 2] [--max-var 24]
"""
import argparse, glob, json, os, random, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from elf import *
ap = argparse.ArgumentParser()
ap.add_argument("--only", required=True); ap.add_argument("--secs", type=float, default=60); ap.add_argument("--jobs", type=int, default=10)
ap.add_argument("--slack", type=int, default=2); ap.add_argument("--max-var", type=int, default=24); ap.add_argument("--vslack", type=int, default=3)
ap.add_argument("--kick", type=int, default=0); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--tag", default="rw"); ap.add_argument("--banks", default="outs_cycle/*/*.csv")
a = ap.parse_args()
P = load_puzzles(); BEST = os.path.join(ROOT, "work/best.csv"); S = read_submission(BEST)
EXE = os.path.join(ROOT, "third_party/rosswade/sa_engine")
OPI = {"+": 0, "-": 1, "*": 2, "/": 3}; IOP = {v: k for k, v in OPI.items()}
ids = [l.strip() for l in open(a.only) if l.strip() in P]; random.shuffle(ids); idset = set(ids)
var = {k: {} for k in ids}
def offer(k, p):
    if k in idset and isinstance(p, list) and len(p) <= len(S[k]) + a.vslack and p != S[k]: var[k].setdefault(json.dumps(p), p)
if os.path.exists(os.path.join(ROOT, "work/archive.jsonl")):
    for ln in open(os.path.join(ROOT, "work/archive.jsonl")):
        try: r = json.loads(ln)
        except Exception: continue
        offer(r.get("k"), r.get("p"))
for f in glob.glob(os.path.join(ROOT, a.banks)):
    try: B = read_submission(f)
    except Exception: continue
    for k in ids: offer(k, B.get(k))
flat = lambda p: " ".join(f"{OPI[o]} {i} {j} {d}" for o, i, j, d in p)
D = tempfile.mkdtemp(prefix="rw_", dir=os.path.join(ROOT, "work"))
picked = {}
for k in ids:
    vs = list(var[k].values()); random.shuffle(vs); picked[k] = []
    for v in vs:
        if len(picked[k]) >= a.max_var: break
        try: replay(P[k], v)
        except Exception: continue
        picked[k].append(v)

def run(job):
    """One engine process per (puzzle, repeat): a work queue keeps every core busy until the last puzzle starts."""
    i, k = job; p = P[k]; t = sorted(set(p.targets))
    with open(f"{D}/bank_{i}.txt", "w") as f:
        f.write(f"P {k} {p.limit} {p.registers}\nS {len(p.starts)} {' '.join(map(str, p.starts))}\nT {len(t)} {' '.join(map(str, t))}\nG {len(S[k])} {flat(S[k])}\n")
    with open(f"{D}/var_{i}.txt", "w") as g:
        for v in picked[k]: g.write(f"V {k} {len(v)} {flat(v)}\n")
    env = {"PATH": os.environ.get("PATH", ""), "SA_VARIANTS": f"{D}/var_{i}.txt", "SA_SEED": str(random.randrange(1, 10**9))}
    if a.kick: env["SA_KICK"] = str(a.kick)
    with open(f"{D}/log_{i}.txt", "w") as lg:
        return subprocess.run([EXE, f"{D}/bank_{i}.txt", f"{D}/out_{i}.txt", str(a.secs), "0", "1", "-", str(a.slack)], env=env,
                              stdout=lg, stderr=subprocess.STDOUT).returncode

t0 = time.time()
with ThreadPoolExecutor(a.jobs) as pool: rcs = list(pool.map(run, enumerate(ids * max(1, a.repeat))))
found, bad = {}, 0
for f in glob.glob(f"{D}/out_*.txt"):
    for ln in open(f):
        x = ln.split()
        if len(x) < 3 or x[0] != "R" or x[1] not in P: continue
        v = list(map(int, x[3:])); prog = [[IOP[v[i]], v[i + 1], v[i + 2], v[i + 3]] for i in range(0, len(v), 4)]
        try: n = replay(P[x[1]], prog)
        except Exception: bad += 1; continue
        if n < len(S[x[1]]) and (x[1] not in found or n < len(found[x[1]])): found[x[1]] = prog
cur = read_submission(BEST); gain = 0
for k, p in found.items():
    if len(p) < len(cur[k]): gain += len(cur[k]) - len(p); cur[k] = p
if gain:
    tmp = BEST + ".tmp"; write_submission(tmp, P, cur); os.replace(tmp, BEST)
tot = sum(len(cur[k]) for k in P)
open(os.path.join(ROOT, "work/history.csv"), "a").write(f"{time.time():.0f},{a.tag}_rosswade_sa,{tot}\n")
done = sum(1 for f in glob.glob(f"{D}/log_*.txt") for ln in open(f) if " start=" in ln)
print(f"[{time.strftime('%H:%M:%S')}] rosswade_sa{'_fresh' if 'fresh' in a.tag else ''} secs={a.secs} puzzles={len(ids)} done={done} variants={sum(min(len(v), a.max_var) for v in var.values())} "
      f"improved={len(found)} gain={gain} invalid={bad} errors={sum(r != 0 for r in rcs)} total={tot} ({time.time() - t0:.0f}s)", flush=True)

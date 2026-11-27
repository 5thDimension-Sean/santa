"""Production driver for triwalk: polishes our best programs (and equal-length archive variants) and alternative lineages.
Every returned program is replayed by our referee; strictly shorter ones are merged into work/best.csv at once, improved
lineages go back to the pool (work/pool.jsonl) so later rounds continue from them.
usage: python src/tri_driver.py --mode lineage --secs 30 --jobs 19 [--max-excess 6] [--budget 1800]
       python src/tri_driver.py --mode best --only ids.txt --secs 30 [--repeat 2]
"""
import argparse, os, sys, subprocess, time, json, random, threading, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from pfmt import *
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="best"); ap.add_argument("--only", default=""); ap.add_argument("--secs", type=float, default=30)
ap.add_argument("--jobs", type=int, default=19); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--tag", default="tri")
ap.add_argument("--max-excess", type=int, default=6); ap.add_argument("--budget", type=float, default=1e9); ap.add_argument("--extra", default="--cap 1 --T 0.2 --rb 0.6 --deep 0.1;--cap 1 --T 0.2 --rb 0.85 --deep 0.4;--cap 2 --T 0.2 --rb 0.6 --deep 0.1;--cap 2 --T 0.2 --rb 0.85 --deep 0.4;--cap 3 --T 0.2 --rb 0.85 --deep 0.2;--cap 2 --T 0.2 --rb 0.85 --deep 0")
ap.add_argument("--variant-prob", type=float, default=0.5); ap.add_argument("--sample", type=int, default=0)
a = ap.parse_args()
P = load_puzzles(); BEST = os.path.join(ROOT, "work/best.csv"); HIST = os.path.join(ROOT, "work/history.csv"); POOL = os.path.join(ROOT, "work/pool.jsonl"); ARCH = os.path.join(ROOT, "work/archive.jsonl")
EXE = os.path.join(ROOT, "bin/triwalk"); S = read_submission(BEST); lock = threading.Lock(); T0 = time.time()
ids = [l.strip() for l in open(a.only) if l.strip() in P] if a.only else [k for k in P if len(S[k]) >= 20]
idset = set(ids)
jobs = []   # (k, program, pool index or -1)
pool = []
if a.mode == "lineage":
    src = POOL if os.path.exists(POOL) else os.path.join(ROOT, "work/lineages.jsonl")
    pool = [json.loads(l) for l in open(src)]
    seen = {json.dumps(r["p"]) for r in pool}
    if src == POOL:   # new lineages collected since
        for l in open(os.path.join(ROOT, "work/lineages.jsonl")):
            r = json.loads(l)
            if r["k"] not in P or len(r["p"]) > len(S[r["k"]]) + a.max_excess: continue
            key = json.dumps(r["p"])
            if key not in seen and r.get("src") not in {x.get("src") for x in pool if x["k"] == r["k"]}: pool.append(r); seen.add(key)
    for i, r in enumerate(pool):
        if r["k"] in idset and 0 <= len(r["p"]) - len(S[r["k"]]) <= a.max_excess and len(r["p"]) >= 20: jobs.append((r["k"], r["p"], i))
    w = lambda j: (pool[j[2]].get("tries", 0) + 1) * (1 + 0.5 * max(0, len(j[1]) - len(S[j[0]]))) * (0.5 + random.random())
    jobs.sort(key=w)
else:
    alts = collections.defaultdict(list)
    if os.path.exists(ARCH):
        for ln in open(ARCH):
            try: r = json.loads(ln)
            except Exception: continue
            k = r.get("k")
            if k in idset and len(r["p"]) == len(S[k]) and r["p"] != S[k]: alts[k].append(r["p"])
    for rep in range(a.repeat):
        for k in ids: jobs.append((k, random.choice(alts[k]) if alts[k] and random.random() < a.variant_prob else S[k], -1))
    random.shuffle(jobs)
if a.sample: jobs = jobs[:a.sample]
stat = collections.Counter(); newvar = []; CFG = [c.strip() for c in a.extra.split(";") if c.strip()]
def rec(k, cfg, n0, n1, b0):
    with lock:
        with open(os.path.join(ROOT, "work/tri_stats.csv"), "a") as f: f.write(f"{time.time():.0f},{a.mode},{a.tag},{k},{cfg},{a.secs},{n0},{n1},{b0}\n")

def merge(k, prog):
    with lock:
        cur = read_submission(BEST)
        if len(prog) < len(cur[k]):
            g = len(cur[k]) - len(prog); cur[k] = prog; S[k] = prog
            tmp = BEST + ".tri.tmp"; write_submission(tmp, P, cur); os.replace(tmp, BEST)
            with open(ARCH, "a") as f: f.write(json.dumps({"k": k, "p": prog}, separators=(",", ":")) + "\n")
            stat["gain"] += g; stat["improved"] += 1
            return g
    return 0

def run(job):
    k, prog, idx = job
    if time.time() - T0 > a.budget: return
    cfg = random.choice(CFG)   # configurations are compared from the production log (work/tri_stats.csv)
    try: r = subprocess.run([EXE, "--time", str(a.secs), "--seed", str(random.randrange(1, 10**9))] + cfg.split(), input=encode(P[k], prog), capture_output=True, text=True, timeout=a.secs * 4 + 60)
    except subprocess.TimeoutExpired: stat["timeout"] += 1; return
    stat["runs"] += 1
    if r.returncode:
        stat["errors"] += 1
        with open(os.path.join(ROOT, "work/tri_err.log"), "a") as f: f.write(json.dumps({"k": k, "rc": r.returncode, "err": r.stderr[-300:], "args": r.args, "p": prog}) + "\n")
        return
    txt = "\n".join("R" + ln[1:] if ln.startswith("V ") else ln for ln in r.stdout.split("\n"))   # V = equal-length end state
    got = parse_results(txt).get(k)
    if idx >= 0: pool[idx]["tries"] = pool[idx].get("tries", 0) + 1
    if not got: rec(k, cfg, len(prog), len(prog), len(S[k])); return
    try: n = replay(P[k], got)
    except ValueError: stat["invalid"] += 1; return
    rec(k, cfg, len(prog), n, len(S[k]))
    if idx >= 0 and n < len(pool[idx]["p"]): stat["lin_gain"] += len(pool[idx]["p"]) - n; pool[idx]["p"] = got; pool[idx]["len"] = n; pool[idx]["tries"] = 0
    if idx >= 0 and n == len(pool[idx]["p"]) and got != pool[idx]["p"]: pool[idx]["p"] = got   # keep walking from the new state
    if n < len(S[k]): merge(k, got)
    elif n == len(S[k]) and got != S[k]: newvar.append((k, got))

with ThreadPoolExecutor(a.jobs) as ex: list(ex.map(run, jobs))
if newvar:
    with open(ARCH, "a") as f:
        for k, g in newvar: f.write(json.dumps({"k": k, "p": g}, separators=(",", ":")) + "\n")
if a.mode == "lineage":
    with open(POOL + ".tmp", "w") as f:
        for r in pool: f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(POOL + ".tmp", POOL)
cur = read_submission(BEST); tot = sum(len(cur[k]) for k in P)
open(HIST, "a").write(f"{time.time():.0f},{a.tag}_triwalk_{a.mode},{tot}\n")
print(f"[{time.strftime('%H:%M:%S')}] triwalk_{a.mode}{'_fresh' if 'fresh' in a.tag else ''} secs={a.secs} puzzles={len(jobs)} runs={stat['runs']} lineage_gain={stat['lin_gain']} improved={stat['improved']} "
      f"gain={stat['gain']} invalid={stat['invalid']} errors={stat['errors']} total={tot} ({time.time() - T0:.0f}s)", flush=True)

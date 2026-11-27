"""Run Ross Wade's GPU exact-repair LNS (third_party/rosswade/lns_gpu.py + gpu_mitm2.py, CC0, unchanged) on our programs.
One process, one GPU. Starts from our best or a random equal-length variant of it; every result is replayed by our own
referee (src/elf.py) and only strictly shorter programs are merged into work/best.csv.
usage: third_party/rosswade/venv/bin/python -I src/gpu_lns.py ids.txt secs_per_puzzle [minutes]
"""
import json, os, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT + "/src", ROOT + "/third_party/rosswade"]
from elf import load_puzzles, read_submission, write_submission, replay
import lns_gpu as G
from gpu_mitm2 import GPUMITM2
ids_file, secs, minutes = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
P = load_puzzles(); BEST = ROOT + "/work/best.csv"; rng = random.Random(); solver = GPUMITM2()
ids = [l.strip() for l in open(ids_file) if l.strip() in P]; rng.shuffle(ids)
alts = {}
for ln in open(ROOT + "/work/archive.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("k") in P: alts.setdefault(r["k"], []).append(r["p"])
end = time.time() + minutes * 60; T = dict(it=0, acc=0, eq=0, gain=0, n=0)
def merge(k, prog):
    cur = read_submission(BEST)
    if len(prog) >= len(cur[k]): return 0
    g = len(cur[k]) - len(prog); cur[k] = prog
    tmp = BEST + ".gpu.tmp"; write_submission(tmp, P, cur); os.replace(tmp, BEST)
    open(ROOT + "/work/history.csv", "a").write(f"{time.time():.0f},gpu_rosswade_lns,{sum(len(v) for v in cur.values())}\n")
    return g
for k in ids:
    if time.time() > end: break
    S = read_submission(BEST); p = P[k]; L0 = len(S[k])
    starts = [S[k]] + [v for v in alts.get(k, []) if len(v) == L0]
    prog0 = rng.choice(starts) if rng.random() < 0.5 else S[k]
    D = dict(regs=p.registers, starts=list(p.starts), targets=sorted(set(p.targets)), limit=p.limit)
    try: replay(p, prog0)
    except Exception: prog0 = S[k]
    nodes = G.to_nodes(D, prog0); best = L0; t0 = time.time(); it = acc = eq = g = 0
    while time.time() - t0 < secs:
        it += 1
        nn, d = G.lns_iteration(D, nodes, solver, rng)
        if nn is None: continue
        nodes = nn; acc += 1; eq += d == 0
        n = sum(1 for x in nodes if x[1] is not None)
        if n < best:
            prog = G.compile_nodes(D, nodes)
            if prog is not None:
                prog = [list(x) for x in prog]
                try:
                    if replay(p, prog) == len(prog) < best: g += merge(k, prog); best = len(prog)
                except Exception: pass
    for a, b in zip(("it", "acc", "eq", "gain", "n"), (it, acc, eq, g, 1)): T[a] += b
    print(f"[{time.strftime('%H:%M:%S')}] gpu_lns {k} {L0}->{best} iters={it} accepted={acc} equal={eq} gain={g} | run: puzzles={T['n']} iters={T['it']} total_gain={T['gain']}", flush=True)

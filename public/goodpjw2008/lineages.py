"""Collect alternative lineages: for every puzzle, programs from all downloaded public files and our archive that are close in
length to our best but share few intermediate values with it. Writes work/lineages.jsonl ({k, len, best, ov, p, src})."""
import sys, glob, os, json, collections, time
sys.path.insert(0, os.path.dirname(__file__))
from elf import *
P = load_puzzles(); S = read_submission(os.path.join(ROOT, "work/best.csv"))
fam = {l.split(',')[-1].strip(): l.split(',')[-3] for l in open(os.path.join(ROOT, "work/scored_feats.csv"))}
REL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
MAXOV = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7

def mids(k, prog):
    p = P[k]; r = list(p.starts) + [None] * (p.registers - len(p.starts)); out = set(); T = set(p.targets)
    for op, i, j, d in prog:
        a, b = r[i], r[j]
        v = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a // b
        r[d] = v
        if v not in T: out.add(v)
    return out

cand = collections.defaultdict(dict)   # k -> frozenset(mids) -> (len, prog, src)
fs = sorted(glob.glob(os.path.join(ROOT, "outs*/**/*.csv"), recursive=True), key=os.path.getmtime); t0 = time.time(); nb = 0
for f in fs:
    try: B = read_submission(f)
    except Exception: continue
    nb += 1
    for k, prog in B.items():
        if k not in P or not isinstance(prog, list): continue
        if len(prog) > len(S[k]) * (1 + REL) + 1: continue
        key = hash(json.dumps(prog)) 
        if key in cand[k]: continue
        cand[k][key] = (len(prog), prog, os.path.relpath(f, ROOT))
for ln in open(os.path.join(ROOT, "work/archive.jsonl")):
    try: r = json.loads(ln)
    except Exception: continue
    k = r.get("k"); prog = r.get("p")
    if k in P and isinstance(prog, list) and len(prog) <= len(S[k]) * (1 + REL) + 1: cand[k].setdefault(hash(json.dumps(prog)), (len(prog), prog, "archive"))
print("files", nb, "programs", sum(len(v) for v in cand.values()), f"{time.time()-t0:.0f}s", flush=True)
out = open(os.path.join(ROOT, "work/lineages.jsonl"), "w"); n = 0; cnt = collections.Counter(); perfam = collections.Counter()
for k in P:
    base = mids(k, S[k]); reps = [base]
    for L, prog, src in sorted(cand[k].values(), key=lambda x: x[0]):
        try: replay(P[k], prog); m = mids(k, prog)
        except Exception: continue
        if not m: continue
        if all(len(m & r) / max(1, min(len(m), len(r))) < MAXOV for r in reps):
            reps.append(m); ov = len(m & base) / max(1, min(len(m), len(base)))
            out.write(json.dumps({"k": k, "len": L, "best": len(S[k]), "ov": round(ov, 3), "src": src, "p": prog}, separators=(",", ":")) + "\n"); n += 1; perfam[fam[k]] += 1; cnt[min(12, L - len(S[k]))] += 1
            if len(reps) > 8: break
print("lineages", n, dict(perfam), "excess length:", sorted(cnt.items()))

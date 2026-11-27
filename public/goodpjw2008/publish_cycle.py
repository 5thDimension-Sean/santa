"""One publication cycle (only proceeds if we are strictly ahead of the public frontier):
refresh public outputs -> merge into work/best.csv -> build notebook -> local test -> dataset version -> push ->
verify the Kaggle run -> submit the notebook's own output (Public Score chip) -> re-pin the kernel id.
usage: python src/publish_cycle.py [--force] [--min-lead 1] [--skip-local-test]
"""
import argparse, os, sys, subprocess, glob, json, time, collections, re, shutil
sys.path.insert(0, os.path.dirname(__file__))
from elf import *

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true"); ap.add_argument("--min-lead", type=int, default=1)
ap.add_argument("--skip-local-test", action="store_true"); ap.add_argument("--kernels", type=int, default=50)
ap.add_argument("--lead", type=int, default=2, help="target lead over the public frontier per publication")
ap.add_argument("--dry", action="store_true"); ap.add_argument("--builder", default="src/make_notebook.py"); ap.add_argument("--nbdir", default="deliver/notebook")
ap.add_argument("--lastf", default="work/last_published.txt"); ap.add_argument("--changelog", default="work/changelog.json")
ap.add_argument("--min-gain", type=int, default=0, help="also require tot <= last published - min_gain, unless the public frontier caught up")
a = ap.parse_args()

env = dict(os.environ)
for line in open(os.path.join(ROOT, ".env")):
    if "=" in line: k, v = line.strip().split("=", 1); env[k] = v
env["KAGGLE_API_TOKEN"] = env["KAGGLE_API_KEY"]; USER = env["KAGGLE_USERNAME"]
KG = os.path.join(ROOT, ".venv/bin/kaggle"); PY = os.path.join(ROOT, ".venv/bin/python")
def kg(*args, timeout=600):
    r = subprocess.run([KG, *args], env=env, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr
def log(*x): print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)

P = load_puzzles()
# 1. refresh public outputs
out = kg("kernels", "list", "--competition", "santa-2026", "--sort-by", "dateRun", "--page-size", str(a.kernels))
meta = {}
for ln in out.splitlines()[2:]:
    parts = re.split(r"\s{2,}", ln.strip())
    if len(parts) >= 3 and "/" in parts[0] and not parts[0].startswith(USER + "/"):
        mt = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?", ln)
        meta[parts[0]] = {"title": parts[1], "author": parts[2], "run": mt.group(0) if mt else ""}
if not meta: log("kernel list failed:", out.strip()[-300:]); sys.exit(1)
# outputs are cached per notebook and re-downloaded only when its lastRunTime changed
OD = os.path.join(ROOT, "outs_cycle"); os.makedirs(OD, exist_ok=True)
srcs = {}; fresh = 0
for ref in meta:
    d = os.path.join(OD, ref.split("/")[1]); stamp = os.path.join(d, ".run")
    cached = os.path.exists(stamp) and open(stamp).read() == meta[ref]["run"] and meta[ref]["run"]
    if not cached:
        try: st = kg("kernels", "status", ref, timeout=60)
        except subprocess.TimeoutExpired: st = ""
        if re.search(r"complete", st, re.I):   # still running / failed -> keep the previous cached output
            shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
            try: kg("kernels", "output", ref, "-p", d, timeout=300)
            except subprocess.TimeoutExpired: continue
            if glob.glob(f"{d}/*.csv"): open(stamp, "w").write(meta[ref]["run"]); fresh += 1
    for f in glob.glob(f"{d}/*.csv"):
        try: S = read_submission(f)
        except Exception: continue
        sd = score_dict(P, S)
        if all(v is not None for v in sd.values()): srcs[ref] = S; break
from_source = set()
# notebooks that publish programs only in their source / attached datasets (no submission file in the output)
from src_extract import candidates, best_valid
base = {k: min((S[k] for S in srcs.values()), key=len) for k in P} if srcs else None
for ref in meta:
    if ref in srcs or base is None: continue
    d = os.path.join(OD, ref.split("/")[1], "_src"); stamp = os.path.join(d, ".run")
    if not (os.path.exists(stamp) and open(stamp).read() == meta[ref]["run"] and meta[ref]["run"]):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
        try:
            kg("kernels", "pull", ref, "-p", d, "-m", timeout=120)
            km = json.load(open(os.path.join(d, "kernel-metadata.json")))
            for ds in km.get("dataset_sources", [])[:6]:
                kg("datasets", "download", ds, "-p", os.path.join(d, "ds_" + ds.replace("/", "__")), "--unzip", timeout=300)
            open(stamp, "w").write(meta[ref]["run"])
        except Exception as e: log("source pull failed", ref, e); continue
    got_src = best_valid(P, candidates(d))
    if len(got_src) >= 990 or any(len(v) < len(base[k]) for k, v in got_src.items()):
        S = dict(base); S.update(got_src); srcs[ref] = S; from_source.add(ref)
        log(f"source programs from {ref}: {sum(len(base[k]) - len(v) for k, v in got_src.items() if len(v) < len(base[k]))} shorter")
ladder = sorted(((r.split("/")[1], sum(len(S[k]) for k in P)) for r, S in srcs.items()), key=lambda x: x[1])
json.dump(ladder, open(os.path.join(ROOT, "work/ladder.json"), "w"))
json.dump([dict(ref=r, title=meta[r]["title"], author=meta[r]["author"], total=sum(len(S[k]) for k in P),
                kind="source" if r in from_source else "output") for r, S in srcs.items()],
          open(os.path.join(ROOT, "work/ladder_full.json"), "w"), indent=1)
pub = {}
for k in P: pub[k] = min((S[k] for S in srcs.values()), key=len)
p_tot = sum(len(pub[k]) for k in P)
def title_score(t):
    v = [int(a + b) for a, b in re.findall(r"(?<!\d)(1[23]\d)[,.]?(\d{3})(?!\d)", t)]
    return min((x for x in v if 120000 <= x <= 140000), default=None)
rivals = {r: sum(len(S[k]) for k in P) for r, S in srcs.items()}
for r in meta:
    ts = title_score(meta[r]["title"])
    if ts and p_tot - 20 <= ts < rivals.get(r, 10**9): rivals[r] = ts   # implausible claims far below the frontier are ignored
rival_ref = min(rivals, key=rivals.get); rival = rivals[rival_ref]
write_submission(os.path.join(ROOT, "work/public_cycle.csv"), P, pub)
hold = collections.Counter(r for k in P for r, S in srcs.items() if len(S[k]) == len(pub[k]))
holders = [dict(ref=r, title=meta[r]["title"], author=meta[r]["author"], holds=c) for r, c in hold.most_common(5)]
json.dump(holders, open(os.path.join(ROOT, "work/frontier_holders.json"), "w"), indent=1)
# 2. merge: full private best (public + Hansen0223 GitHub checkpoint + our finds) -> work/best.csv
try:
    hd = os.path.join(ROOT, "refs3/hansen")
    subprocess.run(["git", "-C", hd, "fetch", "--depth", "1", "origin"], capture_output=True, timeout=180)
    subprocess.run(["git", "-C", hd, "reset", "--hard", "origin/HEAD"], capture_output=True, timeout=60)
    for src, dst in (("public-assisted/submission.csv", "outs_hansen/public-assisted"), ("submission.csv", "outs_hansen/independent")):
        if os.path.exists(os.path.join(hd, src)):
            os.makedirs(os.path.join(ROOT, dst), exist_ok=True); shutil.copy(os.path.join(hd, src), os.path.join(ROOT, dst, "submission.csv"))
except Exception as e:
    log("hansen refresh failed", e)
best = read_submission(os.path.join(ROOT, "work/best.csv"))
for f in glob.glob(os.path.join(ROOT, "outs_hansen/*/submission.csv")):
    try: H = read_submission(f)
    except Exception: continue
    for k in P:
        if k in H and len(H[k]) < len(best[k]):
            try: replay(P[k], H[k]); best[k] = H[k]
            except ValueError: pass
m = {k: (best[k] if len(best[k]) <= len(pub[k]) else pub[k]) for k in P}
full = sum(replay(P[k], m[k]) for k in P)
write_submission(os.path.join(ROOT, "work/best.csv.tmp"), P, m); os.replace(os.path.join(ROOT, "work/best.csv.tmp"), os.path.join(ROOT, "work/best.csv"))
now = int(time.time())
open(os.path.join(ROOT, "work/history.csv"), "a").write(f"{now},merge_public_cycle,{full}\n")
t0 = int(open(os.path.join(ROOT, "work/history.csv")).readline().split(",")[0])
open(os.path.join(ROOT, "work/public_history.csv"), "a").write(f"{(now - t0) / 60:.1f},{p_tot}\n")
# reserve strategy: publish only as many of our unique improvements as needed for a lead of a.lead over the
# public frontier; keep the rest private until the public notebooks catch up again.
LASTF = os.path.join(ROOT, a.lastf)
last = int(open(LASTF).read().split()[0]) if os.path.exists(LASTF) else 10**9
uniq = sorted(((len(pub[k]) - len(m[k]), k) for k in P if len(m[k]) < len(pub[k])))
reserve = sum(g for g, _ in uniq)
# release ladder: besides our best program, any valid intermediate-length program (from the plateau archive /
# earlier snapshots) can be released, so a lead of a.lead costs about a.lead instead of a whole puzzle's saving
need = {k for _, k in uniq}; inter = collections.defaultdict(dict)
def offer(k, prog):
    if k in need and len(m[k]) < len(prog) < len(pub[k]) and len(prog) not in inter[k]:
        try: replay(P[k], prog); inter[k][len(prog)] = prog
        except Exception: pass
try:
    for ln in open(os.path.join(ROOT, "work/archive.jsonl")):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("k") in need: offer(r["k"], r["p"])
except FileNotFoundError: pass
for f in [os.path.join(ROOT, "work/best_snapshot.csv")] + glob.glob(os.path.join(ROOT, "outs_hansen/*/submission.csv")):
    try: B = read_submission(f)
    except Exception: continue
    for k in need:
        if k in B: offer(k, B[k])
dp = {0: []}                                    # exact saving -> [(puzzle, program)], one option per puzzle
for g, k in uniq:
    opts = [(len(pub[k]) - L, prog) for L, prog in inter[k].items()] + [(g, m[k])]
    nd = dict(dp)
    for s0, ch in dp.items():
        for d, prog in opts:
            if s0 + d not in nd: nd[s0 + d] = ch + [(k, prog)]
    dp = nd
need_below = max(0, a.lead - (rival - p_tot))   # merging every public notebook may already beat each single one
reach = [s0 for s0 in dp if s0 >= need_below]
got = min(reach) if reach else max(dp)
chosen = [k for k, _ in dp[got]]
pubset = dict(pub)
for k, prog in dp[got]: pubset[k] = prog
tot = sum(replay(P[k], pubset[k]) for k in P)
write_submission(os.path.join(ROOT, "work/sub_latest.csv"), P, pubset)
lead = rival - tot
caught_up = rival < last or (rival == last and lead >= 1)   # break a tie whenever we can get ahead (own saving or a better merge)
log(f"public frontier {p_tot:,} | best single notebook {rival:,} ({rival_ref.split('/')[0]}) | our notebook {last:,} | private best {full:,} (reserve {reserve}) | would publish {tot:,} "
    f"(lead {lead}, uses {got}) | caught up: {caught_up} | holders {[h['ref'] for h in holders]}")
if a.dry: log("dry run -> stop before building"); sys.exit(0)
if not a.force and not caught_up:
    log("our published notebook is still ahead -> nothing published"); sys.exit(0)
if lead < 1 and not a.force and not (lead == 0 and rival < last):   # when behind, at least tie the best single notebook
    log("no unique improvement left to take the lead -> nothing published"); sys.exit(0)
# daily submission cap (100/day, resets 00:00 UTC): keep a small margin
today = time.strftime("%Y-%m-%d", time.gmtime())
n_today = sum(1 for ln in kg("competitions", "submissions", "santa-2026", "--csv", "--page-size", "200").splitlines() if f",{today} " in ln)
if n_today >= 97: log(f"daily submission cap: {n_today} submissions today (UTC) -> nothing published"); sys.exit(0)
# pacing: spread a daily budget (BUDGET of the 100 submissions) evenly until the reset (00:00 UTC); after that a 45-min floor;
# a rival more than 20 ahead may be answered at once (still under the hard cap above)
BUDGET = 70
mins_left = (86400 - time.time() % 86400) / 60
min_gap = mins_left / (BUDGET - n_today) if n_today < BUDGET else 45
if rival < last: min_gap = min(min_gap, 6)      # strictly behind a rival: answer after 6 minutes; ties keep the even pacing
if rival <= last - 20: min_gap = 0
since_last = (time.time() - os.path.getmtime(LASTF)) / 60 if os.path.exists(LASTF) else 1e9
if since_last < min_gap and not a.force:
    log(f"pacing: {max(0, BUDGET - n_today)} of {BUDGET} budgeted submissions left for {mins_left:.0f} min -> next publish allowed in {min_gap - since_last:.1f} min"); sys.exit(0)
# 3. build notebook (with an automatic changelog note: gains by source since the last publication)
NBDIR = os.path.join(ROOT, a.nbdir)
kmeta = json.load(open(os.path.join(NBDIR, "kernel-metadata.json")))
slug = kmeta["id"].split("/")[1]
CLF = os.path.join(ROOT, a.changelog)
changelog = json.load(open(CLF)) if os.path.exists(CLF) else []
prev_v = int(open(LASTF).read().split()[-1].lstrip("v")) if os.path.exists(LASTF) else 0
since = changelog[-1].get("t", os.path.getmtime(LASTF)) if changelog else 0
groups = collections.Counter(); prev_tot = None
for line in open(os.path.join(ROOT, "work/history.csv")):
    t, tag, v = line.strip().split(","); t, v = float(t), int(v)
    if prev_tot is not None and t > since and v < prev_tot:
        g = ("Wang Jakie strategies" if tag.startswith("wj") else "public re-merge" if tag.startswith("merge_public")
             else "Hansen0223 checkpoint" if tag.startswith("merge_hansen") else "our LNS + CP-SAT recombination")
        groups[g] += prev_tot - v
    prev_tot = v
note = f"Re-merged the public frontier ({p_tot:,}); added {len(chosen)} of our shorter programs (−{got})"
version = prev_v + 1
for f in glob.glob(os.path.join(NBDIR, "*.ipynb")): os.remove(f)
BUILD = [PY, os.path.join(ROOT, a.builder), "--lb", str(tot), "--user", USER, "--public",
         os.path.join(ROOT, "work/public_cycle.csv"), "--ours", os.path.join(ROOT, "work/sub_latest.csv"), "--slug", slug, "--version", str(version), "--note", note]
r = subprocess.run(BUILD, capture_output=True, text=True)
log(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-500:])
if r.returncode: sys.exit(1)
# 4. local test
if not a.skip_local_test:
    TR = os.path.join(ROOT, "deliver/test_run"); shutil.rmtree(TR, ignore_errors=True)
    os.makedirs(f"{TR}/input/santa-2026"); shutil.copy(os.path.join(ROOT, "data/puzzles.csv"), f"{TR}/input/santa-2026/")
    shutil.copytree(os.path.join(ROOT, "deliver/dataset"), f"{TR}/input/santa-2026-dag-lns-programs")
    for h in holders:
        d = os.path.join(OD, h["ref"].split("/")[1]); f = glob.glob(f"{d}/*.csv")
        if f: os.makedirs(f"{TR}/input/{h['ref'].split('/')[1]}", exist_ok=True); shutil.copy(f[0], f"{TR}/input/{h['ref'].split('/')[1]}/submission.csv")
    nbf = glob.glob(os.path.join(NBDIR, "*.ipynb"))[0]; shutil.copy(nbf, f"{TR}/nb.ipynb")
    r = subprocess.run([os.path.join(ROOT, ".venv/bin/jupyter"), "nbconvert", "--to", "notebook", "--execute", "nb.ipynb",
                        "--output", "nb_out.ipynb", "--ExecutePreprocessor.timeout=1200"], cwd=TR, capture_output=True, text=True,
                       env={**os.environ, "TRIWALK_DEMO_SECS": "4"})
    if r.returncode: log("LOCAL TEST FAILED", r.stderr[-1500:]); sys.exit(1)
    lt = sum(replay(P[k], v) for k, v in read_submission(f"{TR}/submission.csv").items())
    if lt != tot: log("local test score mismatch", lt, tot); sys.exit(1)
    log("local test ok", lt)
# 5. dataset version + push
# stale resumable-upload state (kept by the CLI after a failed upload) makes every later upload fail with "Invalid token"
import tempfile
uploaded = False
FAILF = os.path.join(ROOT, "work/upload_failing")
failing = os.path.exists(FAILF) and time.time() - os.path.getmtime(FAILF) < 7200   # re-test fully every 2 h
for attempt in range(1 if failing else 3):
    for f in glob.glob(os.path.join(tempfile.gettempdir(), f".kaggle-uploads-{os.geteuid()}", os.path.join(ROOT, "deliver/dataset").replace(os.path.sep, "_") + "_*.json")): os.remove(f)
    dv = kg("datasets", "version", "-p", os.path.join(ROOT, "deliver/dataset"), "-m", f"LB {tot:,}: re-merged public frontier {p_tot:,}", "-t")
    log(dv.strip().splitlines()[-1])
    if "Client Error" not in dv and "Server Error" not in dv: uploaded = True; break
    time.sleep(20)
LASTOK = os.path.join(ROOT, "work/ds_last_ok")
if uploaded and os.path.exists(FAILF): os.remove(FAILF)
if not uploaded and not failing: open(FAILF, "w").write(time.strftime("%H:%M"))
if uploaded:
    for _ in range(60):
        if "ready" in kg("datasets", "status", f"{USER}/santa-2026-dag-lns-programs").lower(): break
        time.sleep(15)
    os.makedirs(LASTOK, exist_ok=True)
    for f in ("our_programs.csv", "public_frontier.csv"): shutil.copy(os.path.join(ROOT, "deliver/dataset", f), os.path.join(LASTOK, f))
    time.sleep(45)
else:
    if "make_notebook2" not in a.builder or not os.path.exists(os.path.join(LASTOK, "our_programs.csv")):
        log("DATASET UPLOAD FAILED -> not pushing a notebook with a stale dataset"); sys.exit(1)
    log("DATASET UPLOAD FAILED -> embedding a patch over the last uploaded dataset version")
    for f in glob.glob(os.path.join(NBDIR, "*.ipynb")): os.remove(f)
    r = subprocess.run(BUILD + ["--patch-base", LASTOK], capture_output=True, text=True)
    log(" | ".join(r.stdout.strip().splitlines()[-2:]) if r.stdout else r.stderr[-500:])
    if r.returncode: sys.exit(1)
pushed = kg("kernels", "push", "-p", NBDIR)
log(pushed.strip().splitlines()[-1])
mv = re.search(r"Kernel version (\d+)", pushed); mu = re.search(r"kaggle\.com/(?:code/)?" + USER + r"/([a-z0-9-]+)", pushed)
if not mv: log("push failed"); sys.exit(1)
ver = mv.group(1); new_slug = mu.group(1) if mu else slug
K = f"{USER}/{new_slug}"
kmeta["id"] = K; json.dump(kmeta, open(os.path.join(NBDIR, "kernel-metadata.json"), "w"), indent=1)
time.sleep(30)
for _ in range(60):
    st = kg("kernels", "status", K)
    if re.search(r"complete|error|cancel", st, re.I): break
    time.sleep(20)
log(st.strip().splitlines()[-1])
if not re.search(r"complete", st, re.I): log("KERNEL RUN FAILED or timed out"); sys.exit(1)
KO = os.path.join(ROOT, "work/kout_cycle"); shutil.rmtree(KO, ignore_errors=True); os.makedirs(KO)
kg("kernels", "output", K, "-p", KO, timeout=600)
kt = sum(replay(P[k], v) for k, v in read_submission(f"{KO}/submission.csv").items())
if kt != tot: log("Kaggle output mismatch", kt, tot); sys.exit(1)
# 6. submit the notebook's own output -> Public Score chip
res = kg("competitions", "submit", "santa-2026", "-k", K, "-v", ver, "-f", "submission.csv", "-m",
         f"Notebook v{ver} output, LB {tot:,}", "--wait", "600", timeout=900)
log(res.strip().splitlines()[-1])
if not re.search(r"Public score:\s*" + str(tot) + r"\b", res): log("SUBMIT FAILED (chip not updated):", res.strip()[-300:])
open(LASTF, "w").write(f"{tot} {K} v{ver}\n")
changelog.append(dict(v=int(ver), date=time.strftime("%Y-%m-%d %H:%M"), lb=tot, pub=p_tot, note=note, t=time.time()))
json.dump(changelog, open(CLF, "w"), indent=1)
log(f"published {K} v{ver} with LB {tot:,} (public frontier {p_tot:,})")

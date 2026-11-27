"""Ordered, resumable experiment driver. Run cpsat_lns.py --help for entry points."""
from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

import cpsat_lns as L

sys.path.insert(0, str(L.ROOT / "public/asterios/elf_tools"))
from verify import check as independent_check

DB = L.WORK / "experiments.sqlite3"
CONTROL = "scored_0379"
STRATEGIES = ("output", "hub", "region", "disagree", "global")
LOG = ("stage", "arm", "id", "iteration", "strategy", "base", "D", "R", "cands",
       "recipes", "src", "gen_s", "status", "obj", "time", "cpu_s", "accepted",
       "reason", "new_len", "used_sources", "native_source", "native_timeout", "selected_values", "selected_sources")
IMPS = ("stage", "arm", "id", "iteration", "old", "new", "strategy", "sources")


def connect():
    return closing(sqlite3.connect(DB, timeout=60))


def initialize(db):
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS attempts (
          stage INTEGER, arm TEXT, id TEXT, iteration INTEGER, data TEXT, program TEXT,
          PRIMARY KEY(stage,arm,id,iteration));
        CREATE TABLE IF NOT EXISTS manifest (name TEXT PRIMARY KEY, value TEXT);
    """)


def record(db, row, program=None):
    with db:
        db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?)",
                   (row["stage"], row["arm"], row["id"], row["iteration"], json.dumps(row),
                    None if program is None else json.dumps(program, separators=(",", ":"))))


def verify(p, prog):
    L.replay(p, prog)
    ok, message = independent_check(dict(start=p.starts, targets=p.targets, limit=p.limit,
                                         regs=p.registers), prog)
    if not ok:
        raise ValueError(f"{p.id}: {message}")


def cpu_time():
    used = os.times()
    return used.user + used.system + used.children_user + used.children_system


def worker_init(cores):
    if cores and hasattr(os, "sched_setaffinity"):
        from multiprocessing import current_process
        os.sched_setaffinity(0, {cores[(current_process()._identity[0]-1) % len(cores)]})


def run_puzzle(job):
    stage, arm, p, prog, budget, cache, args = job
    key = (stage, arm, p.id)
    with connect() as db:
        history = db.execute("SELECT data,program FROM attempts WHERE stage=? AND arm=? AND id=? "
                             "ORDER BY iteration", key).fetchall()
        spent = 0
        for data, saved in history:
            row = json.loads(data)
            spent += row.get("cpu_s") or 0
            if saved:
                prog = json.loads(saved)
                verify(p, prog)
        if stage == 0 and history and row.get("new_len") == len(prog):
            return
        if stage and (spent >= budget or (stage == 1 and len(prog) <= 108)):
            return
        rules = {}
        if stage != 1:
            with (cache / (p.id + ".pkl")).open("rb") as stream:
                rules = pickle.load(stream)
        iteration = len(history)
        while stage == 0 or spent < budget:
            rng = random.Random(f"{args.seed}/{p.id}/{iteration}")
            row = dict(stage=stage, arm=arm, id=p.id, iteration=iteration, base=len(prog), accepted=0)
            start = cpu_time()
            wall = time.monotonic()
            if stage == 0:
                new = L.full_rebuild(p, prog, rules, args, rng, args.seconds, row)
                if new is not None:
                    verify(p, new)
                row["new_len"] = None if new is None else len(new)
                if row["new_len"] != len(prog):
                    row["reason"] = row.get("reason") or row.get("status") or "length_mismatch"
            else:
                strategy = STRATEGIES[iteration % len(STRATEGIES)]
                seconds = min(args.nb_seconds, budget-spent)
                # The control exercises sources 4 and 5 without a bank or later generators.
                args.ratio = stage >= 2
                args.native_source = ("", "mitm", "finisher")[iteration % 3] if stage >= 2 else ""
                args.native_timeout = False
                args.native_seconds = max(.01, seconds / 2)
                new, _ = L.neighbourhood(p, prog, strategy, rules, args, rng, seconds, row)
                row.update(native_source=args.native_source, native_timeout=args.native_timeout)
                if new is not None:
                    verify(p, new)
                    prog = new
            row["time"] = time.monotonic() - wall
            row["cpu_s"] = cpu_time() - start
            spent += row["cpu_s"]
            record(db, row, new if stage and new is not None else None)
            iteration += 1
            if stage == 0 or (stage == 1 and len(prog) <= 108):
                return


def import_exp0(db, base):
    if db.execute("SELECT 1 FROM manifest WHERE name='imported'").fetchone():
        return
    with db:
        for filename in ("exp0-first.jsonl", "exp0.jsonl"):
            path = L.WORK / filename
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                k = row["id"]
                if row["base"] != len(base[k]):
                    raise ValueError(f"Historical baseline mismatch for {k}")
                i = db.execute("SELECT count(*) FROM attempts WHERE stage=0 AND id=?", (k,)).fetchone()[0]
                row.update(stage=0, arm="bank", iteration=i, accepted=0)
                db.execute("INSERT INTO attempts VALUES(0,'bank',?,?,?,NULL)", (k,i,json.dumps(row)))
        db.execute("INSERT INTO manifest VALUES('imported','1')")


def write_csv(path, fields, rows):
    temp = path.with_suffix(".pending.csv")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return temp


def export_logs(db):
    rows = (json.loads(data) for data, in db.execute("SELECT data FROM attempts ORDER BY stage,arm,id,iteration"))
    path = L.EXP / "cpsat-lns-log.csv"
    write_csv(path, LOG, rows).replace(path)
    path = L.EXP / "cpsat-lns-improvements.csv"
    rows = (dict(r, old=r["base"], new=r["new_len"], sources=r.get("used_sources", ""))
            for data, in db.execute("SELECT data FROM attempts WHERE program IS NOT NULL ORDER BY stage,arm,id,iteration")
            for r in [json.loads(data)])
    write_csv(path, IMPS, rows).replace(path)


def best_programs(db, base, before=7):
    programs = dict(base)
    origins = {}
    for stage,arm,k,data,program in db.execute(
            "SELECT stage,arm,id,data,program FROM attempts WHERE stage>=2 AND stage<? "
            "AND program IS NOT NULL ORDER BY stage,arm,id,iteration", (before,)):
        r = json.loads(data)
        if r["new_len"] < len(programs[k]):
            programs[k] = json.loads(program)
            origins[k] = f"CP-SAT exp{stage}/{arm} {r['strategy']} {r.get('used_sources','')}".strip()
    return programs, origins


def publish(db, puzzles, base):
    best, changed = best_programs(db, base)
    with L.BASE.with_suffix(".sources.csv").open() as stream:
        origins = {r["id"]: r["origin"] for r in csv.DictReader(stream)}
    origins.update(changed)
    path = L.ROOT / "submission-cpsat.csv"
    pending = write_csv(path, ["id","program"],
                        (dict(id=k, program=json.dumps(v,separators=(",",":"))) for k,v in best.items()))
    try:
        checked = L.read_progs(pending, puzzles)
        if len(checked) != 1000:
            raise ValueError("Expected 1000 scored puzzles")
        for k, prog in checked.items():
            verify(puzzles[k], prog)
        provenance = path.with_suffix(".sources.csv")
        sources = write_csv(provenance, ["id","instructions","origin"],
                            (dict(id=k,instructions=len(v),origin=origins[k]) for k,v in checked.items()))
        pending.replace(path)
        sources.replace(provenance)
        return sum(map(len, checked.values()))
    finally:
        pending.unlink(missing_ok=True)


def ranking(args, puzzles, base):
    if args.priority or (L.ROOT / "priority.csv").exists():
        path = args.priority or L.ROOT / "priority.csv"
        with path.open(encoding="utf-8-sig") as stream:
            ids = [r["id"] for r in csv.DictReader(stream)]
        if set(ids) != set(puzzles) or len(ids) != len(puzzles):
            raise ValueError("priority.csv must rank every puzzle ID exactly once")
        return ids
    return sorted(puzzles, key=lambda k: (len(base[k])-len(set(puzzles[k].targets)-set(puzzles[k].starts)), k),
                  reverse=True)


def cohorts(stage, ranked, args):
    rng = random.Random(args.seed)
    if stage == 0: return [("bank", ranked, 0)]
    if stage == 1: return [("control", [CONTROL], args.control_budget)]
    if stage in (2,3,5):
        count = {2:20,3:119,5:1000}[stage]
        return [("top", ranked[:count], args.budget)]
    if stage == 4:
        return [("top", ranked[:100], args.budget), ("random", rng.sample(sorted(ranked),100), args.budget)]
    sample = rng.sample(sorted(ranked),10)
    return [(str(minutes), sample, 60*minutes) for minutes in (10,30,60)]


def run_arm(pool, stage, arm, ids, budget, puzzles, initial, cache, args):
    if stage != 3:
        jobs = ((stage, arm, puzzles[k], initial[k], budget, cache, args) for k in ids)
        for _ in pool.map(run_puzzle, jobs):
            pass
        return
    # Account for previous attempts on resume. Bound even a batch with a custom budget.
    width = args.workers
    for offset in range(0, len(ids), width):
        batch = ids[offset:offset+width]
        while True:
            with connect() as db:
                stats = metrics(db, stage, arm)
                paid = dict(db.execute(
                    "SELECT id,sum(json_extract(data,'$.cpu_s')) FROM attempts "
                    "WHERE stage=? AND arm=? GROUP BY id", (stage,arm)))
            if stats["saved"] == 0 and stats["cpu_hours"] >= 20:
                return
            remaining = [k for k in batch if paid.get(k,0) < budget]
            if not remaining:
                break
            allowance = (20-stats["cpu_hours"])*3600/len(remaining)
            limits = {k: budget if stats["saved"] else min(budget,paid.get(k,0)+allowance)
                      for k in remaining}
            jobs = ((stage,arm,puzzles[k],initial[k],limits[k],cache,args) for k in remaining)
            for _ in pool.map(run_puzzle,jobs):
                pass


def metrics(db, stage, arm):
    last = {}
    missing_cpu, saved, cpu_seconds = 0, 0, 0.0
    sources = set()
    selected = Counter()
    for data, in db.execute("SELECT data FROM attempts WHERE stage=? AND arm=? ORDER BY id,iteration",
                            (stage,arm)):
        row = json.loads(data)
        last[row["id"]] = row
        missing_cpu += row.get("cpu_s") is None
        cpu_seconds += row.get("cpu_s") or 0
        selected.update(row.get("selected_sources", {}))
        if row.get("accepted"):
            saved += row["base"]-row["new_len"]
            sources.update(filter(None,row.get("used_sources","").split("|")))
    return dict(puzzles=len(last), saved=saved, cpu_hours=cpu_seconds/3600,
                historical_rows_without_cpu=missing_cpu, accepted_sources=sorted(sources), selected_recipe_counts=dict(selected),
                matched=sum(r.get("new_len")==r["base"] for r in last.values()))


def gate(stage, arms):
    first = next(iter(arms.values()))
    if stage == 0: return first["matched"] == 1000
    if stage == 1: return first["saved"] >= 1
    if stage == 2: return first["saved"] >= 1
    if stage == 3: return first["saved"] >= 10
    if stage == 4:
        return arms["top"]["saved"]/arms["top"]["cpu_hours"] > arms["random"]["saved"]/arms["random"]["cpu_hours"]
    if stage == 5: return first["saved"] > 0
    return arms["10"]["saved"] < arms["30"]["saved"] < arms["60"]["saved"]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["suite", "publish", *[f"exp{i}" for i in range(7)]])
    p.add_argument("--priority", type=Path, help="CSV in rank order, with an id column")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cores", help="CPU list, e.g. 12-15,20-23 (defaults to allowed affinity)")
    p.add_argument("--seconds", type=float, default=60, help="bank rebuild time limit")
    p.add_argument("--nb-seconds", type=float, default=30)
    p.add_argument("--budget", type=float, default=600, help="CPU seconds per puzzle, experiments 2-5")
    p.add_argument("--control-budget", type=float, default=900)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--chain-cap", type=int, default=32, help="native witnesses per target")
    p.set_defaults(max_values=5000,max_recipes=50000,max_residual=3000,hubs=200,residual=True,dmin=10,dmax=25)
    return p


def main():
    # Hold one coordinator lock for the lifetime of the suite on the training host.
    L.WORK.mkdir(parents=True, exist_ok=True)
    with (L.WORK / "coordinator.lock").open("a+b") as lock:
        if sys.platform.startswith("linux"):
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Another CP-SAT coordinator is using this checkpoint")
        run()


def run():
    args = parser().parse_args()
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else list(range(os.cpu_count() or 1))
    cores = allowed
    if args.cores:
        cores = []
        for part in args.cores.split(","):
            ends = list(map(int,part.split("-")))
            cores.extend(range(ends[0],ends[-1]+1))
    if len(cores)!=len(set(cores)) or not set(cores)<=set(allowed) or not 1<=args.workers<=min(8,len(set(cores))):
        raise SystemExit("Choose 1-8 workers and a distinct allowed CPU for each worker")
    if min(args.seconds,args.nb_seconds,args.budget,args.control_budget,args.chain_cap)<=0:
        raise SystemExit("Budgets and chain cap must be positive")
    L.WORK.mkdir(parents=True,exist_ok=True)
    puzzles = L.load_puzzles(str(L.ROOT/"puzzles.csv"))
    base = L.read_progs(L.BASE,puzzles)
    if len(base)!=1000 or sum(map(len,base.values()))!=127250:
        raise ValueError("Unexpected experiment baseline")
    ranked = ranking(args,puzzles,base)
    signature = hashlib.sha256(json.dumps([ranked,args.seed,args.budget,args.control_budget,args.nb_seconds,
                                            args.chain_cap,base],sort_keys=True).encode()).hexdigest()
    with connect() as db:
        initialize(db)
        with db:
            db.execute("INSERT OR IGNORE INTO manifest VALUES('configuration',?)",(signature,))
            if db.execute("SELECT value FROM manifest WHERE name='configuration'").fetchone()[0]!=signature:
                raise ValueError("Checkpoint configuration differs. Use the original search settings.")
        import_exp0(db,base)
        export_logs(db)
        summary_path = L.EXP/"cpsat-lns-summary.json"
        previous = db.execute("SELECT value FROM manifest WHERE name='summary'").fetchone()
        summary = json.loads(previous[0]) if previous else {"experiments":{}}
        summary["ranking"] = str(args.priority or "priority.csv") if args.priority or (L.ROOT/"priority.csv").exists() else "baseline length minus non-start target count"
        def save_summary():
            with db:
                db.execute("INSERT OR REPLACE INTO manifest VALUES('summary',?)",(json.dumps(summary),))
            temp = summary_path.with_suffix(".pending.json")
            temp.write_text(json.dumps(summary,indent=2)+"\n")
            temp.replace(summary_path)
        if args.mode=="publish":
            summary["verified_total"]=publish(db,puzzles,base)
            save_summary()
            return
        end = 6 if args.mode=="suite" else int(args.mode[-1])
        cache = None
        for stage in range(end+1):
            groups = cohorts(stage,ranked,args)
            run = args.mode=="suite" or stage==end
            if run:
                if cache is None: cache = L.prepare_banks(puzzles)
                if stage>=2:
                    args.helper = str(build_helper())
                if stage==1:
                    initial = L.read_progs(L.ROOT/"public/ishankarnick/submission.csv",puzzles)
                    if len(initial[CONTROL])!=109: raise ValueError("Expected 109-step positive control")
                elif stage==5:
                    initial,_ = best_programs(db,base,before=5)
                else:
                    initial = base
                with ProcessPoolExecutor(args.workers, initializer=worker_init, initargs=(cores,)) as pool:
                    for arm,ids,budget in groups:
                        run_arm(pool,stage,arm,ids,budget,puzzles,initial,cache,args)
            arms = {arm:metrics(db,stage,arm) for arm,_,_ in groups}
            passed = all(arms[arm]["puzzles"]==len(ids) for arm,ids,_ in groups) and gate(stage,arms)
            if not run and not summary["experiments"].get(str(stage),{}).get("passed"):
                passed = False if stage else passed
            summary["experiments"][str(stage)] = dict(passed=passed,arms=arms)
            export_logs(db)
            if stage>=2 and run:
                summary["verified_total"] = publish(db,puzzles,base)
            save_summary()
            print(f"Experiment {stage}: {'passed' if passed else 'gate closed'} {arms}",flush=True)
            if not passed: raise SystemExit(1)


def build_helper():
    binary = L.WORK/("cpsat-chains.exe" if os.name=="nt" else "cpsat-chains")
    sources = [L.ROOT/name for name in ("cpsat_chains.cpp","four_step_leaf.hpp","arithmetic_search.hpp")]
    if not binary.exists() or any(p.stat().st_mtime>binary.stat().st_mtime for p in sources):
        temp = binary.with_suffix(".building")
        subprocess.run([os.environ.get("CXX","g++"),"-std=c++17","-O3",str(sources[0]),"-o",str(temp)],check=True)
        temp.replace(binary)
    return binary

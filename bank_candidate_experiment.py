"""Rebuild top-excess puzzles with CP-SAT and bank-derived candidate recipes."""
import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "public" / "goodpjw2008"))

import recombine as RC
from elf import load_puzzles, replay
from optimize import read_bank, save

OUT = ROOT / "experiments"
BASE_FILE = OUT / "cp-sat-rebuild.csv"
OUTPUT = OUT / "bank-plus-candidates.csv"
RESULTS = OUT / "bank-plus-candidates-results.csv"
BANK_FILES = [
    "experiments/submission-notebook-merge.csv",
    "experiments/native-candidate.csv",
    "experiments/schedule-candidate.csv",
    "experiments/whole-hub-submission.csv",
    "experiments/submission-focal-108.csv",
    "public/basin-lineage-v10/submission.csv",
    "public/bounds-triage-v10/submission.csv",
    "public/frontier-nov27/submission.csv",
    "public/viktorbarbarich/best_programs.csv",
    "public/b05204047elias/submission.csv",
    "public/ishankarnick/submission.csv",
    "public/goodpjw2008/our_programs.csv",
    "public/goodpjw2008/public_frontier.csv",
    "public/alexchilton/rudolph_programs.csv",
    "public/asterios/best_programs.csv",
    "public/farshid/submission.csv",
    "public/asterios/public_day_ago.csv",
    "public/goodpjw2008/frontier_22nov.csv",
    "public/goodpjw2008/frontier_20nov.csv",
]


class COp(ctypes.Structure):
    _fields_ = [("op", ctypes.c_char), ("a", ctypes.c_int64), ("b", ctypes.c_int64)]


class CRule(ctypes.Structure):
    _fields_ = [("v", ctypes.c_int64), ("r", COp)]


class CPatch(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int), ("a", CRule * 4)]


class HubPaths:
    def __init__(self):
        path = Path(tempfile.gettempdir()) / "santa-hub-paths.dll"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; build it with build_hub_search.ps1 before running this experiment."
            )
        self.library = ctypes.CDLL(str(path))
        self.library.patches.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(CPatch),
            ctypes.c_int,
        ]
        self.library.patches.restype = ctypes.c_int

    def options(self, values, target, limit, cap):
        ordered = sorted(set(values) - {target})
        if not ordered:
            return []
        array = np.asarray(ordered, dtype=np.int64)
        buffer = (CPatch * cap)()
        count = self.library.patches(
            ctypes.c_void_p(array.ctypes.data),
            len(array),
            target,
            limit,
            buffer,
            cap,
        )
        result = []
        for i in range(count):
            patch = buffer[i]
            result.append(
                tuple(
                    (
                        int(rule.v),
                        (
                            rule.r.op.decode("ascii"),
                            int(rule.r.a),
                            int(rule.r.b),
                        ),
                    )
                    for rule in patch.a[: patch.n]
                )
            )
        return result


def chain_executable():
    exe = Path(tempfile.gettempdir()) / "santa-joint-chain-limited.exe"
    source = ROOT / "joint_chain_limited.cpp"
    if not exe.exists() or exe.stat().st_mtime < source.stat().st_mtime:
        compiler = os.environ.get("CXX", "g++")
        subprocess.run(
            [compiler, "-std=c++17", "-O3", "-static", str(source), "-o", str(exe)],
            cwd=ROOT,
            check=True,
        )
    return exe


def enumerate_one_step(values, limit):
    ordered = sorted(values)
    recipes = {}

    def add(value, op, a, b):
        if 1 <= value <= limit and value not in values:
            recipes.setdefault(value, (op, a, b))

    for i, a in enumerate(ordered):
        for b in ordered[: i + 1]:
            add(a + b, "+", a, b)
            add(a - b, "-", a, b)
            product = a * b
            if product <= limit:
                add(product, "*", a, b)
            if a % b == 0:
                add(a // b, "/", a, b)
    return recipes


def read_candidate_programs(puzzles, limits):
    candidates = defaultdict(list)

    def add(key, program, label):
        if key not in puzzles or not isinstance(program, list):
            return
        if len(program) > limits[key] + 3:
            return
        replay(puzzles[key], program)
        candidates[key].append((program, label))

    for path in OUT.glob("*-wins.jsonl"):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if isinstance(row, list) and len(row) == 2:
                    add(row[0], row[1], str(path))
                elif isinstance(row, dict) and {"id", "program"} <= row.keys():
                    add(row["id"], row["program"], str(path))
    for path in OUT.glob("*winner*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        match = __import__("re").search(r"scored_\d{4}", path.name)
        if match and isinstance(row, list):
            add(match.group(0), row, str(path))
        elif match and isinstance(row, dict) and "program" in row:
            add(match.group(0), row["program"], str(path))
    return candidates


def read_partial_recipes(puzzles):
    candidates = defaultdict(set)
    focal = OUT / "hub-focal-verified.json"
    if not focal.exists() or "scored_0379" not in puzzles:
        return candidates
    data = json.loads(focal.read_text(encoding="utf-8"))
    puzzle = puzzles["scored_0379"]
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a // b,
    }
    for witness in data.get("six", []):
        for value, rule in witness.get("patch", {}).items():
            op, a, b = rule
            value, a, b = int(value), int(a), int(b)
            if op not in operations or not 1 <= value <= puzzle.limit:
                continue
            if op == "/" and (b == 0 or a % b):
                continue
            if operations[op](a, b) == value:
                candidates["scored_0379"].add((op, a, b, value))
    return candidates


def add_program_recipes(recipes, program, puzzle):
    recipes.update(RC.program_recipes(puzzle, program))


def target_residuals(puzzle, bank_values):
    by_value = defaultdict(set)
    by_target = defaultdict(list)
    for target in puzzle.targets:
        for operand in bank_values:
            for value, make in RC.inverses(target, operand, puzzle.limit):
                op, a, b, result = make(value)
                if result != target:
                    continue
                recipe = (op, a, b, target)
                by_value[value].add(recipe)
                by_target[target].append((value, recipe))
    return by_value, by_target


def add_ratio_recipes(puzzle, bank_values, one_step, recipes, denominator_limit):
    available = bank_values | set(one_step)
    found = 0
    for target in puzzle.targets:
        for source in puzzle.targets:
            if target == source:
                continue
            for denominator in range(1, denominator_limit + 1):
                if denominator not in available:
                    continue
                scaled_target = denominator * target
                quotient, _ = divmod(scaled_target, source)
                for numerator in {quotient, quotient + 1}:
                    if numerator < 1 or numerator not in available:
                        continue
                    product = numerator * source
                    if product > puzzle.limit:
                        continue
                    difference = scaled_target - product
                    residual = abs(difference)
                    if residual > puzzle.limit or (residual and residual not in available):
                        continue
                    scaled = product + difference
                    if not 1 <= scaled <= puzzle.limit or scaled % denominator:
                        continue
                    if scaled // denominator != target:
                        continue
                    for value in (denominator, numerator, residual):
                        if value in one_step:
                            op, a, b = one_step[value]
                            recipes.add((op, a, b, value))
                    recipes.add(("*", numerator, source, product))
                    if residual:
                        op = "+" if difference > 0 else "-"
                        a, b = (product, residual) if difference > 0 else (product, residual)
                        recipes.add((op, a, b, scaled))
                    if denominator > 1:
                        recipes.add(("/", scaled, denominator, target))
                    found += 1
    return found


def add_hub_residual_recipes(
    puzzle,
    hubs,
    coefficients,
    offsets,
    bank_values,
    final_by_residual,
    one_step,
    recipes,
):
    supported_outputs = defaultdict(set)
    for hub in hubs:
        for coefficient in coefficients:
            product = hub * coefficient
            if product > puzzle.limit:
                continue
            product_recipe = ("*", hub, coefficient, product)
            for offset in offsets:
                variants = []
                if product + offset <= puzzle.limit:
                    variants.append((product + offset, "+", product, offset))
                if product > offset:
                    variants.append((product - offset, "-", product, offset))
                if offset > product:
                    variants.append((offset - product, "-", offset, product))
                for residual, op, a, b in variants:
                    finals = final_by_residual.get(residual)
                    if not finals:
                        continue
                    recipes.add(product_recipe)
                    if residual not in bank_values:
                        recipes.add((op, a, b, residual))
                    for final in finals:
                        recipes.add(final)
                        supported_outputs[residual].add(final[3])
    return supported_outputs


def read_finishers(exe, free_values, puzzle, target, cap):
    values = sorted(set(free_values) - {target})
    if not values:
        return []
    request = " ".join(
        map(str, [len(values), puzzle.limit, target, cap, *values])
    )
    result = subprocess.run(
        [str(exe)],
        input=request + "\n",
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    routes = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 16:
            raise ValueError(f"Unexpected finisher output: {line[:160]}")
        outputs = list(map(int, fields[:4]))
        route = {}
        for i, output in enumerate(outputs):
            op = fields[4 + 3 * i]
            a, b = map(int, fields[5 + 3 * i : 7 + 3 * i])
            route[output] = (op, a, b)
        if outputs[-1] != target:
            continue
        routes.append(route)
    return routes


def generate_candidates(
    puzzle, key, base, bank_list, extra_programs, extra_recipes, hub_paths, chain_exe, args
):
    started = time.perf_counter()
    bank_recipes = set()
    values = set(puzzle.starts)
    frequency = Counter()
    for _, bank in bank_list:
        program = bank[key]
        if len(program) > len(base) + 3:
            continue
        recipes = RC.program_recipes(puzzle, program)
        bank_recipes.update(recipes)
    for program, _ in extra_programs.get(key, ()):
        bank_recipes.update(RC.program_recipes(puzzle, program))
    bank_recipes.update(extra_recipes.get(key, ()))
    for op, a, b, value in bank_recipes:
        values.update((a, b, value))
        frequency[value] += 1
        frequency[a] += 1
        frequency[b] += 1
    bank_values = values
    recipes = set(bank_recipes)
    one_step = enumerate_one_step(bank_values, puzzle.limit)

    path_support = defaultdict(set)
    path_count = 0
    for target in puzzle.targets:
        for route in hub_paths.options(bank_values, target, puzzle.limit, args.path_cap):
            path_count += 1
            for value, (op, a, b) in route:
                recipes.add((op, a, b, value))
                if value != target:
                    path_support[value].add(target)
    hubs = sorted(
        path_support,
        key=lambda value: (
            len(path_support[value]),
            frequency[value],
            value in bank_values,
            -value,
        ),
        reverse=True,
    )[: args.hubs]
    all_hubs = set(hubs)

    final_by_residual, residuals_by_target = target_residuals(puzzle, bank_values)
    for target, options in residuals_by_target.items():
        candidates = [
            (value, final)
            for value, final in options
            if value in one_step
        ]
        candidates.sort(
            key=lambda item: (
                abs(target - item[0]).bit_length(),
                abs(target - item[0]),
                item[0],
                item[1],
            )
        )
        for value, final in candidates[: args.residual_cap]:
            op, a, b = one_step[value]
            recipes.add((op, a, b, value))
            recipes.add(final)

    coefficients = [
        value
        for value in range(2, 51)
        if value in bank_values or value in one_step
    ]
    for value in coefficients:
        if value in one_step:
            op, a, b = one_step[value]
            recipes.add((op, a, b, value))
    offsets = sorted(
        bank_values,
        key=lambda value: (-frequency[value], value.bit_length(), value),
    )[: args.offsets]
    first_outputs = add_hub_residual_recipes(
        puzzle,
        hubs,
        coefficients,
        offsets,
        bank_values,
        final_by_residual,
        one_step,
        recipes,
    )
    later_hubs = {
        value
        for value, targets in first_outputs.items()
        if len(targets) >= 2 and value not in all_hubs
    }
    if later_hubs:
        selected_later = sorted(
            later_hubs,
            key=lambda value: (
                len(first_outputs[value]),
                frequency[value],
                -value,
            ),
            reverse=True,
        )[: args.later_hubs]
        more = add_hub_residual_recipes(
            puzzle,
            selected_later,
            coefficients,
            offsets,
            bank_values,
            final_by_residual,
            one_step,
            recipes,
        )
        first_outputs.update(more)
        all_hubs.update(selected_later)

    ratio_count = add_ratio_recipes(
        puzzle, bank_values, one_step, recipes, args.denominator
    )

    chain_values = set(puzzle.starts) | set(all_hubs)
    chain_values.update(
        value
        for value, _ in sorted(
            frequency.items(), key=lambda item: (-item[1], item[0])
        )[: args.chain_bank_values]
    )
    finisher_routes = defaultdict(list)
    for target in puzzle.targets:
        for route in read_finishers(
            chain_exe, chain_values, puzzle, target, args.finisher_cap
        ):
            finisher_routes[target].append(route)
    finisher_support = defaultdict(set)
    for target, routes in finisher_routes.items():
        for route in routes:
            for value in route:
                if value != target:
                    finisher_support[value].add(target)
    finisher_hubs = {
        value
        for value, targets in finisher_support.items()
        if len(targets) >= 2
    }
    for target, routes in finisher_routes.items():
        ranked = sorted(
            routes,
            key=lambda route: (
                sum(len(finisher_support[value]) for value in route if value != target),
                sum(value in all_hubs for value in route),
                tuple(sorted(route)),
            ),
            reverse=True,
        )
        for route in ranked[: args.finisher_routes]:
            for value, (op, a, b) in route.items():
                recipes.add((op, a, b, value))
    if finisher_hubs:
        late = sorted(
            finisher_hubs,
            key=lambda value: (
                len(finisher_support[value]),
                frequency[value],
                -value,
            ),
            reverse=True,
        )[: args.later_hubs]
        add_hub_residual_recipes(
            puzzle,
            late,
            coefficients,
            offsets,
            bank_values,
            final_by_residual,
            one_step,
            recipes,
        )

    generated_values = {
        value
        for _, _, _, value in recipes
        if value not in bank_values
    }
    stats = {
        "bank_recipes": len(bank_recipes),
        "bank_values": len(bank_values),
        "one_step_values": len(one_step),
        "hub_paths": path_count,
        "shared_hubs": len(hubs),
        "residual_routes": sum(
            min(args.residual_cap, sum(value in one_step for value, _ in options))
            for options in residuals_by_target.values()
        ),
        "qh_candidates": len(first_outputs),
        "ratio_routes": ratio_count,
        "finisher_routes": sum(map(len, finisher_routes.values())),
        "finisher_hubs": len(finisher_hubs),
        "candidate_values": len(generated_values),
        "recipes": len(recipes),
        "generation_seconds": round(time.perf_counter() - started, 3),
    }
    return recipes, stats


def build_rank(puzzle, program):
    registers = list(puzzle.starts) + [None] * (puzzle.registers - len(puzzle.starts))
    rank = {value: -1 for value in puzzle.starts}
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a // b,
    }
    for index, (op, a, b, dest) in enumerate(program):
        value = operations[op](registers[a], registers[b])
        rank.setdefault(value, index)
        registers[dest] = value
    return rank


tls = threading.local()


def schedule(puzzle, chosen, level):
    rank = tls.rank
    users = defaultdict(list)
    for value, (_, a, b) in chosen.items():
        users[a].append(value)
        if b != a:
            users[b].append(value)
    for mode in range(5):
        remaining = Counter(x for _, a, b in chosen.values() for x in (a, b))
        indegree = {
            value: sum(x not in puzzle.starts for x in set(chosen[value][1:]))
            for value in chosen
        }
        done = set(puzzle.starts)
        ready = [value for value in chosen if indegree[value] == 0]
        order = []
        while ready:
            def priority(value):
                _, a, b = chosen[value]
                freed = (remaining[a] == 1) + (b != a and remaining[b] == 1)
                position = rank.get(value, 1_000_000_000 + level.get(value, 0))
                if mode == 0:
                    return position, level.get(value, 0), -freed, value
                if mode == 1:
                    return position, -freed, level.get(value, 0), value
                if mode == 2:
                    return -freed, position, level.get(value, 0), value
                if mode == 3:
                    return level.get(value, 0), position, -freed, value
                return position, -level.get(value, 0), -freed, value

            ready.sort(key=priority)
            value = ready.pop(0)
            order.append(value)
            done.add(value)
            _, a, b = chosen[value]
            remaining[a] -= 1
            remaining[b] -= 1
            for user in users[value]:
                indegree[user] -= 1
                if indegree[user] == 0 and user not in done and user not in ready:
                    ready.append(user)
        if len(order) != len(chosen):
            continue
        uses = Counter(x for value in order for x in chosen[value][1:])
        registers = {
            value: i
            for i, value in enumerate(puzzle.starts)
            if uses[value]
        }
        free = [i for i in range(puzzle.registers) if i not in registers.values()]
        program = []
        valid = True
        for value in order:
            op, a, b = chosen[value]
            if a not in registers or b not in registers:
                valid = False
                break
            ra, rb = registers[a], registers[b]
            for operand in (a, b):
                uses[operand] -= 1
                if uses[operand] == 0 and operand in registers:
                    free.append(registers.pop(operand))
            if not free:
                valid = False
                break
            dest = free.pop()
            program.append([op, ra, rb, dest])
            if uses[value]:
                registers[value] = dest
            else:
                free.append(dest)
        if valid:
            try:
                replay(puzzle, program)
                return program
            except ValueError:
                pass
    return None


def run_one(key, puzzles, best, banks, extra, extra_recipes, hub_paths, chain_exe, args):
    puzzle = puzzles[key]
    base = best[key]
    tls.rank = build_rank(puzzle, base)
    recipes, stats = generate_candidates(
        puzzle, key, base, banks, extra, extra_recipes, hub_paths, chain_exe, args
    )
    generation_seconds = stats["generation_seconds"]
    solve_budget = max(0.1, args.seconds - generation_seconds)
    start = time.perf_counter()
    result = RC.recombine(
        puzzle,
        recipes,
        base,
        time_limit=solve_budget,
        cross=False,
        workers=1,
        bridge=False,
    )
    solve_seconds = time.perf_counter() - start
    if result is None:
        return key, None, {**stats, "status": "NO_SOLUTION", "solve_seconds": round(solve_seconds, 3)}
    program, objective, status, variable_count, recipe_count = result
    if program is None:
        return key, None, {
            **stats,
            "status": status + "_UNSCHEDULED",
            "objective": objective,
            "variables": variable_count,
            "model_recipes": recipe_count,
            "solve_seconds": round(solve_seconds, 3),
        }
    length = replay(puzzle, program)
    return key, program, {
        **stats,
        "status": status,
        "objective": objective,
        "variables": variable_count,
        "model_recipes": recipe_count,
        "solve_seconds": round(solve_seconds, 3),
        "instructions": length,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=119)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--path-cap", type=int, default=128)
    parser.add_argument("--hubs", type=int, default=32)
    parser.add_argument("--later-hubs", type=int, default=16)
    parser.add_argument("--offsets", type=int, default=96)
    parser.add_argument("--residual-cap", type=int, default=48)
    parser.add_argument("--denominator", type=int, default=50)
    parser.add_argument("--chain-bank-values", type=int, default=48)
    parser.add_argument("--finisher-cap", type=int, default=24)
    parser.add_argument("--finisher-routes", type=int, default=8)
    parser.add_argument("--count", type=int, default=None, help="Limit to the first N selected puzzles.")
    args = parser.parse_args()

    puzzles = load_puzzles(str(ROOT / "puzzles.csv"))
    best = read_bank(str(BASE_FILE), puzzles, strict=True)
    origins_path = BASE_FILE.with_suffix(".sources.csv")
    origins = {
        row["id"]: row["origin"]
        for row in csv.DictReader(origins_path.open(newline="", encoding="utf-8-sig"))
    }
    if OUTPUT.exists():
        current = read_bank(str(OUTPUT), puzzles, strict=True)
        current_origins = {
            row["id"]: row["origin"]
            for row in csv.DictReader(
                OUTPUT.with_suffix(".sources.csv").open(newline="", encoding="utf-8-sig")
            )
        }
        for key in puzzles:
            if len(current[key]) < len(best[key]):
                best[key] = current[key]
                origins[key] = current_origins[key]

    limits = {key: len(program) for key, program in best.items()}
    extra = read_candidate_programs(puzzles, limits)
    extra_recipes = read_partial_recipes(puzzles)
    bank_files = list(BANK_FILES)
    if str(BASE_FILE.relative_to(ROOT)) not in bank_files:
        bank_files.append(str(BASE_FILE.relative_to(ROOT)))
    if OUTPUT.exists():
        bank_files.append(str(OUTPUT.relative_to(ROOT)))
    banks = [(filename, read_bank(str(ROOT / filename), puzzles)) for filename in bank_files]

    ranking = sorted(
        (
            len(best[key]) - len(set(puzzle.targets) - set(puzzle.starts)),
            key,
        )
        for key, puzzle in puzzles.items()
    )
    keys = [key for _, key in reversed(ranking[-args.top :])]
    if args.count is not None:
        keys = keys[: args.count]
    print(
        "Baseline",
        sum(map(len, best.values())),
        "selected",
        len(keys),
        "top excess",
        ranking[-1][0] if ranking else None,
        "through",
        ranking[-len(keys)][0] if keys else None,
        flush=True,
    )

    chain_exe = chain_executable()
    hub_paths = HubPaths()
    RC.schedule = schedule
    save(str(OUTPUT), puzzles, best, origins)
    rows = []
    start = time.perf_counter()
    gains = 0
    rebuilt = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                key,
                puzzles,
                best,
                banks,
                extra,
                extra_recipes,
                hub_paths,
                chain_exe,
                args,
            ): key
            for key in keys
        }
        for completed, future in enumerate(as_completed(futures), 1):
            key, program, stats = future.result()
            old_length = len(best[key])
            if program is not None and len(program) < old_length:
                best[key] = program
                origins[key] = "CP-SAT bank-plus-generated-candidate reconstruction"
                gains += old_length - len(program)
                rebuilt += 1
                replay(puzzles[key], program)
                print("GAIN", key, old_length, "->", len(program), stats["status"], flush=True)
            row = {
                "id": key,
                "baseline": old_length,
                "final": len(best[key]),
                "saved": old_length - len(best[key]),
                **stats,
            }
            rows.append(row)
            if completed % 5 == 0 or program is not None and len(program) < old_length:
                score = save(str(OUTPUT), puzzles, best, origins)
                with RESULTS.open("w", newline="", encoding="utf-8") as stream:
                    columns = list(rows[0])
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    "PROGRESS",
                    completed,
                    "/",
                    len(keys),
                    "saved",
                    gains,
                    "score",
                    score,
                    "wall",
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    score = save(str(OUTPUT), puzzles, best, origins)
    if rows:
        with RESULTS.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        "COMPLETE",
        "score",
        score,
        "saved",
        gains,
        "puzzles improved",
        rebuilt,
        "wall",
        round(time.perf_counter() - start, 1),
        flush=True,
    )


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    main()

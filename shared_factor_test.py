"""Try missing pairwise target GCDs using the surviving program as a backbone."""
import csv
from itertools import combinations
from math import gcd

from optimize import compile_graph, graph, load_puzzles, read_bank, replay


def dependencies(roots, nodes):
    seen = set()
    order = []

    def visit(v):
        if v in nodes and v not in seen:
            seen.add(v)
            for operand in nodes[v][1:]:
                visit(operand)
            order.append(v)

    for root in roots:
        visit(root)
    return order


def recipes(v, free):
    # Use a deterministic operand order for this bounded scheduling test.
    for x in sorted(free):
        candidates = [('+', x, v-x), ('-', x, x-v), ('-', v+x, x),
                      ('/', v*x, x)]
        if v % x == 0:
            candidates.append(('*', x, v//x))
        if x % v == 0:
            candidates.append(('/', x, x//v))
        for op, a, b in candidates:
            if a in free and b in free:
                yield op, a, b


def test(p, program):
    nodes = graph(p, program)
    starts = set(p.starts)
    existing = starts | nodes.keys()
    best = program
    missing = viable = scheduled = 0
    for a, b in combinations(p.targets, 2):
        h = gcd(a, b)
        if h == 1 or h in existing or a in starts or b in starts:
            continue
        missing += 1
        backbone = dependencies((t for t in p.targets if t not in (a, b)), nodes)
        free = starts.union(backbone)
        if a in free or b in free:
            continue
        needed = sorted({h, a//h, b//h} - free)
        if len(backbone) + len(needed) + 2 >= len(best):
            continue
        options = [next(recipes(v, free), None) for v in needed]
        if None in options:
            continue
        viable += 1
        patch = dict(zip(needed, options))
        patch.update({a: ('*', h, a//h), b: ('*', h, b//h)})
        merged = nodes | patch
        for order in (backbone + needed + [a, b], dependencies(p.targets, merged)):
            candidate = compile_graph(p, merged, order)
            if candidate is not None:
                replay(p, candidate)
                scheduled += 1
                best = candidate
                break
    return best, (missing, viable, scheduled)


if __name__ == '__main__':
    import json
    puzzles = load_puzzles('puzzles.csv')
    programs = read_bank('submission.csv', puzzles, strict=True)
    with open('experiments/shared-factor-test.csv', 'w', newline='') as stream:
        out = csv.writer(stream)
        out.writerow(['id', 'start_steps', 'missing_gcd_pairs', 'viable_pairs',
                      'scheduled_pairs', 'steps_saved', 'improved_program'])
        for key, prog in programs.items():
            best, counts = test(puzzles[key], prog)
            out.writerow([key, len(prog), *counts, len(prog)-len(best),
                          json.dumps(best) if best is not prog else ''])

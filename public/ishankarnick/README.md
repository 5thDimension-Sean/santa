# Santa 2026 frontier programs

One valid program for each of the 1,000 puzzles, scoring **127,441**. I'll keep this file updated: whenever my merge plus polishing beats the last published score, a new version goes up.

## Files

- `submission.csv`: `id,program`, ready to submit. Every program has been replayed through an exact validator (ops, registers, value range, exact division, every target produced).
- `sources.csv`: per puzzle, its instruction count and where the current program came from.
- `CREDITS.md`: the public work this builds on.

## How it's made

1. Every few hours I download the public notebook outputs and datasets listed in `CREDITS.md`, replay every program, and keep the shortest valid one per puzzle.
2. I run the public graph-search polisher (Wang Jakie's notebook, via royalacecat's readable source) on the merged set, on spare CPU at about 20,000 rounds per puzzle. A shorter program is only kept after it replays cleanly.
3. A few programs came from my own beam-search solver.

On 2026-11-21, 95 puzzles had a program shorter than any public one; the other 905 tie the best public program.

## Use

Attach the dataset and copy `submission.csv`, or merge it into your own pool. Please credit the original authors in `CREDITS.md` as well as this dataset.

License: Apache 2.0. The CC0 programs from alexchilton keep their CC0 status.

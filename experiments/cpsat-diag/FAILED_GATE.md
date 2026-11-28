# CP-SAT LNS experiment 1 gate: failed (2026-11-27)

Oracle puzzle: scored_0379. Known 108 (submission.csv) vs public 109 (ishankarnick). Gate 1 needs the search to save at least 1 step without being told the answer.

## Findings
- The 108 needs two new intermediate values at once (787994 and the 11827 hub chain). Neither alone gets the search below 109.
- Forcing 11827 alone recovered 8 of 18 needed recipes; CP-SAT returned 109 (OPTIMAL).
- Forcing the top-300 hubs by votes (diag_oracle.py auto=300) took 364 s to generate and recovered 11 of 18. 787994 was outside the top 300 when ties were broken by value.
- Single-hub voting ranks 787994 about 1,150th of 385,000 by plain votes. Weighted votes rank 11827 about 40-60th.
- Conditional re-ranking (diag_cond.py, output in cond_rank.out): after adding 11827 to the kept set and re-voting over the 7 required values, 787994 ranks 201st (172nd before). Votes cap at 6 with heavy ties, and the top hubs are generic small values. Forcing 11827 gives 787994 no boost.
- Reaching 787994 by votes needs about 1,700 hubs. At about 1.2 s of generation per hub that is not affordable per puzzle, and forcing hub pairs is quadratic.

## Conclusion
Hub voting, single or stepwise, cannot find this kind of joint hub. The beam-over-hubs idea was dropped without implementation because its entry condition (787994 in the top ~50 after re-ranking) failed. No further server time on this method.

## Not pursued
- Pairs from the top ~60 hubs: about 15 CPU-hours for one puzzle, and would only rediscover a known 108.
- Rebuilding around a solver with dynamic candidate generation: weeks of work, uncertain payoff.

## Remaining routes
Daily public merge; team merge or newly published methods.

## Code state
cpsat_lns.py has deep_auto (auto=N) and hubb=N options; test_cpsat_runner.py passes 6/6. Commits a0f127c, bc5efae, ce21e9d on branch cpsat-lns (not yet pushed).

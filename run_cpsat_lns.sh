#!/usr/bin/env bash
# Launch from the checkout on the Linux training machine.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${1:-} != --inside ]]; then
    exec tmux new-session -d -s cpsat-lns bash "$PWD/run_cpsat_lns.sh" --inside "$@"
fi
shift
mkdir -p experiments/cpsat-lns-work
cores=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:8])))')
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if (( $# == 0 )); then set -- suite; fi
systemd-run --user --scope -p MemoryMax=64G -p "AllowedCPUs=$cores" \
    nice -n 19 python3 -u cpsat_lns.py "$@" 2>&1 | tee -a experiments/cpsat-lns-work/suite.out

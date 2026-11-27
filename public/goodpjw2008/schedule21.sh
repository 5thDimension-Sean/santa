#!/usr/bin/env bash
# schedule21: our own value-set annealer (src/triwalk.cpp) takes most of the CPU. In a paired test (95/57 alternative lineages, 20-30 s each,
# same machine) triwalk saved 19-30, Ross Wade's slack annealing 4 and our older elfopt 0. Fresh changed-puzzle pass first, then lineage
# polishing, then our best programs / archive variants; one Ross Wade pass in four. Stop: touch work/STOP_WJ
cd "$(dirname "$0")/.."
G='outs*/*/*.csv'; A=work/archive.jsonl; L=work/sched21.log; H=work/hot_ids.txt
stop() { [ -e work/STOP_WJ ] && exit 0; }
wj() { stop; .venv/bin/python src/wj_driver.py --strategy "$1" --rounds "$2" --only "$3" --jobs 19 --batch "$4" --tag wj --seeds-glob "$G" --archive "$A" --max-seeds 8 $5 >> $L 2>&1; }
ak() { stop; .venv/bin/python src/atakan_driver.py --bin "$1" --args "$2" --only "$3" --jobs 19 --batch "$4" --archive "$A" $5 >> $L 2>&1; }
rw() { stop; .venv/bin/python src/rw_driver.py --only "$1" --secs "$2" --jobs 19 --slack "$3" --kick "$4" --tag "$5" $6 >> $L 2>&1; }
tri() { stop; .venv/bin/python src/tri_driver.py "$@" --jobs 19 >> $L 2>&1; }
FR=("8 4200 9 10 9 {salt} 1500" "14 8000 12 13 9 {salt} 3000" "8 4200 9 10 13 {salt} 1500")
.venv/bin/python src/changed_ids.py > /dev/null
chg() {
  stop; n=$(.venv/bin/python src/changed_ids.py); echo "[$(date +%H:%M:%S)] changed puzzles: $n" >> $L
  if [ "$n" -gt 0 ]; then
    R=$(( 18 / n )); [ $R -lt 1 ] && R=1; [ $R -gt 8 ] && R=8
    cp work/changed_ids.txt work/chg_now.txt
    tri --mode best --only work/chg_now.txt --secs 40 --repeat $R --tag tri_fresh
    for f2 in "${FR[@]}"; do ak frontier "$f2" work/chg_now.txt 1 "--repeat $R"; done
    wj anneal_diverse 4000 work/chg_now.txt 1 "--repeat $R"
    wj incremental_diverse 4000 work/chg_now.txt 1 "--repeat $R"
    rw work/chg_now.txt 60 2 6 rw_fresh "--repeat $R"
  fi
}
SL=($(ls work/rw_slices/s_*.txt)); NS=${#SL[@]}; SK=(2 1 3 2); KK=(12 6 12 20)
for it in $(seq ${START:-0} 100000); do
  [ $((it % 12)) -eq 0 ] && { stop; .venv/bin/python src/lineages.py 0.08 0.7 >> $L 2>&1; }
  chg; tri --mode lineage --secs 30 --budget 420 --max-excess 5 --tag tri
  if [ $((it % 2)) -eq 1 ]; then chg; tri --mode best --secs 60 --sample 114 --variant-prob 0.6 --tag tri; fi
  if [ $((it % 4)) -eq 3 ]; then chg; sl=${SL[$(((it * 7 + 3) % NS))]}; rw $sl 150 ${SK[$((it / 4 % 4))]} ${KK[$((it / 4 % 4))]} rw; fi
done

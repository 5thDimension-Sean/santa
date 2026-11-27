#!/usr/bin/env bash
# schedule20 = schedule19 with (a) Ross Wade's slack-annealing engine also run on freshly changed puzzles (two slack/kick settings) and
# (b) the 20k hot-list pass moved from every 6th to every 30th iteration: 18 hot-list runs (schedule16-18) saved 1 in 5.8 h,
# Ross Wade's engine 4 in 3.6 h. Stop: touch work/STOP_WJ
cd "$(dirname "$0")/.."
G='outs*/*/*.csv'; A=work/archive.jsonl; L=work/sched20.log; H=work/hot_ids.txt
stop() { [ -e work/STOP_WJ ] && exit 0; }
wj() { stop; .venv/bin/python src/wj_driver.py --strategy "$1" --rounds "$2" --only "$3" --jobs 19 --batch "$4" --tag wj --seeds-glob "$G" --archive "$A" --max-seeds 8 $5 >> $L 2>&1; }
ak() { stop; .venv/bin/python src/atakan_driver.py --bin "$1" --args "$2" --only "$3" --jobs 19 --batch "$4" --archive "$A" $5 >> $L 2>&1; }
rw() { stop; .venv/bin/python src/rw_driver.py --only "$1" --secs "$2" --jobs 19 --slack "$3" --kick "$4" --tag "$5" $6 >> $L 2>&1; }
FR=("8 4200 9 10 9 {salt} 1500" "14 8000 12 13 9 {salt} 3000" "8 4200 9 10 13 {salt} 1500")
.venv/bin/python src/changed_ids.py > /dev/null
chg() {
  stop; n=$(.venv/bin/python src/changed_ids.py); echo "[$(date +%H:%M:%S)] changed puzzles: $n" >> $L
  if [ "$n" -gt 0 ]; then
    { cat work/changed_ids.txt; echo; cat $H; } | awk 'NF && !s[$0]++' | head -40 > $H.tmp && mv $H.tmp $H
    R=$(( 18 / n )); [ $R -lt 1 ] && R=1; [ $R -gt 8 ] && R=8
    for f2 in "${FR[@]}"; do ak frontier "$f2" work/changed_ids.txt 1 "--repeat $R"; done
    wj anneal_diverse 4000 work/changed_ids.txt 1 "--repeat $R"
    wj incremental_diverse 4000 work/changed_ids.txt 1 "--repeat $R"
    rw work/changed_ids.txt 60 2 6 rw_fresh "--repeat $R"
    rw work/changed_ids.txt 60 3 12 rw_fresh "--repeat $R"
  fi
}
SL=($(ls work/rw_slices/s_*.txt)); NS=${#SL[@]}
SK=(2 1 3 2); KK=(12 6 12 20)
for it in $(seq ${START:-0} 100000); do
  sl=${SL[$(((it * 7 + 3) % NS))]}; j=$(((it + it / NS) % 4))   # every slice meets every slack/kick setting over four sweeps
  chg; rw $sl 150 ${SK[$j]} ${KK[$j]} rw
  if [ $((it % 30)) -eq 29 ]; then chg; wj anneal_diverse 20000 $H 1 "--repeat 2"; fi
done

#!/usr/bin/env bash
# schedule18 = schedule17 + Ross Wade's slack-annealing engine (CC0) in alternation with the WJ deep passes, so both get similar time
# and can be compared from the log. schedule17 = schedule16 with 20-puzzle deep slices (work/rs_*.txt) so the fresh changed-puzzle pass runs every ~4 minutes.
# schedule15: long (20,000-round) diverse-pool runs swept over all focus puzzles in slices of 60 (sizes near 120 first),
# alternating with the hot list and the fast changed-puzzle pass. Stop: touch work/STOP_WJ
cd "$(dirname "$0")/.."
G='outs*/*/*.csv'; A=work/archive.jsonl; L=work/sched18.log; H=work/hot_ids.txt
stop() { [ -e work/STOP_WJ ] && exit 0; }
wj() { stop; .venv/bin/python src/wj_driver.py --strategy "$1" --rounds "$2" --only "$3" --jobs 19 --batch "$4" --tag wj --seeds-glob "$G" --archive "$A" --max-seeds 8 $5 >> $L 2>&1; }
ak() { stop; .venv/bin/python src/atakan_driver.py --bin "$1" --args "$2" --only "$3" --jobs 19 --batch "$4" --archive "$A" $5 >> $L 2>&1; }
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
  fi
}
SL=($(ls work/rs_*.txt)); NS=${#SL[@]}   # recently changed puzzles first
for it in $(seq ${START:-0} 100000); do
  sl=${SL[$((it % NS))]}
  chg; wj anneal_diverse 20000 $sl 1 "--start-variant"
  chg; wj incremental_diverse 20000 $sl 1 "--start-variant"
  chg; stop; .venv/bin/python src/rw_driver.py --only ${SL[$(((it * 7 + 3) % NS))]} --secs 230 --jobs 19 --kick 12 --tag rw >> $L 2>&1
  if [ $((it % 3)) -eq 2 ]; then chg; wj anneal_diverse 20000 $H 1 "--repeat 2"; fi
done

#!/bin/bash
# README 11.9 (exploratory, post-freeze; rules in rules_11_9.md written before any result).
#   b2: X3 simulated AP sweeps on tuning draws d1/d2 (large detectors), allocate.py, evals.
# Each stage skips work whose outputs exist; the python scripts refuse to overwrite.
# Copy of the b2 branch of fx_queue_b.sh with a fixed download wait (the downloader prints "DL_D <count>" per draw).
# Usage: nohup ./fx_queue_b2.sh b2 >> logs/fx_queue_b2.out 2>&1 &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
B5="0.03 0.05 0.1 0.15 0.2"
gpu_wait() {  # wait for >= 8 GB free on the GPU (other users share it)
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$f" -ge 8000 ] && return
    echo "$(date +%T) gpu free ${f} MiB, waiting"; sleep 300
  done
}
evalpath() {  # model cfg prefix crit...
  local M=$1 CFG=$2 PFX=$3; shift 3
  local keys="" todo="" tmp=/tmp/fx_dd_b_$$.txt
  [ -f $CFG ] || { echo "missing $CFG"; return; }
  for c in "$@"; do for b in $B5; do keys="$keys,$c@$b"; done; done
  $P fx_dedup.py --alloc $CFG --keys ${keys#,} | tee -a logs/fx_dedup_${M}.txt > $tmp
  while read k st rest; do
    t=${PFX}${k/@/_B}
    if [ "$st" = eval ] && [ ! -f metrics/alloc_eval/${M}_int8ent/$t.json ]; then todo="$todo,$k"; fi
  done < $tmp
  rm -f $tmp
  [ -z "$todo" ] && { echo "nothing to eval $M $CFG"; return; }
  gpu_wait; echo "== $(date +%F_%T) eval $M $CFG ${todo#,}"
  $P eval_alloc.py --alloc $CFG --keys ${todo#,} --splits val --prefix $PFX 2>&1 | grep -v "W:onnxruntime"
}
if [ "$1" = b2 ]; then
  for K in d1 d2; do
    n=${K#d}
    while [ "$(grep -c '^DL_D' logs/dl_draws.out)" -lt $n ]; do
      echo "$(date +%T) waiting for tune_$K images ($(ls data/tune_$K/images 2>/dev/null | wc -l))"; sleep 600; done
  done
  for M in yolo26l yolo11l; do
    RX=$($P -c "import json;print(json.load(open('configs/simalloc_${M}_int8ent_ho.json'))['force_regex'])")
    for K in d1 d2; do
      S=metrics/sens_sim_${M}_int8ent_${K}.json C=configs/simalloc_${M}_int8ent_ho_${K}.json
      if [ ! -f $S ]; then gpu_wait; echo "== $(date +%F_%T) simdraw $M $K"
        $P sim_sens_draw.py --model $M --tune-dir data/tune_$K --tag $K 2>&1 | grep -v "W:onnxruntime"; fi
      [ -f $S ] && [ ! -f $C ] && $P allocate.py --sens $S --onnx exports/${M}.onnx --budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --out $C
      evalpath $M $C fx${K}_ mean
    done
  done
  echo FX_QUEUE_B2_DONE
fi

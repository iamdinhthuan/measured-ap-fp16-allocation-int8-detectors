#!/bin/bash
# README 11.8 (exploratory, post-freeze). F1: executor-corrected SQNR/recon at 256 images; F2: factorial at 2000.
# Each stage skips work whose outputs exist; eval_alloc/fx_factorial refuse to overwrite.
# Usage: ./fx_queue_a.sh 2>&1 | tee -a logs/fx_queue_a.out
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
MODELS="yolo26l yolo11l yolo26s yolo11s"
B5="0.03 0.05 0.1 0.15 0.2"
gpu_wait() {  # wait for >= 8 GB free on the GPU (other users share it)
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$f" -ge 8000 ] && return
    echo "$(date +%T) gpu free ${f} MiB, waiting"; sleep 300
  done
}
sens() {  # model mode n tag [extra]
  local M=$1 MODE=$2 N=$3 TAG=$4; shift 4
  [ -f configs/fx_${M}_int8ent_ho_${MODE}${TAG}.json ] && { echo "skip sens $M $MODE$TAG"; return; }
  gpu_wait; echo "== $(date +%F_%T) sens $M $MODE $TAG"
  $P fx_factorial.py --model $M --mode $MODE --n-img $N --tag $TAG "$@" 2>&1 | grep -v "W:onnxruntime"
}
evalcfg() {  # model mode tag prefix crit...
  local M=$1 MODE=$2 TAG=$3 PFX=$4; shift 4
  local CFG=configs/fx_${M}_int8ent_ho_${MODE}${TAG}.json keys="" todo=""
  for c in "$@"; do for b in $B5; do keys="$keys,$c@$b"; done; done
  $P fx_dedup.py --alloc $CFG --keys ${keys#,} | tee -a logs/fx_dedup_${M}.txt > /tmp/fx_dd_$$.txt
  while read k st rest; do
    t=${PFX}${k/@/_B}
    if [ "$st" = eval ] && [ ! -f metrics/alloc_eval/${M}_int8ent/$t.json ]; then todo="$todo,$k"; fi
  done < /tmp/fx_dd_$$.txt
  rm -f /tmp/fx_dd_$$.txt
  [ -z "$todo" ] && { echo "nothing to eval $M $MODE$TAG"; return; }
  gpu_wait; echo "== $(date +%F_%T) eval $M ${todo#,}"
  $P eval_alloc.py --alloc $CFG --keys ${todo#,} --splits val --prefix $PFX 2>&1 | grep -v "W:onnxruntime"
}
for M in $MODELS; do sens $M noise 256 _n256; done
for M in $MODELS; do evalcfg $M noise _n256 fx256_ sqnrq reconq; done
echo STAGE_F1_DONE
for M in $MODELS; do sens $M apq 2000 _n2000; done
for M in $MODELS; do sens $M noise 2000 _n2000 --no-recon; done
for M in $MODELS; do evalcfg $M apq _n2000 fx2k_ apq; evalcfg $M noise _n2000 fx2k_ sqnrq sqnrr; done
echo FX_QUEUE_A_DONE

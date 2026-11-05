#!/bin/bash
# Remaining COCO bootstraps with bootstrap_fast.py (exact vectorised re-implementation of bootstrap_groups.py,
# ~1.6 GB instead of ~15-20 GB; earlyoom killed the old runs on 26-10 at 12:20-12:34). Same sets, pairs,
# seed and n as the queued bootstrap_groups.py runs. Usage: ./queue_boot_fast.sh A|B (two chains in parallel).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
run() {  # model out pairs set1 set2 ...
  local M=$1 OUT=$2 PR=$3; shift 3; local D=metrics/alloc_eval/${M}_int8ent; local S="fp32=$D/fp32_val.pred.json.gz"
  for k in "$@"; do S="$S,$k=$D/${k}_val.pred.json.gz"; done
  echo "== $(date +%F_%T) $M $OUT"
  nice -n 10 $P bootstrap_fast.py --gt $GT --sets "$S" --ref fp32 --pairs "$PR" --n 1000 --out metrics/bootstrap/$OUT.json
  echo "== rc=$?"
}
B5="0.03 0.05 0.1 0.15 0.2"
pairs_vs() { local X=$1 Y=$2 o=""; for b in $B5; do o="$o,${X}_B$b:${Y}_B$b"; done; echo ${o#,}; }
sets_vs() { local o=""; for b in $B5; do o="$o ${1}_B$b ${2}_B$b"; done; echo $o; }
# Reconstruction-error knapsack vs engine mean (added 26-10 after an internal review: the text compared them without intervals).
if [ "$1" = E ]; then
  for M in yolo11s yolo26s; do run $M g_${M}_coco_recon $(pairs_vs recon mean) $(sets_vs mean recon); done
  echo CHAIN_E_DONE
elif [ "$1" = F ]; then
  for M in yolo26l yolo11l; do run $M g_${M}_coco_recon $(pairs_vs recon mean) $(sets_vs mean recon); done
  echo CHAIN_F_DONE
fi

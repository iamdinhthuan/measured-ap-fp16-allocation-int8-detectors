#!/bin/bash
# Post-freeze E2/E3 bootstraps (README 11.6): sim-AP mean allocation vs engine mean allocation, and best-of-33
# random vs engine mean. Usage: ./queue_boot_k2.sh A|B (two chains in parallel).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
run() {  # model out pairs set1 set2 ...
  local M=$1 OUT=$2 PR=$3; shift 3; local D=metrics/alloc_eval/${M}_int8ent; local S="fp32=$D/fp32_val.pred.json.gz"
  for k in "$@"; do S="$S,$k=$D/${k}_val.pred.json.gz"; done
  echo "== $(date +%F_%T) $M $OUT"
  nice -n 10 $P bootstrap_fast.py --gt $GT --sets "$S" --ref fp32 --pairs "$PR" --n 1000 --out metrics/bootstrap/$OUT.json 2>&1 | grep -v "^resample"
  echo "== rc=$?"
}
pairs() { local X=$1 Y=$2 o=""; shift 2; for b in "$@"; do o="$o,${X}_B$b:${Y}_B$b"; done; echo ${o#,}; }
sets() { local X=$1 Y=$2 o=""; shift 2; for b in "$@"; do o="$o ${X}_B$b ${Y}_B$b"; done; echo $o; }
B5="0.03 0.05 0.1 0.15 0.2"
if [ "$1" = A ]; then
  for M in yolo11s yolo26s; do run $M k2_${M}_sim $(pairs sim_mean mean $B5) $(sets sim_mean mean $B5); done
  echo CHAIN_K2A_DONE
elif [ "$1" = B ]; then
  run yolo11l k2_yolo11l_sim $(pairs sim_mean mean $B5) $(sets sim_mean mean $B5)
  run yolo26l k23_yolo26l "$(pairs sim_mean mean 0.1 0.15 0.2),$(pairs best33 mean 0.05 0.1 0.2)" $(sets sim_mean mean 0.1 0.15 0.2) best33_B0.05 mean_B0.05 best33_B0.1 best33_B0.2
  echo CHAIN_K2B_DONE
fi

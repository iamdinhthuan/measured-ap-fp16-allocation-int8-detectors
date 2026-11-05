#!/bin/bash
# Post-freeze E1 (README 11.6): greedy order vs exact mean allocation, at the budgets where the blocks differ.
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
run yolo26s k1_yolo26s_greedy greedy_B0.05:mean_B0.05,greedy_B0.15:mean_B0.15,greedy_B0.2:mean_B0.2 greedy_B0.05 mean_B0.05 greedy_B0.15 mean_B0.15 greedy_B0.2 mean_B0.2
run yolo26l k1_yolo26l_greedy greedy_B0.2:mean_B0.2 greedy_B0.2 mean_B0.2
echo BOOT_K1_DONE

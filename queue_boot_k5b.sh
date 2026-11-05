#!/bin/bash
# Post-freeze E5b (README 11.6, descriptive): output-SQNR knapsack vs sim-AP on the same 256 tuning images.
# Usage: ./queue_boot_k5b.sh F|G
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
run() {  # model out pairs set1 set2 ...
  local M=$1 OUT=$2 PR=$3; shift 3; local D=metrics/alloc_eval/${M}_int8ent; local S="fp32=$D/fp32_val.pred.json.gz"
  for k in "$@"; do S="$S,$k=$D/${k}_val.pred.json.gz"; done
  echo "== $(date +%F_%T) $M $OUT"
  if [ -f metrics/bootstrap/$OUT.json ]; then echo "exists, skip"; return; fi
  nice -n 10 $P bootstrap_fast.py --gt $GT --sets "$S" --ref fp32 --pairs "$PR" --n 1000 --out metrics/bootstrap/$OUT.json 2>&1 | grep -v "^resample"
  echo "== rc=$?"
}
pairs() { local X=$1 Y=$2 o=""; shift 2; for b in "$@"; do o="$o,${X}_B$b:${Y}_B$b"; done; echo ${o#,}; }
sets() { local X=$1 Y=$2 o=""; shift 2; for b in "$@"; do o="$o ${X}_B$b ${Y}_B$b"; done; echo $o; }
B5="0.03 0.05 0.1 0.15 0.2"
if [ "$1" = F ]; then
  run yolo26l k5b_yolo26l_sqnr_sim256 "sqnr_B0.03:mean_B0.03,$(pairs sqnr sim256_mean 0.05 0.1 0.15 0.2)" sqnr_B0.03 mean_B0.03 $(sets sqnr sim256_mean 0.05 0.1 0.15 0.2)
  run yolo11l k5b_yolo11l_sqnr_sim256 $(pairs sqnr sim256_mean $B5) $(sets sqnr sim256_mean $B5)
  echo CHAIN_K5BF_DONE
elif [ "$1" = G ]; then
  for M in yolo26s yolo11s; do run $M k5b_${M}_sqnr_sim256 $(pairs sqnr sim256_mean $B5) $(sets sqnr sim256_mean $B5); done
  echo CHAIN_K5BG_DONE
fi

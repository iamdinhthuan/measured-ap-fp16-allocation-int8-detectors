#!/bin/bash
# Post-freeze E5/E3 bootstraps (README 11.6): sim-AP on 256 tuning images (sim256) vs engine mean allocation,
# and best-of-33 random vs engine mean for the remaining models. Usage: ./queue_boot_k5.sh C|D|E
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
if [ "$1" = C ]; then
  run yolo26l k5_yolo26l_sim256 $(pairs sim256_mean mean 0.05 0.1 0.15 0.2) $(sets sim256_mean mean 0.05 0.1 0.15 0.2)
  run yolo11l k5_yolo11l_sim256 $(pairs sim256_mean mean $B5) $(sets sim256_mean mean $B5)
  echo CHAIN_K5C_DONE
elif [ "$1" = D ]; then
  for M in yolo26s yolo11s; do run $M k5_${M}_sim256 $(pairs sim256_mean mean $B5) $(sets sim256_mean mean $B5); done
  echo CHAIN_K5D_DONE
elif [ "$1" = E ]; then
  until grep -q QUEUE_K3B_DONE logs/queue_k3b_20261028_0348.out; do sleep 300; done
  for M in yolo11l yolo26s yolo11s; do run $M k3_${M}_best33 $(pairs best33 mean 0.05 0.1 0.2) $(sets best33 mean 0.05 0.1 0.2); done
  echo CHAIN_K5E_DONE
fi

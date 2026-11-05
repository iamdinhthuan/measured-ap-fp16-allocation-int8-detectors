#!/bin/bash
# Engine mean vs output-SQNR knapsack on the two large detectors (post-freeze baseline, outside the
# confirmatory rule). Starts after boot_sqnr_r2 so that at most two bootstraps run at once.
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
until grep -q BOOT_SQNR2_DONE logs/boot_sqnr_r2.out; do sleep 60; done
for M in yolo26l yolo11l; do
  D=metrics/alloc_eval/${M}_int8ent
  S="fp32=$D/fp32_val.pred.json.gz"; PR=""
  for B in 0.03 0.05 0.1 0.15 0.2; do
    S="$S,mean_B$B=$D/mean_B${B}_val.pred.json.gz,sqnr_B$B=$D/sqnr_B${B}_val.pred.json.gz"; PR="$PR,sqnr_B$B:mean_B$B"
  done
  echo "== $(date +%F_%T) $M"
  $P bootstrap_groups.py --gt data/coco_src/annotations/instances_val2017.json --sets $S --ref fp32 --pairs ${PR#,} --n 1000 --out metrics/bootstrap/g_${M}_coco_sqnr.json
  echo "== rc=$?"
done
echo BOOT_SQNR_LARGE_DONE

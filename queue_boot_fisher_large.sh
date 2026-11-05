#!/bin/bash
# Confirmatory models: engine mean vs Fisher knapsack (n2000), paired bootstrap on val2017. After queue_boot_26s.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
until grep -q BOOT26S_DONE logs/boot_26s.out; do sleep 60; done
for M in yolo26l yolo11l; do
  D=metrics/alloc_eval/${M}_int8ent
  S="fp32=$D/fp32_val.pred.json.gz"; PR=""
  for B in 0.03 0.05 0.1 0.15 0.2; do S="$S,mean_B$B=$D/mean_B${B}_val.pred.json.gz,fisher_B$B=$D/fisher_B${B}_val.pred.json.gz"; PR="$PR,fisher_B$B:mean_B$B"; done
  echo "== $(date +%F_%T) $M"
  nice -n 10 $P bootstrap_groups.py --gt data/coco_src/annotations/instances_val2017.json --sets $S --ref fp32 --pairs ${PR#,} --n 1000 --out metrics/bootstrap/g_${M}_coco_fisher.json
  echo "== rc=$?"
done
echo BOOT_FISHER_LARGE_DONE

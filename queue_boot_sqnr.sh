#!/bin/bash
# post-freeze: engine mean vs output-SQNR / reconstruction-error knapsacks, paired bootstrap on val2017 (size groups)
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
for M in yolo11s yolo26s; do
  D=metrics/alloc_eval/${M}_int8ent
  S="fp32=$D/fp32_val.pred.json.gz"; PR=""
  for B in 0.03 0.05 0.1 0.15 0.2; do S="$S,mean_B$B=$D/mean_B${B}_val.pred.json.gz,sqnr_B$B=$D/sqnr_B${B}_val.pred.json.gz"; PR="$PR,sqnr_B$B:mean_B$B"; done
  for B in 0.1 0.2; do S="$S,recon_B$B=$D/recon_B${B}_val.pred.json.gz"; PR="$PR,recon_B$B:mean_B$B"; done
  echo "== $(date +%F_%T) $M"
  $P bootstrap_groups.py --gt data/coco_src/annotations/instances_val2017.json --sets $S --ref fp32 --pairs ${PR#,} --n 1000 --out metrics/bootstrap/g_${M}_coco_sqnr.json
  echo "== rc=$?"
done
echo BOOT_SQNR_DONE

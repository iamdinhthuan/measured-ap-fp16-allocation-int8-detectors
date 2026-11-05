#!/bin/bash
cd /home/thuan/isci_method_20261025
for B in 0.05 0.1 0.15 0.2; do
  nice -n 10 /home/thuan/miniconda3/envs/qtsd/bin/python bootstrap_pair.py --gt data/coco_src/annotations/instances_val2017.json --a metrics/alloc_eval/yolo11s_int8ent/fisher_B${B}_val.pred.json.gz --b metrics/alloc_eval/yolo11s_int8ent/mean_B${B}_val.pred.json.gz --n 1000 --out metrics/bootstrap/yolo11s_int8ent_mean_vs_fisher_B${B}.json
done
echo BOOT_DONE

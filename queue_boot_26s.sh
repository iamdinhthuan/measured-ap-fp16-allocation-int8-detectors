#!/bin/bash
# yolo26s COCO: engine mean vs Fisher, and minimax vs cost-matched mean at the three budgets where they differ.
# Replaces the second half of boot_g2.sh (stopped before it started, to keep one bootstrap at a time).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
until grep -q BOOT_LARGE_DONE logs/boot_large.out; do sleep 60; done
D=metrics/alloc_eval/yolo26s_int8ent
S="fp32=$D/fp32_val.pred.json.gz"
for k in fisher_B0.03 mean_B0.03 fisher_B0.05 mean_B0.05 fisher_B0.1 mean_B0.1 fisher_B0.15 mean_B0.15 fisher_B0.2 mean_B0.2 minimax_B0.03 minimax_B0.05 mean_matched_B0.05 minimax_B0.1 stem_headout; do
  S="$S,$k=$D/${k}_val.pred.json.gz"; done
nice -n 10 $P bootstrap_groups.py --gt $GT --sets "$S" --ref fp32 \
  --pairs fisher_B0.03:mean_B0.03,fisher_B0.05:mean_B0.05,fisher_B0.1:mean_B0.1,fisher_B0.15:mean_B0.15,fisher_B0.2:mean_B0.2,mean_B0.03:minimax_B0.03,mean_matched_B0.05:minimax_B0.05,mean_B0.1:minimax_B0.1,stem_headout:mean_B0.03 \
  --n 1000 --out metrics/bootstrap/g_yolo26s_coco.json
echo BOOT26S_DONE

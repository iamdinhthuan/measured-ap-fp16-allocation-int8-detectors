#!/bin/bash
# Part 2: COCO val, size groups (ref = FP32 val). yolo11s objective comparison, then yolo26s
# engine-mean vs Fisher and minimax vs cost-matched mean (waits for boot_26.sh to finish first).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
D=metrics/alloc_eval/yolo11s_int8ent
S="fp32=$D/fp32_val.pred.json.gz"
for k in mean_B0.1 minimax_B0.1 mean_B0.2 minimax_B0.2 mean_matched_B0.2 stem_headout; do S="$S,$k=$D/${k}_val.pred.json.gz"; done
nice -n 15 $P bootstrap_groups.py --gt $GT --sets "$S" --ref fp32 \
  --pairs mean_B0.1:minimax_B0.1,mean_B0.2:minimax_B0.2,mean_matched_B0.2:minimax_B0.2,stem_headout:mean_B0.1 \
  --n 1000 --out metrics/bootstrap/g_yolo11s_coco_objective.json
while pgrep -f boot_26.sh >/dev/null; do sleep 60; done
D=metrics/alloc_eval/yolo26s_int8ent
S="fp32=$D/fp32_val.pred.json.gz"
for k in fisher_B0.03 mean_B0.03 fisher_B0.05 mean_B0.05 fisher_B0.1 mean_B0.1 fisher_B0.15 mean_B0.15 fisher_B0.2 mean_B0.2 minimax_B0.05 mean_matched_B0.05 stem_headout; do
  S="$S,$k=$D/${k}_val.pred.json.gz"; done
nice -n 15 $P bootstrap_groups.py --gt $GT --sets "$S" --ref fp32 \
  --pairs fisher_B0.03:mean_B0.03,fisher_B0.05:mean_B0.05,fisher_B0.1:mean_B0.1,fisher_B0.15:mean_B0.15,fisher_B0.2:mean_B0.2,mean_matched_B0.05:minimax_B0.05,stem_headout:mean_B0.03 \
  --n 1000 --out metrics/bootstrap/g_yolo26s_coco.json
echo BOOTG2_DONE

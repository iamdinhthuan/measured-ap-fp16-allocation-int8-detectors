#!/usr/bin/env bash
# Table 6 completeness: minimax vs cost-matched mean at the YOLO11-s budgets that lacked an interval (3, 5, 15%).
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
D=metrics/alloc_eval/yolo11s_int8ent
S="fp32=$D/fp32_val.pred.json.gz"
for k in mean_B0.03 minimax_B0.03 mean_B0.05 minimax_B0.05 mean_B0.15 minimax_B0.15; do S="$S,$k=$D/${k}_val.pred.json.gz"; done
nice -n 15 $P bootstrap_fast.py --gt $GT --sets "$S" --ref fp32 \
  --pairs mean_B0.03:minimax_B0.03,mean_B0.05:minimax_B0.05,mean_B0.15:minimax_B0.15 \
  --n 1000 --out metrics/bootstrap/g_yolo11s_coco_objective_b.json
echo BOOT_OBJ11S_B_DONE

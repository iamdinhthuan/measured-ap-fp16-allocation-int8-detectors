#!/bin/bash
# yolo26s: FP32 val predictions (reference for worst-group loss), then paired bootstrap minimax vs mean per budget
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
D=metrics/alloc_eval/yolo26s_int8ent
REF=$D/fp32_val.pred.json.gz
[ -f $REF ] || flock sens.lock $P infer_eval.py --engine exports/yolo26s_fp32.plan --gt $GT --images data/coco/images/val2017 --out $D/fp32_val.json --save-pred
for B in 0.05 0.15 0.03 0.1; do
  nice -n 10 $P bootstrap_pair.py --gt $GT --a $D/mean_B${B}_val.pred.json.gz --b $D/minimax_B${B}_val.pred.json.gz --ref $REF --n 1000 --out metrics/bootstrap/yolo26s_int8ent_ho_minimax_vs_mean_B${B}.json
done
echo BOOT_DONE

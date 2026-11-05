#!/bin/bash
# Paired bootstraps with bootstrap_groups.py (all pairs share resamples, seed 20261025, n=1000).
# Part 1: ExDark test, light groups, yolo11s (ref = FP32 on ExDark test).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
D=metrics/alloc_eval/yolo11s_int8ent
S="fp32=metrics/exdark/yolo11s_fp32_test.pred.json.gz,stem=$D/exdark_stem_headout_exdark_test.pred.json.gz"
for k in mean_B0.03 minimax_B0.03 mean_B0.1 minimax_B0.1 mean_B0.15 minimax_B0.15 mean_B0.2 minimax_B0.2; do
  S="$S,$k=$D/exdark_${k}_exdark_test.pred.json.gz"
done
nice -n 15 $P bootstrap_groups.py --gt data/exdark/exdark_test.json --sets "$S" --ref fp32 \
  --pairs mean_B0.03:minimax_B0.03,mean_B0.1:minimax_B0.1,mean_B0.15:minimax_B0.15,mean_B0.2:minimax_B0.2,stem:mean_B0.03,stem:minimax_B0.03 \
  --groups attr:light --group-names Ambient,Strong,Single,Weak,Twilight,Object,Low,Window \
  --n 1000 --out metrics/bootstrap/g_yolo11s_exdark_light.json
echo BOOTG1_DONE

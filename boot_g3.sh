#!/bin/bash
# ExDark B=5%: minimax@0.05 has the same blocks as minimax@0.03 (evaluated as exdark_minimax_B0.03);
# mean_matched@0.05 = mean@0.05. Same seed/n as boot_g1, so resamples are identical.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
D=metrics/alloc_eval/yolo11s_int8ent
nice -n 15 $P bootstrap_groups.py --gt data/exdark/exdark_test.json \
  --sets "fp32=metrics/exdark/yolo11s_fp32_test.pred.json.gz,mean_B0.05=$D/exdark_mean_B0.05_exdark_test.pred.json.gz,minimax_B0.05=$D/exdark_minimax_B0.03_exdark_test.pred.json.gz" \
  --ref fp32 --pairs mean_B0.05:minimax_B0.05 \
  --groups attr:light --group-names Ambient,Strong,Single,Weak,Twilight,Object,Low,Window \
  --n 1000 --out metrics/bootstrap/g_yolo11s_exdark_light_B005.json
echo BOOTG3_DONE

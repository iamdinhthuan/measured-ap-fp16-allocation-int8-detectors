#!/bin/bash
# queued after the yolo26s int8ent sweep (same sens.lock): finish yolo26s manual configs, then ExDark sensitivity
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
exec flock sens.lock bash -c "$P eval_alloc.py --alloc configs/manual_yolo26s_int8ent.json --keys stem_headout > logs/manual_yolo26s_int8ent_b.out 2>&1; $P block_sensitivity.py --model yolo11s --base int8ent --split exdark > logs/sens_yolo11s_int8ent_exdark.out 2>&1" < /dev/null

#!/bin/bash
# evaluate the ModelOpt auto_quantize baselines for yolo11s (native weight-bits grid, and Fisher-score MAC knapsack with head-out forced)
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
exec flock sens.lock bash -c "$P eval_alloc.py --alloc configs/autoquant_yolo11s_int8ent.json --keys autoquant@8.02,autoquant@8.04,autoquant@8.08,autoquant@8.15,autoquant@8.3 > logs/aq_eval_yolo11s.out 2>&1; $P eval_alloc.py --alloc configs/fisher_yolo11s_int8ent_ho.json --keys fisher@0.03,fisher@0.05,fisher@0.1,fisher@0.15,fisher@0.2 > logs/fisher_eval_yolo11s.out 2>&1" < /dev/null

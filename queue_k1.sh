#!/bin/bash
# Post-freeze E1 (README 11.6): greedy engine-sensitivity fallback (Wu 2020 / YOLOv6 style).
# Only budgets where greedy differs from the exact mean program are built; identical selections reuse the mean engines.
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
run $P eval_alloc.py --alloc configs/greedy_yolo26s_int8ent_ho.json --keys greedy@0.05,greedy@0.15,greedy@0.2
run $P eval_alloc.py --alloc configs/greedy_yolo26l_int8ent_ho.json --keys greedy@0.2
echo QUEUE_K1_DONE

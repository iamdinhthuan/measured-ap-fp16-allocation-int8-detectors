#!/usr/bin/env bash
# Short post-freeze additions (2026-10-25 23:30): whole-head heuristic through eval_alloc for both
# development models, and the YOLO11-s 3% allocations that were not evaluated. Development models only.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
{
run $P eval_alloc.py --alloc configs/wholehead_yolo11s_int8ent.json --keys head
run $P eval_alloc.py --alloc configs/wholehead_yolo26s_int8ent.json --keys head
run $P eval_alloc.py --alloc metrics/alloc_yolo11s_int8ent_ho_v2.json --keys mean@0.03,minimax@0.03
echo QUEUE_G_DONE
} > logs/queue_g.out 2>&1

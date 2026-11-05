#!/bin/bash
# Relaunch the post-freeze queues after a pause: set aside outputs of evaluations interrupted mid-way
# (tune/val files without the final per-tag json), then start the resumable queues (they skip finished work).
cd /home/thuan/isci_method_20261025
for d in metrics/alloc_eval/*_int8ent; do
  for f in $d/sim_*_tune.json $d/r33_*_tune.json $d/best33_*_val.json; do
    [ -f "$f" ] || continue; t=${f%_tune.json}; t=${t%_val.json}
    if [ ! -f $t.json ]; then mkdir -p $d/_aborted_$(date +%Y%m%d_%H%M); mv ${t}_* $d/_aborted_$(date +%Y%m%d_%H%M)/; echo "set aside $t"; fi
  done
done
TS=$(date +%Y%m%d_%H%M)
for q in queue_k2b queue_k3b; do nohup ./$q.sh > logs/${q}_$TS.out 2>&1 & done
ls metrics/bootstrap/k1_yolo26l_greedy.json >/dev/null 2>&1 || nohup ./queue_boot_k1.sh > logs/queue_boot_k1_$TS.out 2>&1 &
echo "RESUMED $TS"

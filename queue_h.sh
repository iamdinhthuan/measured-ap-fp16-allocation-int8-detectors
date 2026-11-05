#!/bin/bash
# post-freeze baseline: reconstruction-error / output-SQNR knapsack (recon_select.py), then held-out eval
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
export ORT_THREADS=6
for M in yolo11s yolo26s; do
  echo "== $(date +%F_%T) recon $M"; $P recon_select.py --model $M --n-img 256 2>&1 | grep --line-buffered -v -E "constant_folding|Warning"; echo "== rc=$?"
done
for M in yolo11s yolo26s; do
  K=$(for c in recon sqnr; do for B in 0.03 0.05 0.1 0.15 0.2; do printf "%s@%s," $c $B; done; done); K=${K%,}
  echo "== $(date +%F_%T) eval $M $K"; $P eval_alloc.py --alloc configs/recon_${M}_int8ent_ho.json --keys $K --splits val; echo "== rc=$?"
done
echo QUEUE_H_DONE

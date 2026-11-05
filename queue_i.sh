#!/bin/bash
# post-freeze baseline on the confirmatory models (outside the confirmatory rule): recon/SQNR knapsack + val eval.
# Runs under sens.lock, i.e. after queue_f.
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
export ORT_THREADS=6 RECON_CHUNK=16
for M in yolo26l yolo11l; do
  [ -f metrics/sens_${M}_int8ent.json ] || { echo "skip $M: no sensitivity sweep"; continue; }
  echo "== $(date +%F_%T) recon $M"; $P recon_select.py --model $M --n-img 256 2>&1 | grep --line-buffered -v -E "constant_folding|Warning"
  K=$(for c in recon sqnr; do for B in 0.03 0.05 0.1 0.15 0.2; do printf "%s@%s," $c $B; done; done); K=${K%,}
  echo "== $(date +%F_%T) eval $M"; $P eval_alloc.py --alloc configs/recon_${M}_int8ent_ho.json --keys $K --splits val; echo "== rc=$?"
done
echo QUEUE_I_DONE

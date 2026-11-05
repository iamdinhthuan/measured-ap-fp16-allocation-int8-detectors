#!/bin/bash
# post-freeze: SQNR / recon scores on ExDark train images, knapsack, eval on ExDark test
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
export ORT_THREADS=6
echo "== $(date +%F_%T) recon exdark"; $P recon_select.py --model yolo11s --n-img 256 --split exdark --out-tag _exdark 2>&1 | grep --line-buffered -v -E "constant_folding|Warning"
K=$(for c in recon sqnr; do for B in 0.03 0.05 0.1 0.15 0.2; do printf "%s@%s," $c $B; done; done); K=${K%,}
echo "== $(date +%F_%T) eval"; $P eval_alloc.py --alloc configs/recon_yolo11s_int8ent_ho_exdark.json --keys $K --splits exdark_test --prefix exdark_; echo "== rc=$?"
echo QUEUE_H2_DONE

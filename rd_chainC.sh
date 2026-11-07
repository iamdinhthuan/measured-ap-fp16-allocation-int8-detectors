#!/bin/bash
# rules_11_12d_add1.md A3 (boxes+score SQNR). Starts after chain A, so at most two heavy jobs run.
cd ~/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
until grep -qE "CHAIN_A_DONE|FAILED|gate stopped" logs/rd_chainA.log 2>/dev/null; do sleep 60; done
grep -q CHAIN_A_DONE logs/rd_chainA.log || { echo "chain A did not finish"; exit 1; }
run() { echo "$(date -Is) $*"; "$@" || { echo "FAILED: $*"; exit 1; }; }
[ -e configs/fx_rtdetr-l_int8ent_ho_noise_n2000bs.json ] || run $P rd_fx_bs.py
sha256sum configs/fx_rtdetr-l_int8ent_ho_noise_n2000bs.json metrics/fx_scores_rtdetr-l_int8ent_noise_n2000bs.json >> logs/fx.sha256
run $P rd_queueX.py bs
echo CHAIN_C_DONE

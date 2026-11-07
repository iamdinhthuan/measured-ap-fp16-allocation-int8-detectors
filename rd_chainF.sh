#!/bin/bash
# rules_11_12d_add1.md A2 (secondary measured arm, base INT8 + F). Starts after chain B.
cd ~/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
until grep -qE "CHAIN_B_DONE|FAILED|gate stopped" logs/rd_chainB.log 2>/dev/null; do sleep 60; done
grep -q CHAIN_B_DONE logs/rd_chainB.log || { echo "chain B did not finish"; exit 1; }
run() { echo "$(date -Is) $*"; "$@" || { echo "FAILED: $*"; exit 1; }; }
[ -e metrics/sens_rtdetr-l_int8entF.json ] && grep -q '"fp16_all"' metrics/sens_rtdetr-l_int8entF.json && [ -e logs/rd_sensF.done ] || { run $P rd_sensF.py && touch logs/rd_sensF.done; }
sha256sum exports/rtdetr-l_int8entF.onnx metrics/sens_rtdetr-l_int8entF.json >> logs/fx.sha256
[ -e metrics/alloc_rtdetr-l_int8entF_ho.json ] || run $P rd_allocF.py
sha256sum metrics/alloc_rtdetr-l_int8entF_ho.json >> logs/fx.sha256
run $P rd_queueX.py F
echo CHAIN_F_DONE

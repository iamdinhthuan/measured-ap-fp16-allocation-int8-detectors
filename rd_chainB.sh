#!/bin/bash
# rules_11_12d.md arms 1, 2, 6, 7 (engine). Starts once the sweep is complete.
cd ~/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
until grep -q CHAIN_DONE logs/rd_chain.log 2>/dev/null; do
  grep -qE "GATE_STOP|GATE_ERROR|QUANT_FAILED" logs/rd_chain.log 2>/dev/null && { echo "gate stopped"; exit 0; }
  sleep 60
done
run() { echo "$(date -Is) $*"; "$@" || { echo "FAILED: $*"; exit 1; }; }
[ -e metrics/alloc_rtdetr-l_int8ent_ho.json ] || run $P rd_alloc.py
[ -e configs/greedy_rtdetr-l_int8ent_ho.json ] || run $P greedy_alloc.py --alloc metrics/alloc_rtdetr-l_int8ent_ho.json --out configs/greedy_rtdetr-l_int8ent_ho.json
FR=$($P -c "import rd_common;print(rd_common.FORCE_RX)")
[ -e metrics/random_rtdetr-l_int8ent_ho.json ] || run $P rd_wrap.py random_alloc.py --sens metrics/sens_rtdetr-l_int8ent.json --onnx exports/rtdetr-l.onnx --budgets 0.05,0.1,0.2 --seeds 1,2,3,4,5 --force-regex "$FR" --out metrics/random_rtdetr-l_int8ent_ho.json
sha256sum metrics/alloc_rtdetr-l_int8ent_ho.json configs/greedy_rtdetr-l_int8ent_ho.json metrics/random_rtdetr-l_int8ent_ho.json >> logs/fx.sha256
run $P rd_queue.py engine
echo CHAIN_B_DONE

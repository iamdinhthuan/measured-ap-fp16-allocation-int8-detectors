#!/bin/bash
# rules_11_12d.md: quantization -> gate -> (if GATE_PASS) block sweep
cd ~/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
until grep -q QUANT_DONE logs/rd_quant.log; do
  grep -q "Traceback" logs/rd_quant.log && { echo "QUANT_FAILED"; exit 1; }
  sleep 30
done
sha256sum exports/rtdetr-l_int8ent.onnx >> logs/fx.sha256
$P rd_gate.py > logs/rd_gate.log 2>&1 || { echo GATE_ERROR; exit 1; }
tail -1 logs/rd_gate.log
grep -q GATE_PASS logs/rd_gate.log || exit 0
sha256sum metrics/gate_rtdetr-l.json >> logs/fx.sha256
$P rd_sens.py > logs/rd_sens.log 2>&1 && sha256sum metrics/sens_rtdetr-l_int8ent.json >> logs/fx.sha256
echo CHAIN_DONE

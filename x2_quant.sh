#!/bin/bash
# X2 heavy job 1 (CPU): regenerate FP32/INT8 ONNX + gate G0 per model. setsid nohup ./x2_quant.sh >> logs/x2_quant.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
mkdir -p metrics/x2
P=/home/thuan/miniconda3/envs/qtsd/bin/python
until grep -q X2_PREP_DONE logs/x2_prep.out 2>/dev/null; do sleep 60; done
for M in yolo26l yolo26s yolo11s yolo11l; do
  [ -f metrics/x2/gate_$M.json ] && continue
  echo "== $(date +%F_%T) gate $M"
  $P x2_gate.py $M 2>&1 | grep -v "W:onnxruntime" | grep -v "^Calibrat" | tail -4
  echo "rc=${PIPESTATUS[0]}"
  sha256sum exports/$M.onnx exports/${M}_int8ent.onnx metrics/x2/gate_$M.json >> logs/fx.sha256
done
echo X2_QUANT_DONE

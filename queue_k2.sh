#!/bin/bash
# Post-freeze E2 (README 11.6): simulated-AP control. Same graphs/images/decoder as the engine sweep, onnxruntime
# FP32 fake quantization as executor; sim sensitivities -> allocate.py (same forced set, budgets) -> engines for
# every mean allocation whose blocks differ from the engine-based mean allocation.
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
for m in yolo11s yolo26s yolo11l yolo26l; do
  A=metrics/alloc_${m}_int8ent_ho.json; [ -f metrics/alloc_${m}_int8ent_ho_v2.json ] && A=metrics/alloc_${m}_int8ent_ho_v2.json
  RX=$($P -c "import json;print(json.load(open(\"$A\"))[\"force_regex\"])")
  run $P sim_sensitivity.py --model $m --base int8ent 2>&1 | grep -v "W:onnx"
  run $P allocate.py --sens metrics/sens_sim_${m}_int8ent.json --onnx exports/${m}.onnx --budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --out configs/simalloc_${m}_int8ent_ho.json
  KEYS=$($P - "$A" configs/simalloc_${m}_int8ent_ho.json <<"PY"
import json, sys
e, s = (json.load(open(f))["allocations"] for f in sys.argv[1:3])
d = [k for k in s if k.startswith("mean@") and sorted(s[k]["blocks"]) != sorted(e[k]["blocks"])]
print(",".join(d))
PY
)
  echo "differing mean keys for $m: [$KEYS]"
  [ -n "$KEYS" ] && run $P eval_alloc.py --alloc configs/simalloc_${m}_int8ent_ho.json --keys $KEYS --prefix sim_
done
echo QUEUE_K2_DONE

#!/bin/bash
# Confirmatory large models (protocol frozen in README section 10, 2026-10-25 22:00): run once, no tuning after viewing val.
# Per model: gate (fp32 + int8ent) -> engine sensitivity sweep on tune2000 -> MILP (mean/minimax/mean_matched)
# -> manual baselines -> Fisher/auto_quantize (2000 tune images) -> random -> FP32 val reference for bootstrap.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
{
for m in yolo26l yolo11l; do
  case $m in
    yolo26*) RX="/model\.\d+/((?:one2one_)?cv2\.\d/(?:one2one_)?cv2\.\d\.2|(?:one2one_)?cv3\.\d/(?:one2one_)?cv3\.\d\.2|dfl)/";;
    *)       RX="/model\.\d+/(cv2\.\d/cv2\.\d\.2|cv3\.\d/cv3\.\d\.2|dfl)/";;
  esac
  run $P gate_headroom.py --model $m --variants int8ent
  run $P block_sensitivity.py --model $m --base int8ent
  run $P allocate.py --sens metrics/sens_${m}_int8ent.json --onnx exports/$m.onnx --budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --out metrics/alloc_${m}_int8ent_ho.json
  $P - "$m" "$RX" <<PYEOF
import json, re, sys
from allocate import conv_macs
m, rx = sys.argv[1], sys.argv[2]
hi = max(int(k.split("/")[1].split(".")[1]) for k in conv_macs(f"exports/{m}.onnx") if k.startswith("/model."))
cfg = {"model": m, "base": "int8ent", "note": "folklore baselines, written by queue_f.sh before any evaluation",
       "allocations": {"headout": {"blocks": [], "regex": rx, "cost": None},
                       "stem_headout": {"blocks": [], "regex": "^/model\\\\.0/|" + rx, "cost": None},
                       "head": {"blocks": [], "regex": rf"^/model\\.{hi}/", "cost": None}}}
json.dump(cfg, open(f"configs/manual_{m}_int8ent.json", "w"), indent=1)
PYEOF
  run $P eval_alloc.py --alloc configs/manual_${m}_int8ent.json --keys headout,stem_headout,head
  KEYS=$($P -c "
import json
a = json.load(open(\"metrics/alloc_${m}_int8ent_ho.json\"))[\"allocations\"]
seen, keys = set(), []
for k, v in a.items():
    s = frozenset(v[\"blocks\"])
    if s not in seen:
        seen.add(s); keys.append(k)
print(\",\".join(keys))")
  echo "unique allocations: $KEYS"
  run $P eval_alloc.py --alloc metrics/alloc_${m}_int8ent_ho.json --keys $KEYS
  run $P autoquant_select.py --model $m --base int8ent --bits 8.02,8.04,8.08,8.15,8.3 --n-img 2000 --mac-budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --force-tag ho --out-tag _n2000
  run $P eval_alloc.py --alloc configs/fisher_${m}_int8ent_ho_n2000.json --keys fisher@0.03,fisher@0.05,fisher@0.1,fisher@0.15,fisher@0.2
  run $P eval_alloc.py --alloc configs/autoquant_${m}_int8ent_n2000.json --keys autoquant@8.02,autoquant@8.04,autoquant@8.08,autoquant@8.15,autoquant@8.3 --splits val
  run $P random_alloc.py --sens metrics/sens_${m}_int8ent.json --onnx exports/$m.onnx --force-regex "$RX" --out metrics/random_${m}_int8ent_ho.json
  K=$($P -c "print(\",\".join(f\"random@{b}_s{s}\" for b in (0.05,0.1,0.2) for s in range(1,6)))")
  run $P eval_alloc.py --alloc metrics/random_${m}_int8ent_ho.json --keys $K --splits val
  D=metrics/alloc_eval/${m}_int8ent
  [ -f $D/fp32_val.pred.json.gz ] || run $P infer_eval.py --engine exports/${m}_fp32.plan --gt data/coco_src/annotations/instances_val2017.json --images data/coco/images/val2017 --out $D/fp32_val.json --save-pred
done
echo QUEUE_F_DONE
} > logs/queue_f.out 2>&1

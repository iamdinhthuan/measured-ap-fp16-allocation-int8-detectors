#!/bin/bash
# Post-freeze E3 (README 11.6): search-cost-matched random baseline. K=33 random allocations per budget (as many
# engine evaluations as the sensitivity sweep), each built and scored on the tuning split only; the best one per
# budget (tuning AP) is then evaluated once on val2017.
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
SEEDS=$(seq -s, 101 133)
for m in yolo26l yolo11l yolo26s yolo11s; do
  A=metrics/alloc_${m}_int8ent_ho.json; [ -f metrics/alloc_${m}_int8ent_ho_v2.json ] && A=metrics/alloc_${m}_int8ent_ho_v2.json
  RX=$($P -c "import json;print(json.load(open(\"$A\"))[\"force_regex\"])")
  R=configs/random33_${m}_int8ent_ho.json
  [ -f $R ] || run $P random_alloc.py --sens metrics/sens_${m}_int8ent.json --onnx exports/${m}.onnx --budgets 0.05,0.1,0.2 --seeds $SEEDS --force-regex "$RX" --out $R
  KEYS=$($P -c "import json;print(\",\".join(json.load(open(\"$R\"))[\"allocations\"]))")
  run $P eval_alloc.py --alloc $R --keys $KEYS --splits tune --prefix r33_
  $P - $R $m <<"PY"
import json, sys
R, m = sys.argv[1:3]
p = json.load(open(R)); out = dict(p, allocations={}, criterion="best of 33 random (tuning AP)")
for B in ("0.05", "0.1", "0.2"):
    ks = [k for k in p["allocations"] if k.startswith(f"random@{B}_")]
    ap = {k: json.load(open(f"metrics/alloc_eval/{m}_int8ent/r33_{k.replace(@, _B)}.json"))["splits"]["tune"]["stats"]["AP"] for k in ks}
    best = max(ap, key=ap.get)
    out["allocations"][f"best33@{B}"] = dict(p["allocations"][best], source_key=best, tune_AP=ap[best], tune_AP_all=ap)
    print(m, B, best, round(100 * ap[best], 2), "median", round(100 * sorted(ap.values())[len(ap) // 2], 2))
json.dump(out, open(R.replace("random33_", "best33_"), "w"), indent=1)
PY
  run $P eval_alloc.py --alloc configs/best33_${m}_int8ent_ho.json --keys best33@0.05,best33@0.1,best33@0.2 --splits val
done
echo QUEUE_K3_DONE

#!/bin/bash
# Resume of queue_k2.sh (E2) after the pause of 27-10: finish the yolo11s sim evaluations, then the other models.
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
for m in yolo11s yolo26s yolo11l yolo26l; do
  A=metrics/alloc_${m}_int8ent_ho.json; [ -f metrics/alloc_${m}_int8ent_ho_v2.json ] && A=metrics/alloc_${m}_int8ent_ho_v2.json
  RX=$($P -c "import json;print(json.load(open(\"$A\"))[\"force_regex\"])")
  S=configs/simalloc_${m}_int8ent_ho.json
  [ -f metrics/sens_sim_${m}_int8ent.json ] || run $P sim_sensitivity.py --model $m --base int8ent > logs/sim_${m}.out 2>&1
  [ -f $S ] || run $P allocate.py --sens metrics/sens_sim_${m}_int8ent.json --onnx exports/${m}.onnx --budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --out $S
  KEYS=$($P - "$A" $S <<"PY"
import json, sys
e, s = (json.load(open(f))["allocations"] for f in sys.argv[1:3])
print(",".join(k for k in s if k.startswith("mean@") and sorted(s[k]["blocks"]) != sorted(e[k]["blocks"])))
PY
)
  echo "differing mean keys for $m: [$KEYS]"
  TODO=""
  for k in ${KEYS//,/ }; do t=sim_${k/@/_B}; [ -f metrics/alloc_eval/${m}_int8ent/$t.json ] || TODO="$TODO,$k"; done
  TODO=${TODO#,}
  echo "to evaluate for $m: [$TODO]"
  [ -n "$TODO" ] && run $P eval_alloc.py --alloc $S --keys $TODO --prefix sim_
done
echo QUEUE_K2B_DONE

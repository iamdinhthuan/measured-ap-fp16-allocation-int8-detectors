#!/bin/bash
# Post-freeze E5 (README 11.6): sim-AP sensitivities on the 256 tuning images used by the output-SQNR and
# reconstruction scores, then the same allocation and evaluation as E2 (prefix sim256_).
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
for m in yolo26l yolo11l yolo26s yolo11s; do
  A=metrics/alloc_${m}_int8ent_ho.json; [ -f metrics/alloc_${m}_int8ent_ho_v2.json ] && A=metrics/alloc_${m}_int8ent_ho_v2.json
  RX=$($P -c "import json;print(json.load(open(\"$A\"))[\"force_regex\"])")
  S=configs/simalloc256_${m}_int8ent_ho.json
  [ -f metrics/sens_sim_${m}_int8ent_n256.json ] || run $P sim_sensitivity.py --model $m --base int8ent --n-img 256 --out-tag _n256 > logs/sim256_${m}.out 2>&1
  [ -f $S ] || run $P allocate.py --sens metrics/sens_sim_${m}_int8ent_n256.json --onnx exports/${m}.onnx --budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --out $S
  KEYS=$($P - "$A" $S <<"PY"
import json, sys
e, s = (json.load(open(f))["allocations"] for f in sys.argv[1:3])
print(",".join(k for k in s if k.startswith("mean@") and sorted(s[k]["blocks"]) != sorted(e[k]["blocks"])))
PY
)
  echo "differing mean keys for $m: [$KEYS]"
  TODO=""
  for k in ${KEYS//,/ }; do t=sim256_${k/@/_B}; [ -f metrics/alloc_eval/${m}_int8ent/$t.json ] || TODO="$TODO,$k"; done
  TODO=${TODO#,}
  [ -n "$TODO" ] && run $P eval_alloc.py --alloc $S --keys $TODO --prefix sim256_
done
echo QUEUE_K5_DONE

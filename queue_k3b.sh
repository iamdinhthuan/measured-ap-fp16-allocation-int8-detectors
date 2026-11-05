#!/bin/bash
# Resume of queue_k3.sh (E3) after the pause of 27-10; the best-of-33 selection moved to best33.py
# (the inline version had lost its quotes and would not have run).
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
SEEDS=$(seq -s, 101 133)
for m in yolo26l yolo11l yolo26s yolo11s; do
  A=metrics/alloc_${m}_int8ent_ho.json; [ -f metrics/alloc_${m}_int8ent_ho_v2.json ] && A=metrics/alloc_${m}_int8ent_ho_v2.json
  RX=$($P -c "import json;print(json.load(open(\"$A\"))[\"force_regex\"])")
  R=configs/random33_${m}_int8ent_ho.json
  [ -f $R ] || run $P random_alloc.py --sens metrics/sens_${m}_int8ent.json --onnx exports/${m}.onnx --budgets 0.05,0.1,0.2 --seeds $SEEDS --force-regex "$RX" --out $R
  TODO=$($P missing_keys.py $R $m r33_ random@)
  echo "to evaluate for $m: $(echo $TODO | tr , '\n' | grep -c .) keys"
  [ -n "$TODO" ] && run $P eval_alloc.py --alloc $R --keys $TODO --splits tune --prefix r33_
  [ -f configs/best33_${m}_int8ent_ho.json ] || run $P best33.py $R $m
  [ -f metrics/alloc_eval/${m}_int8ent/best33_B0.05.json ] || run $P eval_alloc.py --alloc configs/best33_${m}_int8ent_ho.json --keys best33@0.05,best33@0.1,best33@0.2 --splits val
done
echo QUEUE_K3B_DONE

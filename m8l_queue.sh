#!/bin/bash
# M8L (rules_20261108_m8l.md, written before any result): YOLO11-l. prep -> reconstruct+gate G1/G2 -> F-sweep -> allocate
# -> eval differing mean keys (fxs_ new, m8lr_ published rebuild = G3) -> FP32 val preds -> paired bootstrap.
# Second heavy job only. Usage: setsid nohup ./m8l_queue.sh >> logs/m8l_queue.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
M=yolo11l
until grep -q M8L_PREP_DONE logs/m8l_prep.out 2>/dev/null; do sleep 120; done
echo "== $(date +%F_%T) prep done"
gate() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    r=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$f" -ge 8000 ] && [ "$r" -ge 15 ] && return
    echo "$(date +%T) gpu free ${f} MiB ram ${r} GB, waiting"; sleep 300
  done
}
gate
if [ ! -f metrics/m8l_gate.json ]; then
  echo "== $(date +%F_%T) reconstruct + gate"
  $P m8l_gate.py 2>&1 | grep -v "W:onnxruntime"
  rc=${PIPESTATUS[0]}; echo "rc=$rc"
fi
sha256sum exports/$M.onnx exports/${M}_int8ent.onnx metrics/m8l_gate.json >> logs/fx.sha256
$P -c "import json,sys;sys.exit(0 if json.load(open('metrics/m8l_gate.json'))['gate_pass'] else 3)" || { echo M8L_ABORT_GATE; exit 3; }
gate
echo "== $(date +%F_%T) sweep"
$P fx_sens_f.py --model $M 2>&1 | grep -v "W:onnxruntime"
rc=${PIPESTATUS[0]}; echo "rc=$rc"; [ $rc -eq 0 ] || { echo M8L_FAILED_SWEEP; exit 1; }
C=configs/fx_${M}_int8ent_ho_fsweep.json
PUB=metrics/alloc_${M}_int8ent_ho.json
RX=$($P -c "import json;print(json.load(open('$PUB'))['force_regex'])")
[ -f $C ] || $P allocate.py --sens metrics/sens_${M}_int8ent_fbase.json --onnx exports/${M}.onnx \
  --budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --out $C
sha256sum metrics/sens_${M}_int8ent_fbase.json $C >> logs/fx.sha256
KEYS=$($P fx_m8_tail.py $C $PUB)
echo "== $(date +%F_%T) mean keys differing from published: '${KEYS}'"
D=metrics/alloc_eval/${M}_int8ent
if [ -n "$KEYS" ]; then
  for pfx in fxs_ m8lr_; do
    [ $pfx = fxs_ ] && A=$C || A=$PUB
    todo=""
    for k in ${KEYS//,/ }; do [ -f $D/${pfx}${k/@/_B}.json ] || todo="$todo,$k"; done
    if [ -n "$todo" ]; then
      gate
      echo "== $(date +%F_%T) eval $pfx ${todo#,}"
      $P eval_alloc.py --alloc $A --keys ${todo#,} --splits val --prefix $pfx 2>&1 | grep -v "W:onnxruntime"
      rc=${PIPESTATUS[0]}; echo "rc=$rc"; [ $rc -eq 0 ] || { echo M8L_FAILED_EVAL; exit 1; }
    fi
  done
  mkdir -p metrics/m8l
  if [ ! -f metrics/m8l/fp32_val.json ]; then
    gate
    echo "== $(date +%F_%T) fp32 val preds"
    $P tools/build_engine.py --onnx exports/$M.onnx --engine exports/tmp/${M}_m8l_fp32.plan >> logs/m8l_fp32.log 2>&1
    $P infer_eval.py --engine exports/tmp/${M}_m8l_fp32.plan --gt data/coco_src/annotations/instances_val2017.json \
      --images data/coco/images/val2017 --out metrics/m8l/fp32_val.json --save-pred 2>&1 | grep -v "W:onnxruntime" | tail -3
    rm -f exports/tmp/${M}_m8l_fp32.plan
  fi
  SETS="fp32=metrics/m8l/fp32_val.pred.json.gz"; PAIRS=""
  for k in ${KEYS//,/ }; do
    b=${k#mean@}
    SETS="$SETS,fsw_$b=$D/fxs_mean_B${b}_val.pred.json.gz,pub_$b=$D/m8lr_mean_B${b}_val.pred.json.gz"
    PAIRS="$PAIRS,pub_$b:fsw_$b"
  done
  mkdir -p metrics/bootstrap
  O=metrics/bootstrap/m8l_${M}_fsweep.json
  if [ ! -f $O ]; then
    echo "== $(date +%F_%T) bootstrap"
    $P bootstrap_fast.py --gt data/coco_src/annotations/instances_val2017.json --sets "$SETS" --ref fp32 \
      --pairs "${PAIRS#,}" --n 1000 --out $O 2>&1 | grep -v "^resample"
    echo "rc=${PIPESTATUS[0]}"
  fi
  sha256sum $O $D/fxs_mean_B*_val.json $D/m8lr_mean_B*_val.json metrics/m8l/fp32_val.json >> logs/fx.sha256
fi
echo M8L_QUEUE_DONE

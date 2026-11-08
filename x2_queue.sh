#!/bin/bash
# X2 heavy job 2 (GPU), rules_20261108_x2.md: per model, after G0: rebuild published (x2r_/x2fr_) and exact (x2e_/x2fe_)
# selections, G3, FP32 val preds, paired bootstrap. setsid nohup ./x2_queue.sh >> logs/x2_queue.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
GT=data/coco_src/annotations/instances_val2017.json
mkdir -p metrics/x2
gate() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    r=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$f" -ge 8000 ] && [ "$r" -ge 12 ] && return
    echo "$(date +%T) gpu free ${f} MiB ram ${r} GB, waiting"; sleep 300
  done
}
until grep -q X2_PREP_DONE logs/x2_prep.out 2>/dev/null; do sleep 60; done
$P x2_plans.py > metrics/x2/plans.txt || { echo X2_FAILED_PLANS; exit 1; }
cat metrics/x2/plans.txt
sha256sum configs/x2_*_exact.json configs/x2_*_fexact.json metrics/x2/plans.txt >> logs/fx.sha256
evalk() {  # plan keys prefix
  local A=$1 K=$2 pfx=$3 todo=""
  for k in ${K//,/ }; do [ -f $D/${pfx}${k/@/_B}.json ] || todo="$todo,$k"; done
  [ -z "$todo" ] && return 0
  gate
  echo "== $(date +%F_%T) eval $M $pfx ${todo#,}"
  $P eval_alloc.py --alloc $A --keys ${todo#,} --splits val --prefix $pfx 2>&1 | grep -v "W:onnxruntime"
  local rc=${PIPESTATUS[0]}; echo "rc=$rc"; return $rc
}
for M in yolo26l yolo26s yolo11s yolo11l; do
  until [ -f metrics/x2/gate_$M.json ] || grep -q X2_QUANT_DONE logs/x2_quant.out 2>/dev/null; do sleep 60; done
  [ -f metrics/x2/gate_$M.json ] || { echo "X2 $M NO_GATE_FILE (regeneration failed; reported, not evaluated)"; continue; }
  $P -c "import json,sys;sys.exit(0 if json.load(open('metrics/x2/gate_$M.json'))['G0_pass'] else 3)" || { echo "X2 $M G0_FAIL (reported, not evaluated)"; continue; }
  D=metrics/alloc_eval/${M}_int8ent
  KP=$(awk -v m=$M '$1==m && $2=="published"{print $3}' metrics/x2/plans.txt)
  KF=$(awk -v m=$M '$1==m && $2=="fsweep"{print $3}' metrics/x2/plans.txt)
  [ -n "$KP" ] && { evalk metrics/alloc_${M}_int8ent_ho.json $KP x2r_ && evalk configs/x2_${M}_exact.json $KP x2e_ || { echo X2_FAILED_EVAL; exit 1; }; }
  [ -n "$KF" ] && { evalk configs/fx_${M}_int8ent_ho_fsweep.json $KF x2fr_ && evalk configs/x2_${M}_fexact.json $KF x2fe_ || { echo X2_FAILED_EVAL; exit 1; }; }
  # G3: rebuilt published selections vs published val AP
  $P - $M "$KP" "$KF" <<'PY' | tee -a metrics/x2/g3.txt
import json, sys
m, kp, kf = sys.argv[1], sys.argv[2], sys.argv[3]
d = f"metrics/alloc_eval/{m}_int8ent/"
for keys, pub, reb in ((kp, "", "x2r_"), (kf, "fxs_", "x2fr_")):
    for k in filter(None, keys.split(",")):
        t = k.replace("@", "_B")
        a = 100 * json.load(open(d + f"{pub}{t}_val.json"))["stats"]["AP"]
        b = 100 * json.load(open(d + f"{reb}{t}_val.json"))["stats"]["AP"]
        f = abs(b - a) > 0.07
        print(f"G3 {m} {reb}{t} rebuilt={b:.3f} published={a:.3f} diff={b-a:+.3f}{' FLAG' if f else ''}")
PY
  if [ $M = yolo11l ]; then FP=metrics/m8l/fp32_val.pred.json.gz; else
    FP=metrics/x2/fp32_${M}_val.pred.json.gz
    if [ ! -f metrics/x2/fp32_${M}_val.json ]; then
      gate; echo "== $(date +%F_%T) fp32 $M"
      $P tools/build_engine.py --onnx exports/$M.onnx --engine exports/tmp/${M}_x2_fp32.plan >> logs/x2_fp32.log 2>&1
      $P infer_eval.py --engine exports/tmp/${M}_x2_fp32.plan --gt $GT --images data/coco/images/val2017 \
        --out metrics/x2/fp32_${M}_val.json --save-pred 2>&1 | grep -v "W:onnxruntime" | tail -2
      rm -f exports/tmp/${M}_x2_fp32.plan
    fi
  fi
  SETS="fp32=$FP"; PAIRS=""
  for k in ${KP//,/ }; do b=${k#mean@}; SETS="$SETS,r_$b=$D/x2r_mean_B${b}_val.pred.json.gz,e_$b=$D/x2e_mean_B${b}_val.pred.json.gz"; PAIRS="$PAIRS,r_$b:e_$b"; done
  for k in ${KF//,/ }; do b=${k#mean@}; SETS="$SETS,fr_$b=$D/x2fr_mean_B${b}_val.pred.json.gz,fe_$b=$D/x2fe_mean_B${b}_val.pred.json.gz"; PAIRS="$PAIRS,fr_$b:fe_$b"; done
  O=metrics/bootstrap/x2_${M}.json
  if [ ! -f $O ]; then
    echo "== $(date +%F_%T) bootstrap $M"
    $P bootstrap_fast.py --gt $GT --sets "$SETS" --ref fp32 --pairs "${PAIRS#,}" --n 1000 --out $O 2>&1 | grep -v "^resample\|^packed"
    echo "rc=${PIPESTATUS[0]}"
  fi
  sha256sum $O $D/x2*_val.json metrics/x2/fp32_${M}_val.json 2>/dev/null >> logs/fx.sha256
done
sha256sum metrics/x2/g3.txt >> logs/fx.sha256
echo X2_QUEUE_DONE

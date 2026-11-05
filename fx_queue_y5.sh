#!/bin/bash
# README 11.10 Y4 bootstrap + Y5 (rules_11_10.md, written before any Y4/Y5 result).
# Waits for fx_exdarkcal.py (keeps the 2-heavy-job limit), runs the Y4 group bootstrap, then Y5 per YOLO26 model:
# training-graph Fisher score + knapsack, val2017 engines (prefix fxo_), paired bootstrap.
# Usage: nohup ./fx_queue_y5.sh >> logs/fx_queue_y5.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
gpu_wait() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$f" -ge 10000 ] && return
    echo "$(date +%T) gpu free ${f} MiB, waiting"; sleep 300
  done
}
until grep -q EXDARKCAL_DONE logs/fx_exdarkcal.out; do
  if ! kill -0 431070 2> /dev/null; then  # PID of fx_exdarkcal.py
    grep -q EXDARKCAL_DONE logs/fx_exdarkcal.out || { echo "$(date +%T) exdarkcal not running and not done"; break; }
  fi
  sleep 300
done
if grep -q EXDARKCAL_DONE logs/fx_exdarkcal.out; then
  D=metrics/alloc_eval/yolo11s_int8ent; E=metrics/exdarkcal
  O=metrics/bootstrap/g_yolo11s_exdarkcal.json
  S="fp32=metrics/exdark/yolo11s_fp32_test.pred.json.gz,base=$D/exdark_base_exdark_test.pred.json.gz,headout=$D/exdark_headout_exdark_test.pred.json.gz"
  S="$S,cbase=$E/exdarkcal_base_exdark_test.pred.json.gz,cheadout=$E/exdarkcal_headout_exdark_test.pred.json.gz"
  echo "== $(date +%F_%T) boot y4"
  [ -f $O ] || nice -n 15 $P bootstrap_groups.py --gt data/exdark/exdark_test.json --sets "$S" --ref fp32 \
    --pairs base:cbase,headout:cheadout --groups attr:light \
    --group-names Ambient,Strong,Single,Weak,Twilight,Object,Low,Window --n 1000 --out $O
  echo "rc=$?"
  sha256sum $O $E/*.json $E/*.pred.json.gz exports/yolo11s_int8ent_exdarkcal.onnx >> logs/fx.sha256
fi
FRX='/model\.\d+/((?:one2one_)?cv2\.\d/(?:one2one_)?cv2\.\d\.2|(?:one2one_)?cv3\.\d/(?:one2one_)?cv3\.\d\.2|dfl)/'
BUD=0.03,0.05,0.1,0.15,0.2
for MN in yolo26s:512 yolo26l:2000; do
  M=${MN%:*}; N=${MN#*:}
  C=configs/fisher_${M}_int8ent_ho_o2m.json
  if [ ! -f $C ]; then
    gpu_wait; echo "== $(date +%F_%T) score y5 $M n=$N"
    $P fx_fisher_o2m.py --model $M --n-img $N --bits 8.4 --mac-budgets $BUD --force-regex "$FRX" --force-tag ho 2>&1 \
      | grep -v -E "it/s\]|s/it\]|W:onnxruntime"
    echo "rc=${PIPESTATUS[0]}"
  fi
  [ -f $C ] || { echo "no $C, skip $M"; continue; }
  todo=""
  for B in ${BUD//,/ }; do
    [ -f metrics/alloc_eval/${M}_int8ent/fxo_fisher_B$B.json ] || todo="$todo,fisher@$B"
  done
  if [ -n "$todo" ]; then
    gpu_wait; echo "== $(date +%F_%T) eval y5 $M ${todo#,}"
    $P eval_alloc.py --alloc $C --keys ${todo#,} --splits val --prefix fxo_ 2>&1 | grep -v "W:onnxruntime"
    echo "rc=${PIPESTATUS[0]}"
  fi
  echo "== $(date +%F_%T) boot y5 $M"
  $P fx_boot2.py --model $M --out fx_${M}_y5 --budgets $BUD --arm mean=@mean --arm fo2m=@fxo_fisher \
    --arm fisher=@fisher --pairs fo2m:mean,fisher:fo2m 2>&1 | grep -v "^resample"
  echo "rc=${PIPESTATUS[0]}"
  sha256sum metrics/bootstrap/fx_${M}_y5.json $C configs/autoquant_${M}_int8ent_o2m.json >> logs/fx.sha256
done
echo FX_QUEUE_Y5_DONE

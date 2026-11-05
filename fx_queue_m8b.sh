#!/bin/bash
# M8 (rules_11_10.md, appended before any result): YOLO26-l sweep with F restored -> allocate -> eval (val, fxs_) -> bootstrap.
# fx_queue_m8b.sh: identical to fx_queue_m8.sh minus the Y2 gate (relaunch after the 04-11 reboot; one Y2 chain runs beside it).
# Usage: nohup ./fx_queue_m8.sh >> logs/fx_queue_m8b.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
M=yolo26l
echo "== $(date +%F_%T) M8 restart after server reboot of 2026-11-04 ~12:13 (no M8 step had run)"
gate() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    r=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$f" -ge 8000 ] && [ "$r" -ge 15 ] && return
    echo "$(date +%T) gpu free ${f} MiB ram ${r} GB, waiting"; sleep 300
  done
}
gate
echo "== $(date +%F_%T) sweep"
$P fx_sens_f.py --model $M 2>&1 | grep -v "W:onnxruntime"
rc=${PIPESTATUS[0]}; echo "rc=$rc"; [ $rc -eq 0 ] || { echo FX_QUEUE_M8_FAILED; exit 1; }
C=configs/fx_${M}_int8ent_ho_fsweep.json
RX=$($P -c "import json;print(json.load(open('metrics/alloc_${M}_int8ent_ho.json'))['force_regex'])")
[ -f $C ] || $P allocate.py --sens metrics/sens_${M}_int8ent_fbase.json --onnx exports/${M}.onnx \
  --budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --out $C
sha256sum metrics/sens_${M}_int8ent_fbase.json $C >> logs/fx.sha256
KEYS=$($P fx_m8_tail.py $C metrics/alloc_${M}_int8ent_ho.json)
echo "== $(date +%F_%T) mean keys differing from published: '${KEYS}'"
if [ -n "$KEYS" ]; then
  todo=""
  for k in ${KEYS//,/ }; do
    [ -f metrics/alloc_eval/${M}_int8ent/fxs_${k/@/_B}.json ] || todo="$todo,$k"
  done
  if [ -n "$todo" ]; then
    gate
    $P eval_alloc.py --alloc $C --keys ${todo#,} --splits val --prefix fxs_ 2>&1 | grep -v "W:onnxruntime"
    echo "rc=${PIPESTATUS[0]}"
  fi
fi
echo "== $(date +%F_%T) boot m8"
$P fx_boot2.py --model $M --out fx_${M}_m8 --arm mean=@mean --arm fsw=$C:mean --prefer mean=mean --pairs fsw:mean --dry
$P fx_boot2.py --model $M --out fx_${M}_m8 --arm mean=@mean --arm fsw=$C:mean --prefer mean=mean --pairs fsw:mean 2>&1 | grep -v "^resample"
echo "rc=${PIPESTATUS[0]}"
sha256sum metrics/bootstrap/fx_${M}_m8.json >> logs/fx.sha256
echo FX_QUEUE_M8_DONE

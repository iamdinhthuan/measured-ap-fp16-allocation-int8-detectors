#!/bin/bash
# README 11.10 Y1 (rules_11_10.md, written before any result): head-first SQNR hybrid.
# Per model: eval feasible hyb@B allocations on val2017 (prefix fxh_), then paired bootstrap
# hyb:mean, sqnrq:hyb, head:hyb at the feasible budgets. One chain, sequential (2-heavy-job limit).
# Usage: nohup ./fx_queue_y.sh >> logs/fx_queue_y.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
gpu_wait() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$f" -ge 8000 ] && return
    echo "$(date +%T) gpu free ${f} MiB, waiting"; sleep 300
  done
}
for M in yolo26l yolo11l yolo26s yolo11s; do
  C=configs/fx_${M}_int8ent_ho_hyb.json
  BUD=$($P -c "import json;print(','.join(k.split('@')[1] for k in json.load(open('$C'))['allocations']))")
  todo=""
  for B in ${BUD//,/ }; do
    [ -f metrics/alloc_eval/${M}_int8ent/fxh_hyb_B$B.json ] || todo="$todo,hyb@$B"
  done
  if [ -n "$todo" ]; then
    gpu_wait; echo "== $(date +%F_%T) eval $M ${todo#,}"
    $P eval_alloc.py --alloc $C --keys ${todo#,} --splits val --prefix fxh_ 2>&1 | grep -v "W:onnxruntime"
    echo "rc=${PIPESTATUS[0]}"
  fi
  PREF=""
  [ $M = yolo26l ] && PREF="--prefer sqnrq=sqnr --prefer sqnrq=fx256_sqnrq"
  echo "== $(date +%F_%T) boot y1 $M budgets $BUD"
  $P fx_boot2.py --model $M --out fx_${M}_y1 --budgets $BUD --arm mean=@mean \
    --arm hyb=$C:hyb --arm sqnrq=configs/fx_${M}_int8ent_ho_noise_n256.json:sqnrq \
    --arm head=!metrics/alloc_eval/${M}_int8ent/head_val.pred.json.gz $PREF \
    --pairs hyb:mean,sqnrq:hyb,head:hyb 2>&1 | grep -v "^resample"
  echo "rc=${PIPESTATUS[0]}"
  sha256sum metrics/bootstrap/fx_${M}_y1.json $C >> logs/fx.sha256
done
echo FX_QUEUE_Y_DONE

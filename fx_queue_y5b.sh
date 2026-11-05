#!/bin/bash
# README 11.10 Y5, YOLO26-l only: re-run after a CUDA OOM in fx_queue_y5.sh (GPU memory held by other users).
# Same command and arguments as fx_queue_y5.sh; waits for >= 14 GB free GPU memory. Rule unchanged (rules_11_10.md Y5).
# Usage: nohup ./fx_queue_y5b.sh >> logs/fx_queue_y5b.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu_wait() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$f" -ge 14000 ] && return
    echo "$(date +%T) gpu free ${f} MiB, waiting"; sleep 300
  done
}
FRX='/model\.\d+/((?:one2one_)?cv2\.\d/(?:one2one_)?cv2\.\d\.2|(?:one2one_)?cv3\.\d/(?:one2one_)?cv3\.\d\.2|dfl)/'
BUD=0.03,0.05,0.1,0.15,0.2
M=yolo26l; N=2000
C=configs/fisher_${M}_int8ent_ho_o2m.json
for try in 1 2 3; do
  [ -f $C ] && break
  gpu_wait; echo "== $(date +%F_%T) score y5 $M n=$N try $try"
  $P fx_fisher_o2m.py --model $M --n-img $N --bits 8.4 --mac-budgets $BUD --force-regex "$FRX" --force-tag ho 2>&1 \
    | grep -v -E "it/s\]|s/it\]|W:onnxruntime" | tail -40
  echo "rc=${PIPESTATUS[0]}"
done
[ -f $C ] || { echo "no $C, giving up"; exit 1; }
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
echo FX_QUEUE_Y5B_DONE

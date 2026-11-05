#!/bin/bash
# README 11.10 Y5, YOLO26-l: bootstrap only. fx_queue_y5b.sh scored and evaluated (configs/fisher_yolo26l_int8ent_ho_o2m.json,
# metrics/alloc_eval/yolo26l_int8ent/fxo_fisher_B*.json) but its fx_boot2.py died silently after packing (no output file).
# Same fx_boot2 command; up to 3 attempts; waits for >= 25 GB free RAM. Rule unchanged (rules_11_10.md Y5).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
M=yolo26l; BUD=0.03,0.05,0.1,0.15,0.2; O=metrics/bootstrap/fx_${M}_y5.json
for try in 1 2 3; do
  [ -f $O ] && break
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 25 ]; do echo "$(date +%T) ram low, waiting"; sleep 300; done
  echo "== $(date +%F_%T) boot y5 $M try $try"
  nice -n 10 $P fx_boot2.py --model $M --out fx_${M}_y5 --budgets $BUD --arm mean=@mean --arm fo2m=@fxo_fisher \
    --arm fisher=@fisher --pairs fo2m:mean,fisher:fo2m 2>&1 | grep -v "^resample"
  echo "rc=${PIPESTATUS[0]}"
done
[ -f $O ] && sha256sum $O configs/fisher_${M}_int8ent_ho_o2m.json configs/autoquant_${M}_int8ent_o2m.json >> logs/fx.sha256
echo FX_QUEUE_Y5C_DONE

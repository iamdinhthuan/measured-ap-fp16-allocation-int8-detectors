#!/bin/bash
# X1 + X3 bootstraps (rules_11_9.md), exploratory, post-freeze. Sequential within each branch.
#   L: large detectors, one call per model: X3 pairs (sqnrq:simd1, sqnrq:simd2, simd1:mean, simd2:mean)
#      and X1 pairs (bnorm:mean, chnorm:mean, sqnrq:bnorm). Starts after the b1 queue ends (2-heavy-job limit).
#   S: small detectors, X1 pairs only. Starts after the X2 bootstraps end.
# Usage: nohup ./fx_boot_x13.sh L >> logs/fx_boot_x13_L.out 2>&1 &   (same with S)
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
DRY=${DRY:-}
waitfor() { while ! grep -q "^$2" $1 2>/dev/null; do echo "$(date +%T) waiting for $2"; sleep 600; done; }
if [ "$1" = L ]; then
  [ -z "$DRY" ] && waitfor logs/fx_queue_b1.out FX_QUEUE_B1_DONE
  for M in yolo26l yolo11l; do
    C=configs/fx_${M}_int8ent_ho; S=configs/simalloc_${M}_int8ent_ho
    echo "== $(date +%F_%T) boot x13 $M"
    $P fx_boot.py --model $M --out fx_${M}_x13 $DRY --arm mean=@mean \
      --arm sqnrq=${C}_noise_n2000.json:sqnrq --arm simd1=${S}_d1.json:mean --arm simd2=${S}_d2.json:mean \
      --arm bnorm=${C}_bnorm_n2000.json:bnorm --arm chnorm=${C}_bnorm_n2000.json:chnorm \
      --prefer sqnrq=sqnr --prefer sqnrq=fx256_sqnrq --prefer simd1=mean --prefer simd1=sim_mean --prefer simd1=fxd1_mean \
      --prefer simd2=mean --prefer simd2=sim_mean --prefer simd2=fxd2_mean \
      --pairs sqnrq:simd1,sqnrq:simd2,simd1:mean,simd2:mean,bnorm:mean,chnorm:mean,sqnrq:bnorm 2>&1 | grep -v "^resample"
    echo "rc=${PIPESTATUS[0]}"
  done
  echo FX_BOOT_X13_L_DONE
else
  [ -z "$DRY" ] && waitfor logs/fx_boot_x2.out FX_BOOT_X2_DONE
  [ -z "$DRY" ] && waitfor logs/fx_queue_b1.out FX_QUEUE_B1_DONE
  for M in yolo26s yolo11s; do
    C=configs/fx_${M}_int8ent_ho
    echo "== $(date +%F_%T) boot x1 $M"
    $P fx_boot.py --model $M --out fx_${M}_x1 $DRY --arm mean=@mean \
      --arm sqnrq=${C}_noise_n2000.json:sqnrq --arm bnorm=${C}_bnorm_n2000.json:bnorm --arm chnorm=${C}_bnorm_n2000.json:chnorm \
      --prefer sqnrq=sqnr --prefer sqnrq=fx256_sqnrq \
      --pairs bnorm:mean,chnorm:mean,sqnrq:bnorm 2>&1 | grep -v "^resample"
    echo "rc=${PIPESTATUS[0]}"
  done
  echo FX_BOOT_X13_S_DONE
fi

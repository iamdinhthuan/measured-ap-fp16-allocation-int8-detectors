#!/bin/bash
# X2 bootstraps (rules_11_9.md), sequential, one at a time
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
for M in yolo26l yolo11l yolo26s yolo11s; do
  echo "== $(date +%F_%T) boot x2 $M"
  $P fx_boot.py --model $M --out fx_${M}_x2 --arm mean=@mean --arm fisherblk=configs/fx_${M}_int8ent_ho_fisherblk.json:fisherblk --pairs fisherblk:mean
  echo "rc=$?"
done
echo FX_BOOT_X2_DONE

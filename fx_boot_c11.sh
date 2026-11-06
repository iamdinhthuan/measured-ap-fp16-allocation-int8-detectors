#!/bin/bash
# rules_11_11c.md C: paired bootstrap fit:mean (b - a = measured - FIT) and fisherblk:fit, COCO val. Usage: ./fx_boot_c11.sh "yolo26l yolo11l"
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
for M in $1; do
  [ -f metrics/bootstrap/fx_${M}_c11.json ] && { echo "skip $M"; continue; }
  echo "== $(date +%F_%T) $M"
  $P fx_boot.py --model $M --out fx_${M}_c11 --arm mean=@mean --arm fit=configs/fx_${M}_int8ent_ho_fit.json:fit \
    --arm fisherblk=configs/fx_${M}_int8ent_ho_fisherblk.json:fisherblk --prefer fit=fxf_fit \
    --pairs fit:mean,fisherblk:fit 2>&1 | grep -v "^resample"
  echo "== rc=${PIPESTATUS[0]} $M"
  [ -f metrics/bootstrap/fx_${M}_c11.json ] && sha256sum metrics/bootstrap/fx_${M}_c11.json >> logs/fx.sha256
done
echo FX_BOOT_C11_DONE "$1"

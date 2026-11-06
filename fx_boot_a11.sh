#!/bin/bash
# rules_11_11.md A: paired bootstrap pandey:mean, sqnrfx:pandey on COCO val (20 pairs); identical selections share files.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
for M in yolo26l yolo11l yolo26s yolo11s; do
  C=configs/fx_${M}_int8ent_ho
  [ -f metrics/bootstrap/fx_${M}_a11.json ] && { echo "skip $M"; continue; }
  echo "== $(date +%F_%T) $M"
  $P fx_boot.py --model $M --out fx_${M}_a11 --arm mean=@mean --arm sqnrfx=${C}_noise_n256.json:sqnrq \
    --arm pandey=${C}_pandey_n256.json:pandey \
    --prefer sqnrfx=sqnr --prefer sqnrfx=fx256_sqnrq --prefer pandey=sqnr --prefer pandey=fx256_sqnrq --prefer pandey=fxp_pandey \
    --pairs pandey:mean,sqnrfx:pandey 2>&1 | grep -v "^resample"
  echo "== rc=${PIPESTATUS[0]} $M"
  [ -f metrics/bootstrap/fx_${M}_a11.json ] && sha256sum metrics/bootstrap/fx_${M}_a11.json >> logs/fx.sha256
done
echo FX_BOOT_A11_DONE

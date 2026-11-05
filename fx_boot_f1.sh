#!/bin/bash
# README 11.8 F1: corrected SQNR/recon arms vs engine mean (val2017 paired bootstrap). Two chains: A (large), B (small).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
run() { local M=$1; echo "== $(date +%F_%T) $M"
  $P fx_boot.py --model $M --out fx_${M}_f1 --arm mean=@mean \
    --arm sqnrfx=configs/fx_${M}_int8ent_ho_noise_n256.json:sqnrq \
    --arm reconfx=configs/fx_${M}_int8ent_ho_noise_n256.json:reconq \
    --pairs sqnrfx:mean,reconfx:mean 2>&1 | grep -v "^resample"
  echo "== rc=${PIPESTATUS[0]} $M"; }
if [ "$1" = A ]; then run yolo26l; run yolo11l; echo F1BOOT_A_DONE
else run yolo26s; run yolo11s; echo F1BOOT_B_DONE; fi

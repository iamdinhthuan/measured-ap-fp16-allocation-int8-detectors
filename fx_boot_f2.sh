#!/bin/bash
# README 11.8 F2: criterion x direction factorial, paired bootstrap on val2017. A: large models; B: small (after the ExDark bootstrap).
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
args() { local M=$1; echo --model $M --arm mean=@mean --arm simmean=configs/simalloc_${M}_int8ent_ho.json:mean \
  --arm apq=configs/fx_${M}_int8ent_ho_apq_n2000.json:apq \
  --arm sqnrq=configs/fx_${M}_int8ent_ho_noise_n2000.json:sqnrq --arm sqnrr=configs/fx_${M}_int8ent_ho_noise_n2000.json:sqnrr \
  --pairs sqnrq:apq,sqnrr:simmean,apq:simmean,sqnrq:sqnrr,apq:mean,sqnrq:mean,sqnrr:mean,simmean:mean; }
run() { local M=$1; echo "== $(date +%F_%T) $M"
  $P fx_boot.py $(args $M) --out fx_${M}_f2 2>&1 | grep -v "^resample"; echo "== rc=${PIPESTATUS[0]} $M"; }
if [ "$1" = dry ]; then for M in yolo26l yolo11l yolo26s yolo11s; do $P fx_boot.py $(args $M) --out dry --dry | tr '\n' ' '; echo; done; exit; fi
if [ "$1" = A ]; then run yolo26l; run yolo11l; echo F2BOOT_A_DONE
else while ! grep -q BOOT_EXSQNRFX_DONE logs/fx_boot_exdark.out; do sleep 60; done; run yolo26s; run yolo11s; echo F2BOOT_B_DONE; fi

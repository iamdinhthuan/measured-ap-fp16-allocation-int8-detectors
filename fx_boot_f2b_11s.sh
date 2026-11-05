#!/bin/bash
# README 11.8: YOLO11-s factorial re-bootstrap with the AP x r arm on its own sim_mean engine (identical regex to
# minimax at 3% and 10%; the f2 run picked the alphabetically first build). New ID, nothing overwritten.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
M=yolo11s; C=configs/fx_${M}_int8ent_ho
echo "== $(date +%F_%T) $M f2b"
$P fx_boot.py --model $M --out fx_${M}_f2b --arm mean=@mean --arm simmean=configs/simalloc_${M}_int8ent_ho.json:mean \
  --arm apq=${C}_apq_n2000.json:apq --arm sqnrq=${C}_noise_n2000.json:sqnrq --arm sqnrr=${C}_noise_n2000.json:sqnrr \
  --prefer simmean=sim_mean \
  --pairs sqnrq:apq,sqnrr:simmean,apq:simmean,sqnrq:sqnrr,apq:mean,sqnrq:mean,sqnrr:mean,simmean:mean 2>&1 | grep -v "^resample"
echo "== rc=${PIPESTATUS[0]}"; echo F2B_11S_DONE

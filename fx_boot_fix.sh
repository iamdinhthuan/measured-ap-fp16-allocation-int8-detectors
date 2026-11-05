#!/bin/bash
# README 11.8: one build per arm. YOLO26-l re-bootstrapped with each arm on its own-name engine when the
# allocation is identical to a published one (sqnr/recon tags); plus the missing Fisher YOLO11-s 3% pair.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
M=yolo26l; C=configs/fx_${M}_int8ent_ho
echo "== $(date +%F_%T) f1b"
$P fx_boot.py --model $M --out fx_${M}_f1b --arm mean=@mean --arm sqnrfx=${C}_noise_n256.json:sqnrq --arm reconfx=${C}_noise_n256.json:reconq \
  --prefer sqnrfx=sqnr --prefer sqnrfx=fx256_sqnrq --prefer reconfx=recon --prefer reconfx=fx256_reconq \
  --pairs sqnrfx:mean,reconfx:mean 2>&1 | grep -v "^resample"
echo "== $(date +%F_%T) f2b"
$P fx_boot.py --model $M --out fx_${M}_f2b --arm mean=@mean --arm simmean=configs/simalloc_${M}_int8ent_ho.json:mean \
  --arm apq=${C}_apq_n2000.json:apq --arm sqnrq=${C}_noise_n2000.json:sqnrq --arm sqnrr=${C}_noise_n2000.json:sqnrr \
  --prefer sqnrq=sqnr --prefer sqnrq=fx256_sqnrq --prefer apq=fx2k_apq --prefer sqnrr=fx2k_sqnrr --prefer simmean=sim_mean \
  --pairs sqnrq:apq,sqnrr:simmean,apq:simmean,sqnrq:sqnrr,apq:mean,sqnrq:mean,sqnrr:mean,simmean:mean 2>&1 | grep -v "^resample"
echo "== $(date +%F_%T) fisher3"
$P fx_boot.py --model yolo11s --out fx_yolo11s_fisher3 --arm mean=@mean --arm fisher=@fisher --budgets 0.03 --pairs fisher:mean 2>&1 | grep -v "^resample"
echo FX_BOOT_FIX_DONE

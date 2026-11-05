#!/bin/bash
# README 11.10 Y2 (rules_11_10.md, written before results): 10k bootstrap of sqnrfx:mean, published files.
# Usage: nohup ./fx_queue_y2.sh "yolo26l:f1b yolo11l:f1" A >> logs/fx_queue_y2_A.out 2>&1 < /dev/null &
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
for MT in $1; do
  M=${MT%%:*}; T=${MT##*:}
  [ -f metrics/bootstrap/fx_${M}_y2_10k.json ] && { echo "skip $M"; continue; }
  for try in 1 2 3; do
    while [ $(free -g | awk '/Mem:/{print $7}') -lt 15 ]; do echo "$(date +%T) low RAM, waiting"; sleep 300; done
    echo "== $(date +%F_%T) y2 $M ($T) try $try"
    $P y2_driver.py $M $T 2>&1 | grep -v "^resample"
    rc=${PIPESTATUS[0]}; echo "rc=$rc"
    [ $rc -eq 0 ] && [ -f metrics/bootstrap/fx_${M}_y2_10k.json ] && break
  done
  [ -f metrics/bootstrap/fx_${M}_y2_10k.json ] && sha256sum metrics/bootstrap/fx_${M}_y2_10k.json y2_driver.py >> logs/fx.sha256
done
echo FX_QUEUE_Y2_$2_DONE

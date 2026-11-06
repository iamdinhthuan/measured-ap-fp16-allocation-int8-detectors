#!/bin/bash
# Two CPU bootstrap chains (heavy-job cap 2). Resumable: every step skips existing outputs.
# chain 1: A COCO (fx_boot_a11.sh, 4 models) -> C yolo26s yolo11s; chain 2: A ExDark -> C yolo26l yolo11l -> B groups.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
if [ "$1" = 1 ]; then
  while kill -0 1492190 2>/dev/null; do sleep 60; done
  ./fx_boot_a11.sh >> logs/fx_boot_a11.out 2>&1
  ./fx_boot_c11.sh "yolo26s yolo11s" >> logs/fx_boot_c11.out 2>&1
  echo CHAIN1_DONE >> logs/fx_boot_c11.out
else
  while kill -0 1492602 2>/dev/null; do sleep 60; done
  [ -f metrics/bootstrap/g_yolo11s_exdark_a11.json ] || ./fx_boot_a11ex.sh >> logs/fx_boot_a11ex.out 2>&1
  ./fx_boot_c11.sh "yolo26l yolo11l" >> logs/fx_boot_c11.out 2>&1
  if [ ! -f metrics/bootstrap/g_yolo11s_exdarkcal_b11.json ]; then
    CMD=$($P b11_bootcmd.py | grep -v "^#")
    echo "== $(date +%F_%T) b11 boot" >> logs/fx_boot_b11.out
    eval "$CMD" >> logs/fx_boot_b11.out 2>&1; echo "== rc=$?" >> logs/fx_boot_b11.out
    [ -f metrics/bootstrap/g_yolo11s_exdarkcal_b11.json ] && sha256sum metrics/bootstrap/g_yolo11s_exdarkcal_b11.json logs/b11_boot_alias.json >> logs/fx.sha256
  fi
  echo CHAIN2_DONE >> logs/fx_boot_b11.out
fi

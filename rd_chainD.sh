#!/bin/bash
# rules_11_12d_add2.md A5, after chain C (at most two heavy jobs).
cd ~/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
until grep -qE "CHAIN_C_DONE|FAILED|did not finish" logs/rd_chainC.log 2>/dev/null; do sleep 60; done
$P rd_det.py && sha256sum metrics/det_rtdetr-l.json >> logs/fx.sha256 && echo CHAIN_D_DONE

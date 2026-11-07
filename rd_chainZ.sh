#!/bin/bash
# Final step of D: bootstrap + C1/C2/C3 reading (rd_boot.py, written before any sweep/val number was read).
cd ~/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
for c in C F; do
  until grep -qE "CHAIN_${c}_DONE|FAILED|did not finish|gate stopped" logs/rd_chain$c.log 2>/dev/null; do sleep 120; done
  grep -q "CHAIN_${c}_DONE" logs/rd_chain$c.log || echo "WARN chain $c incomplete; affected arms are skipped or the script stops"
done
$P rd_boot.py && sha256sum metrics/bootstrap/rd_rtdetr-l.json metrics/rd_rtdetr-l_verdict.json >> logs/fx.sha256 && echo CHAIN_Z_DONE

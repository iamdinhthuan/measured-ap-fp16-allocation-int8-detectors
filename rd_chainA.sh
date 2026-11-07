#!/bin/bash
# rules_11_12d.md arms 3-5 (onnxruntime, then their engines). Starts once the sweep has written the block list.
cd ~/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
stop() { grep -qE "GATE_STOP|GATE_ERROR|QUANT_FAILED" logs/rd_chain.log 2>/dev/null; }
until [ -e metrics/sens_rtdetr-l_int8ent.json ]; do stop && { echo "gate stopped"; exit 0; }; sleep 60; done
F=configs/fisher_rtdetr-l_int8ent_ho.json
[ -e $F ] || $P -c "
import json, rd_common
json.dump({'force_regex': rd_common.FORCE_RX, 'note': 'forced set F only (rules_11_12d.md); no Fisher score was computed for RT-DETR-l. Read by fx_factorial.py for F.'}, open('$F', 'w'), indent=1)"
run() { echo "$(date -Is) $*"; "$@" || { echo "FAILED: $*"; exit 1; }; }
[ -e configs/fx_rtdetr-l_int8ent_ho_noise_n2000.json ] || run $P rd_wrap.py fx_factorial.py --model rtdetr-l --mode noise --n-img 2000 --no-recon --tag _n2000
[ -e configs/fx_rtdetr-l_int8ent_ho_hyb.json ] || run $P fx_hybrid_rd.py
[ -e configs/fx_rtdetr-l_int8ent_ho_noise_n256.json ] || run $P rd_wrap.py fx_factorial.py --model rtdetr-l --mode noise --n-img 256 --no-recon --tag _n256
[ -e configs/fx_rtdetr-l_int8ent_ho_pandey_n256.json ] || run $P rd_wrap.py fx_pandey.py --model rtdetr-l --n-img 256 --tag _n256
sha256sum configs/*rtdetr-l* metrics/fx_scores_rtdetr-l_* >> logs/fx.sha256
run $P rd_queue.py proxies
echo CHAIN_A_DONE

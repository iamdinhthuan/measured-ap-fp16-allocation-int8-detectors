#!/bin/bash
# Rerun of the queue_f Fisher/auto_quantize steps for the confirmatory models, which failed with CUDA OOM
# (another job held 15 GB; the script kept all 2000 tuning images on the device). autoquant_select.py now keeps
# batches on the host (backup: logs/autoquant_select.py.bak_20261026). A yolo11s n512 run first checks that the
# patched script reproduces the scores of the original run. Then queue_i (recon/SQNR, needs the Fisher config).
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
run $P autoquant_select.py --model yolo11s --base int8ent --bits 8.02 --n-img 512 --out-tag _verify_hostbatch
$P - <<"PY"
import json
a=json.load(open("configs/autoquant_yolo11s_int8ent.json"))["fisher_scores"]
b=json.load(open("configs/autoquant_yolo11s_int8ent_verify_hostbatch.json"))["fisher_scores"]
import numpy as np
k=sorted(a); x=np.array([a[i] for i in k]); y=np.array([b[i] for i in k])
print("VERIFY max_rel_diff=%.3g corr=%.6f rank_equal=%s" % (np.max(np.abs(x-y)/np.maximum(np.abs(x),1e-30)), np.corrcoef(x,y)[0,1], list(np.argsort(x))==list(np.argsort(y))))
PY
for m in yolo26l yolo11l; do
  case $m in
    yolo26*) RX="/model\.\d+/((?:one2one_)?cv2\.\d/(?:one2one_)?cv2\.\d\.2|(?:one2one_)?cv3\.\d/(?:one2one_)?cv3\.\d\.2|dfl)/";;
    *)       RX="/model\.\d+/(cv2\.\d/cv2\.\d\.2|cv3\.\d/cv3\.\d\.2|dfl)/";;
  esac
  run $P autoquant_select.py --model $m --base int8ent --bits 8.02,8.04,8.08,8.15,8.3 --n-img 2000 --mac-budgets 0.03,0.05,0.1,0.15,0.2 --force-regex "$RX" --force-tag ho --out-tag _n2000
  run $P eval_alloc.py --alloc configs/fisher_${m}_int8ent_ho_n2000.json --keys fisher@0.03,fisher@0.05,fisher@0.1,fisher@0.15,fisher@0.2
  run $P eval_alloc.py --alloc configs/autoquant_${m}_int8ent_n2000.json --keys autoquant@8.02,autoquant@8.04,autoquant@8.08,autoquant@8.15,autoquant@8.3 --splits val
done
echo QUEUE_J_FISHER_DONE
./queue_i.sh > logs/queue_i.out 2>&1
echo QUEUE_J_DONE

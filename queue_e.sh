#!/bin/bash
# Confirmatory queue (protocol frozen in README section 10, 2026-10-25 22:00):
# yolo26s baselines, matched-MAC mean, ExDark lighting groups, Fisher n2000, random selection
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
RX11=$($P -c "import json;print(json.load(open(\"configs/manual_yolo11s_int8ent.json\"))[\"allocations\"][\"headout\"][\"regex\"])")
RX26=$($P -c "import json;print(json.load(open(\"configs/manual_yolo26s_int8ent.json\"))[\"allocations\"][\"headout\"][\"regex\"])")
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
{
run $P eval_alloc.py --alloc configs/autoquant_yolo26s_int8ent.json --keys autoquant@8.02,autoquant@8.04,autoquant@8.08,autoquant@8.15,autoquant@8.3
run $P eval_alloc.py --alloc configs/fisher_yolo26s_int8ent_ho.json --keys fisher@0.03,fisher@0.05,fisher@0.1,fisher@0.15,fisher@0.2
run $P eval_alloc.py --alloc metrics/alloc_yolo26s_int8ent_ho_v2.json --keys mean_matched@0.05
run $P eval_alloc.py --alloc metrics/alloc_yolo11s_int8ent_ho_v2.json --keys mean_matched@0.2
run $P eval_alloc.py --alloc configs/manual_yolo11s_int8ent_exdark.json --keys base,fp16_all,headout,stem_headout --splits exdark_test --prefix exdark_
run $P eval_alloc.py --alloc metrics/alloc_yolo11s_int8ent_exdark_ho.json --keys mean@0.03,minimax@0.03,mean@0.05,mean@0.1,minimax@0.1,mean@0.15,minimax@0.15,mean@0.2,minimax@0.2 --splits exdark_test --prefix exdark_
run $P autoquant_select.py --model yolo11s --base int8ent --bits 8.04 --n-img 2000 --mac-budgets 0.05,0.1,0.15,0.2 --force-regex "$RX11" --force-tag ho --out-tag _n2000
run $P eval_alloc.py --alloc configs/fisher_yolo11s_int8ent_ho_n2000.json --keys fisher@0.05,fisher@0.1,fisher@0.15,fisher@0.2 --splits val --prefix n2000_
run $P autoquant_select.py --model yolo26s --base int8ent --bits 8.04 --n-img 2000 --mac-budgets 0.05,0.1,0.15,0.2 --force-regex "$RX26" --force-tag ho --out-tag _n2000
run $P eval_alloc.py --alloc configs/fisher_yolo26s_int8ent_ho_n2000.json --keys fisher@0.05,fisher@0.1,fisher@0.15,fisher@0.2 --splits val --prefix n2000_
for m in yolo26s yolo11s; do
  K=$($P -c "print(\",\".join(f\"random@{b}_s{s}\" for b in (0.05,0.1,0.2) for s in range(1,6)))")
  run $P eval_alloc.py --alloc metrics/random_${m}_int8ent_ho.json --keys $K --splits val
done
echo QUEUE_E_DONE
} > logs/queue_e.out 2>&1

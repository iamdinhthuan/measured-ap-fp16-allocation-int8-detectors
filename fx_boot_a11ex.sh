#!/bin/bash
# rules_11_11.md A, ExDark: only pandey@0.1 differs from the pooled selection (other budgets identical, difference 0).
cd /home/thuan/isci_method_20261025
nice -n 10 /home/thuan/miniconda3/envs/qtsd/bin/python bootstrap_groups.py --gt data/exdark/exdark_test.json --sets fp32=metrics/exdark/yolo11s_fp32_test.pred.json.gz,mean_B0.1=metrics/alloc_eval/yolo11s_int8ent/exdark_mean_B0.1_exdark_test.pred.json.gz,minimax_B0.1=metrics/alloc_eval/yolo11s_int8ent/exdark_minimax_B0.1_exdark_test.pred.json.gz,sqnr_B0.1=metrics/alloc_eval/yolo11s_int8ent/fxex_sqnrq_B0.1_exdark_test.pred.json.gz,pandey_B0.1=metrics/alloc_eval/yolo11s_int8ent/exdark_fxp_pandey_B0.1_exdark_test.pred.json.gz --ref fp32 --pairs pandey_B0.1:mean_B0.1,pandey_B0.1:minimax_B0.1,sqnr_B0.1:pandey_B0.1 --groups attr:light --group-names Ambient,Strong,Single,Weak,Twilight,Object,Low,Window --n 1000 --out metrics/bootstrap/g_yolo11s_exdark_a11.json
rc=$?; echo rc=$rc; [ -f metrics/bootstrap/g_yolo11s_exdark_a11.json ] && sha256sum metrics/bootstrap/g_yolo11s_exdark_a11.json >> logs/fx.sha256
echo BOOT_A11EX_DONE

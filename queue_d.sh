#!/bin/bash
# yolo26s: measure mean vs minimax allocations (head-out forced; minimax@0.2 == mean@0.2, not rebuilt), then auto_quantize baseline
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
exec flock sens.lock bash -c "$P eval_alloc.py --alloc metrics/alloc_yolo26s_int8ent_ho.json --keys mean@0.1,minimax@0.1,mean@0.05,minimax@0.05,mean@0.15,minimax@0.15,mean@0.2,mean@0.03,minimax@0.03 > logs/alloc_eval_yolo26s_ho.out 2>&1; $P autoquant_select.py --model yolo26s --base int8ent --bits 8.02,8.04,8.08,8.15,8.3 --mac-budgets 0.03,0.05,0.1,0.15,0.2 --force-regex \"/model\.\d+/((?:one2one_)?cv2\.\d/(?:one2one_)?cv2\.\d\.2|(?:one2one_)?cv3\.\d/(?:one2one_)?cv3\.\d\.2|dfl)/\" --force-tag ho > logs/autoquant_yolo26s.out 2>&1" < /dev/null

#!/bin/bash
# usage: launch_aq.sh MODEL BITS [extra autoquant_select.py args]
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
exec flock aq.lock $P autoquant_select.py --model "$1" --base int8ent --bits "$2" "${@:3}" > "logs/autoquant_$1.out" 2>&1 < /dev/null

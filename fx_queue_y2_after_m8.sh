#!/bin/bash
# Runs fx_queue_y2.sh "$1" "$2" after the M8 queue has finished (2-heavy-job limit). Relaunch after the 04-11 reboot.
cd /home/thuan/isci_method_20261025
until grep -qE "FX_QUEUE_M8_(DONE|FAILED)" logs/fx_queue_m8b.out 2>/dev/null; do sleep 600; done
exec ./fx_queue_y2.sh "$1" "$2"

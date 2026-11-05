#!/bin/bash
# waits for queue_k1 (greedy) then runs queue_k3 (best-of-33), so at most two GPU jobs share the 5090
cd /home/thuan/isci_method_20261025
while pgrep -f queue_k1.sh >/dev/null; do sleep 30; done
./queue_k3.sh > logs/queue_k3.out 2>&1

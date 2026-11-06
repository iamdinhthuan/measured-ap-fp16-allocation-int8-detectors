#!/bin/bash
# README 11.11 A (rules_11_11.md): per-image (Pandey Eq. 3) output SQNR. Stage 1 scores + sanity gate, stage 2 engines.
# Usage: ./fx_queue_p.sh 2>&1 | tee -a logs/fx_queue_p.out   (resumable: every stage skips existing outputs)
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
MODELS="yolo26l yolo11l yolo26s yolo11s"
B5="0.03 0.05 0.1 0.15 0.2"
gpu_wait() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$f" -ge 8000 ] && return
    echo "$(date +%T) gpu free ${f} MiB, waiting"; sleep 300
  done
}
gate_ok() { $P -c "import json,sys; sys.exit(0 if json.load(open('$1'))['sanity_gate']['pass'] else 1)"; }
for M in $MODELS; do
  C=configs/fx_${M}_int8ent_ho_pandey_n256.json
  [ -f $C ] || { gpu_wait; echo "== $(date +%F_%T) pandey $M"; $P fx_pandey.py --model $M --n-img 256 --tag _n256 2>&1 | grep -v "W:onnxruntime"; }
  [ -f $C ] && sha256sum $C metrics/fx_scores_${M}_int8ent_pandey_n256.json >> logs/fx.sha256
done
C=configs/fx_yolo11s_int8ent_ho_pandey_exdark_n256.json
[ -f $C ] || { gpu_wait; echo "== $(date +%F_%T) pandey yolo11s exdark"; $P fx_pandey.py --model yolo11s --n-img 256 --tag _exdark_n256 --split exdark 2>&1 | grep -v "W:onnxruntime"; }
[ -f $C ] && sha256sum $C metrics/fx_scores_yolo11s_int8ent_pandey_exdark_n256.json >> logs/fx.sha256
for M in $MODELS; do
  C=configs/fx_${M}_int8ent_ho_pandey_n256.json
  gate_ok $C || { echo "GATE FAIL $M: no engines (rules_11_11 A)"; continue; }
  keys=""; for b in $B5; do keys="$keys,pandey@$b"; done
  $P fx_dedup.py --alloc $C --keys ${keys#,} | tee -a logs/fx_dedup_${M}.txt > /tmp/fx_pd_$$.txt
  todo=""; while read k st rest; do t=fxp_${k/@/_B}; [ "$st" = eval ] && [ ! -f metrics/alloc_eval/${M}_int8ent/$t.json ] && todo="$todo,$k"; done < /tmp/fx_pd_$$.txt
  rm -f /tmp/fx_pd_$$.txt
  [ -z "$todo" ] && { echo "nothing to eval $M"; continue; }
  gpu_wait; echo "== $(date +%F_%T) eval $M ${todo#,}"
  $P eval_alloc.py --alloc $C --keys ${todo#,} --splits val --prefix fxp_ 2>&1 | grep -v "W:onnxruntime"
done
C=configs/fx_yolo11s_int8ent_ho_pandey_exdark_n256.json
if gate_ok $C; then
  todo=$($P -c "
import json; c=json.load(open('$C')); r=json.load(open('configs/fx_yolo11s_int8ent_ho_noise_exdark_n256.json'))
print(','.join(k for k in c['allocations'] if k.startswith('pandey@') and sorted(c['allocations'][k]['blocks'])!=sorted(r['allocations']['sqnrq@'+k.split('@')[1]]['blocks'])))")
  echo "exdark keys differing from pooled: ${todo:-none}"
  todo2=""; for k in ${todo//,/ }; do [ -f metrics/alloc_eval/yolo11s_int8ent/exdark_fxp_${k/@/_B}.json ] || todo2="$todo2,$k"; done
  [ -n "$todo2" ] && { gpu_wait; $P eval_alloc.py --alloc $C --keys ${todo2#,} --splits exdark_test --prefix exdark_fxp_ 2>&1 | grep -v "W:onnxruntime"; }
else echo "GATE FAIL exdark"; fi
echo FX_QUEUE_P_DONE

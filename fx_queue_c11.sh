#!/bin/bash
# rules_11_11c.md (C): FIT score + engines. Waits for fx_queue_p.sh (A) to finish (heavy-job cap). Resumable.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
MODELS="yolo26l yolo11l yolo26s yolo11s"
FAIL=0
run() { echo "== $(date +%F_%T) $*"; "$@"; local rc=$?; echo "== rc=$rc"; [ $rc -eq 0 ] || FAIL=1; return $rc; }
gate_ok() { $P -c "import json,sys; sys.exit(0 if json.load(open('$1'))['sanity_gate']['pass'] else 1)"; }
{
echo "== C11 start $(date +%F_%T)"
while pgrep -u thuan -x fx_queue_p.sh >/dev/null || kill -0 1459494 2>/dev/null; do sleep 60; done
for M in $MODELS; do
  C=configs/fx_${M}_int8ent_ho_fit.json
  if [ ! -f $C ]; then
    [ -f metrics/fit_scores_${M}_int8ent.json ] && mv metrics/fit_scores_${M}_int8ent.json metrics/fit_scores_${M}_int8ent_partial.json
    run $P fit_score.py --model $M
  fi
  [ -f $C ] && sha256sum $C metrics/fit_scores_${M}_int8ent.json >> logs/fx.sha256
done
for M in $MODELS; do
  C=configs/fx_${M}_int8ent_ho_fit.json
  [ -f $C ] || { echo "MISSING $C"; FAIL=1; continue; }
  gate_ok $C || { echo "GATE FAIL $M: no engines (rules_11_11c)"; FAIL=1; continue; }
  $P fx_dedup.py --alloc $C --keys fit@0.03,fit@0.05,fit@0.1,fit@0.15,fit@0.2 | tee -a logs/fx_dedup_${M}.txt > logs/fx_dedup_fit_${M}.tmp
  for k in $(awk '$2=="eval"{print $1}' logs/fx_dedup_fit_${M}.tmp); do
    t=fxf_${k/@/_B}
    [ -f metrics/alloc_eval/${M}_int8ent/$t.json ] || run $P eval_alloc.py --alloc $C --keys $k --splits val --prefix fxf_
  done
  for f in metrics/alloc_eval/${M}_int8ent/fxf_*.json; do [ -f "$f" ] && sha256sum "$f" >> logs/fx.sha256; done
done
echo "fail=$FAIL"
[ $FAIL -eq 0 ] && echo FX_QUEUE_C11_DONE || echo FX_QUEUE_C11_INCOMPLETE
} >> logs/fx_queue_c11.out 2>&1

#!/bin/bash
# rules_11_11b.md (B), v2 of fx_queue_b11.sh: run() propagates rc; allocation gated on a complete sweep;
# SQNR gated on A's YOLO11-s ExDark sanity gate; DONE only with every expected output and no failed step.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
M=yolo11s; BASE=int8ent_exdarkcal
RX11=$($P -c "import json;print(json.load(open(\"configs/manual_yolo11s_int8ent.json\"))[\"allocations\"][\"headout\"][\"regex\"])")
SENS=metrics/sens_${M}_${BASE}_exdark.json
ALLOC=metrics/alloc_${M}_${BASE}_exdark_ho.json
SC=metrics/fx_scores_${M}_${BASE}_sqnr_exdark_n256.json
SQ=configs/fx_${M}_${BASE}_ho_sqnr_exdark_n256.json
MAN=configs/manual_${M}_${BASE}_exdark.json
AGATE=configs/fx_${M}_int8ent_ho_pandey_exdark_n256.json
OUT=metrics/alloc_eval/${M}_${BASE}
FAIL=0
run() { echo "== $(date +%F_%T) $*"; "$@"; local rc=$?; echo "== rc=$rc"; [ $rc -eq 0 ] || FAIL=1; return $rc; }
log_sha() { for f in "$@"; do [ -f "$f" ] && sha256sum "$f" >> logs/fx.sha256; done; }
partial() {  # rows written by infer_eval but not completed by block_sensitivity (no latency_ms)
  for f in metrics/tune_exdark/$M/${BASE}*.json; do
    [ -f "$f" ] || continue
    $P -c "import json,sys;sys.exit(0 if 'latency_ms' in json.load(open('$f')) else 1)" 2>/dev/null || { echo "partial $f"; mv "$f" "${f%.json}_partial.json"; }
  done
}
complete() { $P -c "
import json,sys
s=json.load(open('$SENS'));b=json.load(open('metrics/sens_${M}_int8ent.json'))['blocks']
ok=s['blocks']==b and all(k in s['rows'] for k in list(b)+['$BASE','fp16_all'])
print('sens complete' if ok else 'sens INCOMPLETE');sys.exit(0 if ok else 1)"; }
evalk() {
  local tag="exdarkcal_${2/@/_B}"
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; else run $P eval_alloc.py --alloc "$1" --keys "$2" --splits exdark_test --prefix exdarkcal_; fi
}
{
echo "== B11b start $(date +%F_%T)"
while kill -0 1467283 2>/dev/null; do sleep 30; done   # sweep started by fx_queue_b11.sh
partial
complete || run $P block_sensitivity.py --model $M --base $BASE --split exdark
complete || { echo "STOP: sweep incomplete"; exit 1; }
[ -f $ALLOC ] || run $P allocate.py --sens $SENS --onnx exports/${M}.onnx --budgets 0.03,0.05,0.1,0.15,0.2 --groups attr:light \
   --group-names Ambient,Strong,Single,Weak,Twilight,Object,Low,Window --force-regex "$RX11" --out $ALLOC
[ -f $MAN ] || $P - <<PY
import json
src = json.load(open("configs/manual_${M}_int8ent_exdark.json"))
json.dump({"model": "$M", "base": "$BASE", "split": "exdark",
           "note": "rules_11_11b B: stem baseline on the ExDark-calibrated graph",
           "allocations": {"stem_headout": src["allocations"]["stem_headout"]}}, open("$MAN", "w"), indent=1)
PY
evalk $MAN stem_headout
if [ -f $ALLOC ]; then for k in $($P -c "import json;print(' '.join(json.load(open('$ALLOC'))['allocations']))"); do evalk $ALLOC $k; done; fi
while [ ! -f $AGATE ]; do sleep 60; done
if $P -c "import json,sys;sys.exit(0 if json.load(open('$AGATE'))['sanity_gate']['pass'] else 1)"; then
  if [ ! -f $SQ ]; then [ -f $SC ] && mv $SC ${SC%.json}_partial.json; run $P exdarkcal_sqnr.py; fi
  if [ -f $SQ ]; then for k in $($P -c "import json;print(' '.join(json.load(open('$SQ'))['allocations']))"); do evalk $SQ $k; done; fi
else echo "A ExDark gate FAILED: SQNR arm skipped"; FAIL=1; fi
log_sha $SENS $SC $SQ $ALLOC $MAN $OUT/exdarkcal_*.json
N=$(ls $OUT/exdarkcal_*.json 2>/dev/null | grep -vc pred)
EXP=$($P -c "import json;print(1+len(json.load(open('$ALLOC'))['allocations'])+len(json.load(open('$SQ'))['allocations']))" 2>/dev/null || echo -1)
echo "outputs $N expected $EXP fail=$FAIL"
[ "$N" = "$EXP" ] && [ $FAIL -eq 0 ] && echo FX_QUEUE_B11_DONE || echo FX_QUEUE_B11_INCOMPLETE
} >> logs/fx_queue_b11.out 2>&1

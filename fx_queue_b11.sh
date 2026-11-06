#!/bin/bash
# rules_11_11b.md (B): allocation on the ExDark-calibrated YOLO11-s graph. Resumable: every step skips existing outputs.
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
M=yolo11s; BASE=int8ent_exdarkcal
RX11=$($P -c "import json;print(json.load(open(\"configs/manual_yolo11s_int8ent.json\"))[\"allocations\"][\"headout\"][\"regex\"])")
SENS=metrics/sens_${M}_${BASE}_exdark.json
ALLOC=metrics/alloc_${M}_${BASE}_exdark_ho.json
SQ=configs/fx_${M}_${BASE}_ho_sqnr_exdark_n256.json
MAN=configs/manual_${M}_${BASE}_exdark.json
OUT=metrics/alloc_eval/${M}_${BASE}
run() { echo "== $(date +%F_%T) $*"; "$@"; echo "== rc=$?"; }
log_sha() { for f in "$@"; do [ -f "$f" ] && sha256sum "$f" >> logs/fx.sha256; done; }
evalk() {  # plan key
  local tag="exdarkcal_${2/@/_B}"
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; else run $P eval_alloc.py --alloc "$1" --keys "$2" --splits exdark_test --prefix exdarkcal_; fi
}
{
echo "== B11 start $(date +%F_%T)"
grep -q SENS_DONE logs/fx_queue_b11_sens.done 2>/dev/null || { run $P block_sensitivity.py --model $M --base $BASE --split exdark && echo SENS_DONE > logs/fx_queue_b11_sens.done; }
grep -q SENS_DONE logs/fx_queue_b11_sens.done || { echo "SENS FAILED, stop"; exit 1; }
$P -c "import json,sys;a=json.load(open('$SENS'))['blocks'];b=json.load(open('metrics/sens_${M}_int8ent.json'))['blocks'];sys.exit(0 if a==b else 1)" || { echo "BLOCKS DIFFER, stop"; exit 1; }
echo "blocks identical"
[ -f $SQ ] || run $P exdarkcal_sqnr.py
[ -f $ALLOC ] || run $P allocate.py --sens $SENS --onnx exports/${M}.onnx --budgets 0.03,0.05,0.1,0.15,0.2 --groups attr:light \
   --group-names Ambient,Strong,Single,Weak,Twilight,Object,Low,Window --force-regex "$RX11" --out $ALLOC
[ -f $MAN ] || $P - <<PY
import json
src = json.load(open("configs/manual_${M}_int8ent_exdark.json"))
json.dump({"model": "$M", "base": "$BASE", "split": "exdark",
           "note": "rules_11_11b B: stem baseline on the ExDark-calibrated graph",
           "allocations": {"stem_headout": src["allocations"]["stem_headout"]}}, open("$MAN", "w"), indent=1)
PY
log_sha $SENS $SQ $ALLOC $MAN metrics/fx_scores_${M}_${BASE}_sqnr_exdark_n256.json
evalk $MAN stem_headout
for k in $($P -c "import json;print(' '.join(json.load(open('$ALLOC'))['allocations']))"); do evalk $ALLOC $k; done
for k in $($P -c "import json;print(' '.join(json.load(open('$SQ'))['allocations']))"); do evalk $SQ $k; done
log_sha $OUT/exdarkcal_*.json
echo FX_QUEUE_B11_DONE
} >> logs/fx_queue_b11.out 2>&1

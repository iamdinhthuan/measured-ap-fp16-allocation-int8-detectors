#!/usr/bin/env bash
# Latency of measured allocations ON the Jetson (runs on-device, TensorRT 10.3).
# Every <tag>.onnx in DIR is an hp32 Q/DQ graph (orin_export.py); it is built with
# --int8 --fp16, so Q/DQ-wrapped layers run INT8 and restored layers FP16.
# fp16.onnx (the FP32 export) is built with --fp16 only, as the all-FP16 reference.
# All engines are built first; then REPS rounds each time every engine once (rounds
# interleave engines so thermal drift does not line up with one configuration).
# Each engine is timed REPS times (trtexec GPU compute median), with VDD_IN from tegrastats.
# Output: one JSON line per engine and repetition, appended to DIR/bench.jsonl
#
# Usage (on the Jetson):  bash orin_bench_alloc.sh ~/isci_orin/yolo11s [REPS]
set -u
DIR=$1
REPS=${2:-3}
T=/usr/src/tensorrt/bin/trtexec
mkdir -p "$DIR/engines" "$DIR/logs"
OUT=$DIR/bench.jsonl
for O in "$DIR"/*.onnx; do
  NAME=$(basename "$O" .onnx)
  E=$DIR/engines/$NAME.plan
  if [ "$NAME" = fp16 ]; then FLAGS="--fp16"; else FLAGS="--int8 --fp16"; fi
  if [ ! -f "$E" ]; then
    "$T" --onnx="$O" $FLAGS --saveEngine="$E" --skipInference > "$DIR/logs/build_$NAME.log" 2>&1 \
      || { echo "{\"engine\":\"$NAME\",\"error\":\"build_failed\"}" >> "$OUT"; continue; }
  fi
  echo "built $NAME $(date +%T)"
done
for R in $(seq 1 "$REPS"); do
  for E in "$DIR"/engines/*.plan; do
    NAME=$(basename "$E" .plan)
    tegrastats --interval 200 > "$DIR/logs/tegra_${NAME}_$R.log" &
    TPID=$!
    sleep 2
    "$T" --loadEngine="$E" --warmUp=1000 --duration=20 --useSpinWait > "$DIR/logs/bench_${NAME}_$R.log" 2>&1
    RC=$?
    kill $TPID 2>/dev/null; wait $TPID 2>/dev/null
    [ $RC -ne 0 ] && { echo "{\"engine\":\"$NAME\",\"rep\":$R,\"error\":\"trtexec_failed\"}" >> "$OUT"; continue; }
    STATS=$(grep "GPU Compute Time: min" "$DIR/logs/bench_${NAME}_$R.log" | tail -1)
    MEDIAN=$(echo "$STATS" | grep -oE "median = [0-9.]+" | grep -oE "[0-9.]+")
    MEAN=$(echo "$STATS" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+")
    P99=$(echo "$STATS" | grep -oE "percentile\(99%\) = [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
    PW=$(awk 'NR>10 {for(i=1;i<=NF;i++) if($i=="VDD_IN"){split($(i+1),a,"/"); gsub("mW","",a[1]); s+=a[1]; n++}} END{if(n>0) printf "%.0f", s/n; else print "-1"}' "$DIR/logs/tegra_${NAME}_$R.log")
    echo "{\"engine\":\"$NAME\",\"rep\":$R,\"median_ms\":${MEDIAN:-null},\"mean_ms\":${MEAN:-null},\"p99_ms\":${P99:-null},\"vdd_in_mw\":${PW:-null},\"engine_bytes\":$(stat -c %s "$E")}" >> "$OUT"
  done
  echo "round $R done $(date +%T)"
done
echo ORIN_BENCH_DONE

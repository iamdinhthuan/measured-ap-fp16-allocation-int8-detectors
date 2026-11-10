#!/usr/bin/env bash
# JHF copy of orin_bench_gated.sh (power mode read from /var/lib/nvpmodel/status).
# Gated variant of orin_bench_alloc.sh: same trtexec flags and timing, but every build and
# every timing waits until no foreign trtexec is seen at 3 consecutive checks 10 s apart (60 s after a sighting), and a timing
# starts only after GR3D load has stayed <=20% at 3 samples 30 s apart (the idle desktop reads 0-7%), and a timing during which a foreign trtexec appears is redone.
# Records the power mode in every JSON line. Never changes clocks or power mode.
# Usage: bash orin_bench_gated.sh DIR [REPS]
set -u
DIR=$1
REPS=${2:-3}
T=/usr/src/tensorrt/bin/trtexec
mkdir -p "$DIR/engines" "$DIR/logs"
OUT=${OUT:-$DIR/bench.jsonl}
PM=$(cat /var/lib/nvpmodel/status 2>/dev/null)  # read-only; never runs nvpmodel
wait_idle() { local n=0; while [ $n -lt 3 ]; do if pgrep -x trtexec > /dev/null; then n=0; sleep 60; else n=$((n+1)); [ $n -lt 3 ] && sleep 10; fi; done; }
gr3d_idle() { local g; g=$(timeout 3 tegrastats --interval 500 | head -2 | tail -1 | grep -oE "GR3D_FREQ [0-9]+%" | grep -oE "[0-9]+" | sort -n | tail -1); [ "${g:-100}" -le 20 ]; }
# wait (without limit) until GR3D is idle at 3 consecutive samples 30 s apart: the desktop can hold the GPU for hours
wait_gpu() { local n=0; while [ $n -lt 3 ]; do if gr3d_idle; then n=$((n+1)); [ $n -lt 3 ] && sleep 30; else n=0; sleep 120; fi; done; }
for O in "$DIR"/*.onnx; do
  NAME=$(basename "$O" .onnx)
  E=$DIR/engines/$NAME.plan
  if [ "$NAME" = fp16 ]; then FLAGS="--fp16"; else FLAGS="--int8 --fp16"; fi
  if [ ! -f "$E" ]; then
    wait_idle
    "$T" --onnx="$O" $FLAGS --saveEngine="$E" --skipInference > "$DIR/logs/build_$NAME.log" 2>&1 \
      || { echo "{\"engine\":\"$NAME\",\"error\":\"build_failed\"}" >> "$OUT"; rm -f "$E"; continue; }
  fi
  echo "built $NAME $(date +%T)"
done
for R in $(seq 1 "$REPS"); do
  for E in "$DIR"/engines/*.plan; do
    NAME=$(basename "$E" .plan)
    FOREIGN=0; RC=1
    for TRY in 1 2 3 4 5 6; do
      wait_idle
      wait_gpu; wait_idle
      tegrastats --interval 200 > "$DIR/logs/tegra_${NAME}_$R.log" &
      TPID=$!
      sleep 2
      "$T" --loadEngine="$E" --warmUp=1000 --duration=20 --useSpinWait > "$DIR/logs/bench_${NAME}_$R.log" 2>&1 &
      BPID=$!
      FOREIGN=0
      while kill -0 $BPID 2>/dev/null; do
        [ "$(pgrep -x trtexec | grep -vc "^$BPID$")" -gt 0 ] && FOREIGN=1
        sleep 1
      done
      wait $BPID; RC=$?
      kill $TPID 2>/dev/null; wait $TPID 2>/dev/null
      [ $FOREIGN -eq 1 ] && { echo "foreign trtexec during $NAME r$R try $TRY, redo"; continue; }
      break
    done
    [ $FOREIGN -eq 1 ] && { echo "{\"engine\":\"$NAME\",\"rep\":$R,\"error\":\"foreign_load\"}" >> "$OUT"; continue; }
    [ $RC -ne 0 ] && { echo "{\"engine\":\"$NAME\",\"rep\":$R,\"error\":\"trtexec_failed\"}" >> "$OUT"; continue; }
    STATS=$(grep "GPU Compute Time: min" "$DIR/logs/bench_${NAME}_$R.log" | tail -1)
    MEDIAN=$(echo "$STATS" | grep -oE "median = [0-9.]+" | grep -oE "[0-9.]+")
    MEAN=$(echo "$STATS" | grep -oE "mean = [0-9.]+" | grep -oE "[0-9.]+")
    P99=$(echo "$STATS" | grep -oE "percentile\(99%\) = [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
    PW=$(awk 'NR>10 {for(i=1;i<=NF;i++) if($i=="VDD_IN"){split($(i+1),a,"/"); gsub("mW","",a[1]); s+=a[1]; n++}} END{if(n>0) printf "%.0f", s/n; else print "-1"}' "$DIR/logs/tegra_${NAME}_$R.log")
    echo "{\"engine\":\"$NAME\",\"rep\":$R,\"median_ms\":${MEDIAN:-null},\"mean_ms\":${MEAN:-null},\"p99_ms\":${P99:-null},\"vdd_in_mw\":${PW:-null},\"engine_bytes\":$(stat -c %s "$E"),\"power_mode\":\"$PM\"}" >> "$OUT"
  done
  echo "round $R done $(date +%T)"
done
echo ORIN_BENCH_DONE

#!/bin/bash
# Confirmatory COCO leg: minimax vs cost-matched mean on the large models (pairs resolved to the evaluated
# configuration with the identical block set; see make_tables.resolve), plus heuristics. Runs after boot_sqnr_r2
# (one bootstrap at a time: earlyoom killed concurrent ones on 26-10).
P=/home/thuan/miniconda3/envs/qtsd/bin/python
cd /home/thuan/isci_method_20261025
until grep -q BOOT_SQNR2_DONE logs/boot_sqnr_r2.out; do sleep 60; done
declare -A MM=( [yolo26l]="mean_B0.03 mean_B0.05 mean_B0.1 mean_matched_B0.15 mean_B0.2" [yolo11l]="mean_B0.03 mean_B0.05 mean_matched_B0.1 mean_B0.15 mean_matched_B0.2" )
for M in yolo26l yolo11l; do
  D=metrics/alloc_eval/${M}_int8ent
  S="fp32=$D/fp32_val.pred.json.gz,stem_headout=$D/stem_headout_val.pred.json.gz,head=$D/head_val.pred.json.gz"; PR=""
  i=0; BS=(0.03 0.05 0.1 0.15 0.2)
  for A in ${MM[$M]}; do B=${BS[$i]}; i=$((i+1))
    [[ $S == *"$A="* ]] || S="$S,$A=$D/${A}_val.pred.json.gz"
    S="$S,minimax_B$B=$D/minimax_B${B}_val.pred.json.gz"; PR="$PR,$A:minimax_B$B"
  done
  [[ $S == *"mean_B0.1="* ]] || S="$S,mean_B0.1=$D/mean_B0.1_val.pred.json.gz"
  PR="$PR,stem_headout:mean_B0.1,head:mean_B0.1"
  echo "== $(date +%F_%T) $M $PR"
  $P bootstrap_groups.py --gt data/coco_src/annotations/instances_val2017.json --sets $S --ref fp32 --pairs ${PR#,} --n 1000 --out metrics/bootstrap/g_${M}_coco_objective.json
  echo "== rc=$?"
done
echo BOOT_LARGE_DONE

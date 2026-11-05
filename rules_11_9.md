# Decision rules for README 11.9 experiments (written 2026-11-02, BEFORE any X1-X3 score, allocation or AP exists)

All runs are post-freeze and exploratory. New tags only; nothing is overwritten. Reference arm in every
pair is the published engine-measured mean allocation (`mean`, tag `mean_B<B>`). Paired image bootstrap on
val2017 (bootstrap_fast.py via fx_boot.py, n = 1000, seed as fx_boot), pair `X:mean` (difference mean - X;
positive favours the measured allocation). "Significant" = 95% percentile interval excludes 0. Holm counts
are reported within each family named below, from two-sided p = 2 min(p_le0, 1 - p_le0), floor 2/(n+1).

## X1. Branch-normalized output SQNR (baseline strengthening; reviewers: SQNR on the raw mixed output is handicapped)

- Tensor: the pre-decoding head tensor [1, 84, 8400] (rows 0-3 decoded box, rows 4-83 class sigmoid):
  YOLO11 `output0`; YOLO26 `/model.23/Concat_3_output_0` (the input of the top-k post-processing).
- Direction: only block b quantized in the FP32-lifted Q/DQ graph, reference = every Q/DQ bypassed (as published SQNR).
- Images: all 2000 tuning images (train2017, disjoint from calibration).
- Scores per block: primary `bnorm` = 0.5 (N_box/S_box + N_cls/S_cls); secondary `chnorm` = mean over the
  84 rows of N_c/S_c (N = summed squared error, S = summed squared reference, over images and anchors).
- Allocation: identical to fx_factorial.py (score / max score, same knapsack, forced set, block costs, budgets 3/5/10/15/20%).
- Family for the rule: the 10 large-detector pairs (YOLO11-l, YOLO26-l x 5 budgets) of `bnorm:mean`. Small detectors and `chnorm` are descriptive.
- Consequences:
  - (a) measured significantly better at >= 4 of 10: the claim "outperforms output SQNR on large detectors" stays; Results and Supplementary add the variant with its counts.
  - (b) measured significantly better at <= 1 of 10, or bnorm significantly better at >= 2 of 10: abstract, introduction, conclusion and highlights must name the baseline "raw output SQNR" and state that branch normalization closes (or reverses) the gap; the supplementary sentence "variants ... were not tested" is replaced by the result.
  - (c) otherwise: abstract/intro/conclusion name the baseline "raw output SQNR"; Results/Supplementary report bnorm descriptively with counts.

## X2. Block-aggregated Fisher (granularity control; reviewer: per-convolution choice set confounds the Fisher comparison)

- Score per block = sum of the published per-convolution Fisher scores (same autoquant score file as the published Fisher arm: 512 images for YOLO11-s/YOLO26-s, 2000 for YOLO11-l/YOLO26-l) over the convolutions matched by the block regex.
- Allocation: identical to fx_factorial.py (block granularity, same knapsack, forced set, costs, budgets).
- Family: all 20 detector-budget pairs of `fisherblk:mean`.
- Consequences:
  - (a) measured significantly better at >= 10 of 20: keep "15 of 20" for per-convolution Fisher and add the block-Fisher count.
  - (b) <= 5 of 20: every Fisher claim (Results, Conclusion) is scoped to "per-convolution Fisher" and the text states that block aggregation narrows the gap to the stated count.
  - (c) otherwise: report both counts side by side; Conclusion names both.

## X3. Selection stability over tuning draws (reviewers: interval conditions on one selected allocation)

- Two new 2000-image train2017 draws (seed 20261102; one shuffle, first 2000 = draw 1, next 2000 = draw 2) from images with >= 1 non-crowd annotation, excluding the 512 calibration images and the existing tune2000 images.
- Simulated AP x r sweep (FP32-lifted Q/DQ graph, onnxruntime CUDA, as sim_sensitivity.py), allocate.py mean objective with the published forced set and budgets. Draw 0 = existing `simalloc` (tune2000). Large detectors only.
- Reported (descriptive, no claim upgrade): Jaccard of selected block sets of draws 1/2 vs draw 0 per budget; val AP range over the three draws; pairs `sqnrfx:simd1`, `sqnrfx:simd2` (positive favours the draw allocation) and `simdK:mean`.
- Consequence: if at any large-detector pair where the published measured allocation is significantly better than raw SQNR a draw-1 or draw-2 allocation is below raw SQNR in point estimate, the main text must say that the advantage at that pair depends on the tuning draw. The text states that stability was measured for the simulated variant; deployed engine allocations come from one draw.

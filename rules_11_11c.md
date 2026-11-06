# rules_11_11c.md: experiment C (FIT score on the deployed quantizer), written 2026-11-06 before any FIT score exists

Status: exploratory, post-freeze, outside the confirmatory rule (as the other proxy arms).
Block-summed Fisher (fisherblk, rules_11_9.md X2) is already evaluated and reported in Supplementary S6
(11/20 significant, 9 after Holm); it is the existing primary block-level gradient comparison and is not re-run.
C adds exactly one new score: FIT (Zandonati et al.), adapted to the shared block knapsack. No HAWQ arm.

## Score (fixed now)
For every Conv l of the deployed graph exports/<m>_int8ent.onnx, with the fused FP torch model at its FP weights:
  FIT_l = sum_c (s_{w,l,c}^2 / 12) * sum_{k in channel c} E_i[g_{i,k}^2]  +  (s_{a,l}^2 / 12) * sum_j E_i[(dL_i/da_{i,l,j})^2]
- g_{i,k}: gradient of the per-image loss L_i w.r.t. weight element k; a_{i,l}: input tensor of conv l (gradient
  through this conv only: the input is cloned per conv in a forward pre-hook).
- E_i: mean over per-image gradients, batch size 1 (empirical Fisher trace = mean of squared per-sample gradients).
- s_{w,l,c}: per-channel scale of the weight DequantizeLinear of conv l; s_{a,l}: per-tensor scale of the
  DequantizeLinear feeding its input, both read from exports/<m>_int8ent.onnx (the executed quantizer).
- Noise model Delta^2/12 (as FIT), not the realised rounding error.
- Loss, images, preprocessing, targets and gradient path identical to the main-text Fisher arm (autoquant_select.py):
  Ultralytics training loss with tune2000 GT; YOLO26: undetached one-to-one path, one-to-one loss term (tal_topk=7,
  topk2=1); YOLO11: init_criterion(). Images: the first 512 (yolo11s, yolo26s) or 2000 (yolo11l, yolo26l) sorted tune
  images, as the score files behind the main-text Fisher arm.
- Block score = sum of FIT_l over the convs matched by the block regex (sens_<m>_int8ent.json); blocks without
  convolutions are excluded (as fisherblk). Score / max, recon_select.knapsack, forced set (main-text Fisher plan),
  block MAC costs, budgets 0.03, 0.05, 0.1, 0.15, 0.2. Keys fit@B in configs/fx_<m>_int8ent_ho_fit.json.

## Sanity gate (stop the model's arm if any fails)
1. Every Conv of exports/<m>.onnx maps to a fused torch Conv2d with identical weight shape, and every Conv of the
   int8ent graph has a weight DequantizeLinear and an input DequantizeLinear.
2. Fused torch weights equal the FP32 ONNX initializers: max |diff| <= 1e-4 * max|w| per conv.
3. Reported, not a failure: blocks whose count_qdq differs from 2 x (#convs) (non-conv quantized inputs such as
   attention MatMul or Concat); FIT under-scores them and the supplement states it.
4. No NaN/inf in any FIT_l; per-image loss finite for every image.

## Evaluation
fx_dedup.py on fit@B, then eval_alloc.py --splits val --prefix fxf_ for non-duplicates (identical regex reuses the
existing engine row). Bootstrap with fx_boot.py, 1000 resamples, seed as fx_boot default:
  pairs fit:mean (b - a = measured minus FIT) on all 20 detector x budget pairs; fisherblk:fit (descriptive).

## Consequences (fixed now)
- Counting unit: the 20 pairs; report uncorrected and Holm (within the 20) counts and the AP margin range.
- FIT is reported in Supplementary S6 next to fisherblk; at most one line-neutral sentence in Results.
- If the measured allocation is significantly more accurate at >= 10 of 20 pairs: the conclusion's proxy sentence
  may add FIT with its count.
- If at <= 5 of 20: Discussion/limitations states that the FIT-type score (activation noise included) closes most of
  the gap, and conclusion/abstract statements about proxy scores are scoped to the scores where the gap holds.
- If FIT is significantly more accurate than the measured allocation at >= 1 pair: Results main text states it with
  the pairs and margins, and the limitations paragraph names it.
- Between 6 and 9: counts reported in S6 and in the Results sentence, no change to abstract/conclusion wording.
- No result here changes the failed confirmatory test or the latency rows.

## Addendum (2026-11-06, after an 8-image smoke test of the script only; no full-data FIT score exists yet)
The smoke test (logs/*_fit_smoke8.json, yolo11s) shows that the entropy-calibrated graph also quantizes the inputs of
non-conv operators in every block (SiLU Sigmoid/Mul, Concat, Add, Upsample), so count_qdq != 2 x #convs for all
blocks, not only attention blocks. Gate item 3 stays "reported, not a failure"; the supplement states that FIT scores
only the conv weight and input quantizers. Score definition, gate items 1, 2, 4 and the consequences are unchanged.

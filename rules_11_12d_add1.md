# rules_11_12d_add1.md: addendum 1 to rules_11_12d.md (experiment D, RT-DETR-l)

Written 2026-11-06 ~22:40 (+07). The only D numbers seen when this was written are the gate numbers below
(metrics/gate_rtdetr-l.json). No sweep row (logs/rd_sens.log, metrics/sens_rtdetr-l_int8ent.json rows), no
output-SQNR score and no val2017 number had been read.

Gate (tuning split, 2000 images, AP): Ultralytics predictor 62.86, FP32-lifted onnxruntime 62.87, TensorRT FP32
62.85, all-FP16 62.86, whole-graph INT8 (unchanged recipe) 0.28, INT8 + F 26.31. G0, G1, G2 pass as written.

The rules did not anticipate a collapsed INT8 base. Nothing in rules_11_12d.md is changed: the recipe, blocks, F,
cost, primary arms, statistics and the consequences C1/C2/C3 stay as written and stay tied to the primary arms.
This addendum adds one diagnostic, one secondary arm and one proxy member, all fixed now.

## A1. Diagnostic of the collapse (descriptive, does not touch the base)
- The unrestored INT8 Q/DQ graph, lifted to FP32, through onnxruntime CPU EP on the first 200 tuning images
  (file-name order), scored against those images, beside the FP32-lifted all-restored graph on the same images.
- Census of the op types with DQ-fed inputs in the INT8 graph (in particular GridSample, Log, the reference-point
  Adds and the attention MatMuls).
- Reading: ORT INT8 near FP32 while TensorRT gives 0.28 -> engine-versus-simulation divergence, reported as such
  (it bears on the paper's thesis and on what the ORT-based SQNR scores). ORT INT8 also collapsed -> a property of
  the recipe on this graph, reported as such. Either way the collapse is reported as a result.

## A2. Secondary measured arm with base INT8 + F
- Reason: with the primary base at 0.28 AP and F (worth ~26 AP) still INT8 in every sweep row, single-block
  recoveries may be near zero, so a C3 outcome could mean either "measured AP fails" or "all-INT8 conditioning
  fails when the base has collapsed". Output SQNR conditions on all other blocks being restored.
- Design: base graph = INT8 with F restored (exports/rtdetr-l_int8entF.onnx); the same blocks, sweep code,
  MILP, cost and budgets; allocations evaluated as the primary ones; tag base "int8entF".
- Status: secondary. It may be used to explain a C3 outcome; it cannot move D above C2. It is compared with the
  same proxy family P (paired bootstrap, Holm within five budgets) and reported in full whatever the outcome.
- Note the asymmetry: the YOLO sweeps were also measured without F; this is disclosed with the arm.

## A3. Additional member of P: output SQNR on boxes and score only
- In RT-DETR's output, boxes and score lie in [0, 1] while the class index runs 0..79, so the row-wise squared error
  of the primary SQNR is dominated by class-index flips among low-score rows (in YOLO26 pixel boxes dominate).
- Added proxy: pooled output SQNR as arm 3 (2000 tuning images, same knapsack) with the class-index column
  excluded from both noise and signal power (columns 0-4 only). Tag _n2000bs, criterion sqnrq, prefix fxbs_.
- P becomes {pooled SQNR, per-image SQNR, decoder-first hybrid, pooled SQNR boxes+score}. A larger P only makes
  C1 harder. Disclosed with the result.

## A4. Checks
- FP32 val2017 of the TensorRT engine must be close to the published ~53.0 AP of RT-DETR-l; the value is reported.
- Operations unchanged: at most two heavy jobs; A3 runs after chain A, A2 after chain B.

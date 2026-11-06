# rules_11_12d.md: experiment D, RT-DETR-l (non-YOLO architecture), exploratory, post-freeze

Written 2026-11-06 ~22:00 (+07), before any AP number of RT-DETR-l exists (gate included).
At writing time the only RT-DETR artefacts are: exports/rtdetr-l.pt (cp of ~/topic_c_ivc/rtdetr-l.pt,
sha 6de60b10...), exports/rtdetr-l.onnx (sha 34a9e085...), the adapters rd_common.py, infer_eval_rd.py,
rd_quant.py (sha in logs/fx.sha256) and an INT8 quantization in progress (no evaluation).

Purpose: test whether the main finding (measured-AP block allocation vs simulation proxies, same solver,
budget, forced set and engine) carries over to a transformer detector. One model, one tuning draw:
whatever the outcome, it is exploratory.

## 1. Model, export, preprocessing, decoding
- Ultralytics RT-DETR-l (COCO, 32.97 M params, 108.3 GFLOPs per Ultralytics), ONNX exported exactly as the
  YOLO models: opset 17, simplify, static 1x3x640x640, FP32, nms=False. The export embeds the top-300
  post-processing; output (1,300,6) = [cx, cy, w, h normalised, score, class].
- Preprocessing: Ultralytics RTDETRPredictor (LetterBox scale_fill): stretch resize to 640x640, INTER_LINEAR,
  BGR->RGB, /255, no padding. Decoding: tools/rtdetr_decode.decode_rtdetr(mode="stretch"), boxes scaled by the
  original width and height. Confidence floor 1e-3 and max 300 detections, as for YOLO.
- Every metrics JSON written by infer_eval_rd.py carries arch="rtdetr"; downstream scripts refuse rows without it.
- The published scripts are not modified; rd_common.py rebinds preprocess, decode_output, conv_macs, blocks and
  run before they are imported.

## 2. Quantization (fixed before any number)
- Recipe of gate_headroom.py variant int8ent, unchanged: ModelOpt ONNX INT8, entropy calibration, the same
  512 COCO calibration images (RT-DETR stretch preprocessing), CPU calibration, op_types_to_exclude=["Sigmoid"],
  no other exclusion (no Softmax, LayerNorm or attention-MatMul exclusion), default quantized op types.
- Engines: tools/build_engine.py (TensorRT strongly typed), RTX 5090. No 5090 speed claim.

## 3. Blocks, forced set, cost
- Blocks (regexes in rd_common.blocks_rd; kept only if they restore >= 1 Q/DQ input):
  m0..m27 = top-level /model.K/ (backbone HGNet, AIFI encoder m11, CCFM neck);
  decoder /model.28/ split into dec.input_proj, dec.enc_query (enc_output + enc_bbox_head layers 0-1),
  dec.query_pos, dec.layer0..dec.layer5, dec.box_mlp (dec_bbox_head.* layers 0-1), dec.rest (remaining nodes).
- Forced set F (always restored, cost taken from every budget; analogue of the YOLO last box/class layers + DFL):
  the layers that produce the outputs: enc_score_head, dec_score_head.*, enc_bbox_head layers.2,
  dec_bbox_head.* layers.2 (rd_common.FORCE_RX; 9 nodes, 0.34% of MACs).
- Cost: MACs of Conv + MatMul + Gemm on the shape-inferred static graph (rd_common.lin_macs; 52.80 G in total,
  against 54.2 G = 108.3 GFLOPs/2 reported by Ultralytics; GridSample, Softmax and LayerNorm are not counted).
  This differs from the Conv-only cost of the YOLO runs because RT-DETR's transformer layers are Gemm/MatMul;
  disclosed wherever D is reported. Budgets B = 3, 5, 10, 15, 20%.

## 4. Gate (tuning split, 2000 images; val2017 not read)
- G0 FP32 reproduction: (a) the .pt through Ultralytics' own predictor (conf 0.001, max_det 300, FP32),
  converted to COCO JSON and scored with infer_eval.evaluate; (b) TensorRT engine of rtdetr-l.onnx through
  infer_eval_rd.py; (c) the FP32-lifted onnxruntime graph with every Q/DQ restored (the SQNR reference executor).
  Pass if |b - a| <= 0.3 AP and |c - a| <= 0.3 AP. On failure the adapters (preprocessing/decoding only) may be
  corrected and the gate re-run, logged; the quantization recipe and every rule above stay fixed. If it still
  fails, D stops and is reported as not run.
- G1 FP16: all-FP16 engine (every Q/DQ restored). If AP(FP32 engine) - AP(FP16) > 1.0, D stops (TensorRT FP16
  would be the bottleneck, not INT8); otherwise the gap is reported.
- G2 headroom: AP(all-FP16) - AP(INT8 + F) >= 1.0 AP. Otherwise D stops and is reported as a scope result
  (INT8 + F already within 1 AP of FP16 on RT-DETR-l).

## 5. Arms (fixed list, no grid; identical block sets at a budget are evaluated once and shared)
1. INT8 + F (no budget).
2. Measured, mean objective: block_sensitivity sweep through the adapters (engine, 2000 tuning images),
   allocate.py mean program at each B with F (same MILP).
3. Output SQNR, pooled: fx_factorial --mode noise --no-recon --n-img 2000, criterion sqnrq, same knapsack
   (the main-text form). Tensor: the final graph output (1,300,6), compared row by row without matching,
   exactly as for YOLO26 (whose output is also top-300 rows). Caveat stated with the result: rows follow the
   top-k ranking, so the noise includes reordering.
4. Output SQNR, per image: fx_pandey.py --n-img 256 (with its fx_factorial n256 reference run), as S6.
5. Decoder-first hybrid (analogue of fx_hybrid.py head-first): F and every /model.28/ block restored, the pooled
   SQNR scores fill the remaining budget over the other blocks; budgets where c(F u decoder) > B are recorded
   as infeasible (expected: 3, 5, 10%).
6. Greedy on the measured sensitivities (greedy_alloc.py), ablation only.
7. Random: random_alloc.py, seeds 1-5 at 5, 10, 20%, reported as mean +- sd.
Not run, with reason: Fisher knapsack and FIT (need the RT-DETR Hungarian/denoising loss through the deployed
graph scales, not adapted), reconstruction error (scope), auto_quantize (ModelOpt torch path not set up for
RT-DETR), worst-group/minimax and ExDark (scope: D tests the mean-objective comparison only).

## 6. Evaluation and statistics
- Each allocation: eval_alloc.py through the adapters, tuning split and full val2017 (5000 images), predictions saved.
- Paired image bootstrap on val2017, 1000 resamples (fx_boot.py conventions; pair X:Y means Y - X; exact
  p = max(2 min(p<=0, 1 - p<=0), 2/(n+1))). Pairs per budget: measured vs pooled SQNR, vs per-image SQNR,
  vs hybrid (feasible budgets), vs greedy, vs INT8 + F.
- Proxy family P = {pooled SQNR, per-image SQNR, hybrid}. At each budget, the strongest proxy is the member of P
  with the highest observed val2017 AP. Holm within the five budgets for the measured-vs-strongest-proxy test.

## 7. Consequences (fixed now)
- C1: measured significantly ahead of the strongest proxy at >= 3 of 5 budgets after Holm, and never
  significantly behind any member of P (uncorrected) -> a Results paragraph; abstract and conclusion may name
  RT-DETR-l as a single exploratory transformer check (one model, one tuning draw).
- C2: 1-2 budgets (after Holm), never significantly behind -> one Results sentence and a Supplementary section;
  claims in abstract, highlights and conclusion are scoped to YOLO detectors.
- C3: 0 budgets, or any member of P significantly ahead at any budget (uncorrected) -> stated in Results and named
  in the limitations; claims scoped to YOLO detectors.
- Gate stop (G0/G1/G2) -> Supplementary note with the gate numbers; claims scoped to YOLO detectors.
- Main-text additions must keep the review build layout (Table 4 p34, CRediT p40); otherwise D goes to the
  Supplement with a one-line pointer.
- No speed claim, no latency from fake-quant, no re-tuning of the recipe, budgets or blocks after any number.

## 8. Operations
- At most two heavy jobs; daily reboot ~12:13: block_sensitivity skips finished rows, the queue skips evaluated
  tags (eval_alloc refuses to overwrite). New tags only; sha256 of every new script and result in logs/fx.sha256.

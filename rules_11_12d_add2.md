# rules_11_12d_add2.md: addendum 2 to rules_11_12d.md (experiment D, RT-DETR-l), POST-HOC

Written 2026-11-07 ~00:10 (+07). Post-hoc: already read are the sweep rows (metrics/sens_rtdetr-l_int8ent.json),
the tune/val AP of the proxy allocations evaluated by chain A (fx2000_, fxp256_, hyb_), F@0 (tune 0.25, val 0.15),
FP32 val2017 (52.79) and the A1 diagnostic. The bootstrap/verdict (rd_boot.py) has not run.

Observation that triggers this addendum: F@0 (metrics/alloc_eval/rtdetr-l_int8ent/F_B0.json) and the gate's INT8+F
(rd_gate.py, exports/tmp/rtdetr-l_int8ent_restore_F.onnx) are the byte-identical ONNX graph (sha256 of the
serialized models equal, checked 2026-11-07), yet gave 0.25 and 26.31 AP on the same 2000 tuning images. Sweep rows
sit on a few levels (~0, ~26, ~54, ~61) across unrelated blocks, and allocations are non-monotone in budget.
The FP32 engine gave 62.85 (gate) and 62.57 (sweep). Hypothesis: TensorRT build-to-build (tactic) variation in the
INT8 engine of RT-DETR-l is large; then the sweep measures that variation, not block sensitivity.

Nothing in rules_11_12d.md or add1 is changed. This addendum adds one diagnostic and fixes how D is reported.

## A5. Build reproducibility test (fixed now)
- Graphs (each as eval_alloc.py builds it from exports/rtdetr-l_int8ent.onnx with qdq_restore.restore):
  (g1) FP32 exports/rtdetr-l.onnx; (g2) INT8 + F (F@0 regex); (g3) the sqnrq@0.03 allocation of
  configs/fx_rtdetr-l_int8ent_ho_noise_n2000.json (tune 61.53); (g4) the unrestored INT8 graph.
- For each graph: 4 independent builds with tools/build_engine.py, unchanged (no timing cache, no tactic flags,
  no op exclusion); each engine evaluated with infer_eval_rd.py on the first 200 tuning images (file-name order,
  data/tune2000/instances_tune_first200.json); build 1 is evaluated a second time (runtime repeat).
- Reading: "not reproducible" if the AP range across the 4 builds of g2, g3 or g4 exceeds 1.0 AP.
  The FP32 range (g1) and the runtime repeat are reported beside it.
- Runs after chain C (at most two heavy jobs). Results: metrics/det_rtdetr-l.json.

## Consequences (fixed now)
- Not reproducible: rd_boot.py's verdict is still computed and reported in full, labelled as confounded by
  build-to-build variation of the INT8 engine; D cannot support any claim beyond YOLO detectors (C1 is excluded
  whatever rd_boot returns); claims in abstract, highlights and conclusion stay scoped to YOLO detectors; one
  limitation sentence in the article and a Supplementary section with the gate, A1, A5 and the verdict.
- Reproducible (all ranges <= 1.0 AP): the F@0 vs gate discrepancy is reported as unexplained, D is reported
  per rules_11_12d.md section 7 with that discrepancy disclosed.
- A2 (chain F) runs as registered; if not reproducible, it is reported with the same label.
- No engine "fix" (deterministic tactics, timing cache, op exclusion, precision constraints) is part of D. Any such
  run would be a labelled diagnostic in the Supplement only and cannot change the consequences above.

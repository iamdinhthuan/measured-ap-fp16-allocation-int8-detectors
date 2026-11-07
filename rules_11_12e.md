# rules_11_12e.md: TensorRT build reproducibility on the YOLO detectors of the article (exploratory, post-freeze)

Written 2026-11-07 (+07; sha256 timestamp in logs/fx.sha256 is authoritative), before any of the runs below.
Motivation: rules_11_12d_add2.md A5 found that four builds of one INT8 graph of RT-DETR-l ranged over 53.6 AP
(metrics/det_rtdetr-l.json). The article reports one YOLO26-l rebuild (0.05 AP). This test asks whether the YOLO
engines of the article are reproducible across builds. No YOLO result of the article is re-run or replaced.

- Graphs (exports/<model>_int8ent.onnx, restored with qdq_restore.restore as eval_alloc.py does):
  yolo11l and yolo26l whole-graph INT8; yolo11l and yolo26l mean@0.1 allocations (regex of
  metrics/alloc_eval/<model>_int8ent/mean_B0.1.json, forced set included).
- 4 independent builds each with tools/build_engine.py unchanged; each engine evaluated with infer_eval.py (YOLO
  letterbox path of the article) on the first 200 tuning images (data/tune2000/instances_tune_first200.json).
- Reading, fixed now: max range over the four graphs <= 0.3 AP -> one sentence in the limitations ("four builds of
  each of four YOLO graphs ranged by at most X AP"); 0.3-1.0 AP -> the same sentence plus a statement that
  differences below that size between YOLO allocations are within build variation; > 1.0 AP -> the article's YOLO
  results are reported to the author as at risk and no manuscript change is made before a decision.
- Output: metrics/det_yolo.json. One heavy job; runs while at most one other heavy job runs.

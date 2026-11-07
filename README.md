# Reproducibility package for the measured-AP FP16 block allocation study

This archive accompanies the article

> *Measured detection accuracy versus proxy scores for FP16 block allocation in INT8 object detectors: a controlled study* (prepared for Information Sciences).

It contains the code, configurations, tuning-split manifests, allocation plans, and the metric and bootstrap files from which the tables and figures of the article and its supplementary material were produced. Datasets, checkpoints, ONNX graphs, TensorRT engines and raw prediction files are not redistributed (see "Not included").

## What the code does

1. `make_tuning_split.py` draws a 2000-image `train2017` tuning split (seed 20261025) that is disjoint from the 512 calibration images (`data/calib512/calib_files.txt`, the content of `data/coco/calib/images` read by `gate_headroom.py` through `tools/coco_common.py`).
2. The INT8 base graph is built with NVIDIA ModelOpt 0.45 (entropy calibration on the 512 images, Sigmoid unquantized). `qdq_restore.py` restores selected nodes of the explicit-Q/DQ ONNX graph to FP16. `tools/build_engine.py` builds strongly typed TensorRT engines.
3. `block_sensitivity.py` restores each block alone on the compiled engine and measures the AP recovery on the tuning split, overall and per group (`metrics/sens_*.json`).
4. `allocate.py` solves the budgeted integer program (mean or worst-group objective, forced set F) and writes the allocation plans (`configs/`).
5. Proxy baselines on the same solver, budget and forced set: `autoquant_select.py` (Fisher and ModelOpt `auto_quantize`), `recon_select.py` and `fx_factorial.py` (reconstruction error, output SQNR, criterion x direction factorial), `greedy_alloc.py`, `random_alloc.py`, `sim_sensitivity.py` (simulated-executor sweep).
6. `eval_alloc.py` builds and evaluates each allocation on COCO `val2017` (and ExDark test); `group_eval.py` computes per-group AP.
7. `bootstrap_fast.py`, `bootstrap_groups.py`, `fx_boot.py`, `fx_boot2.py` run the paired image-level bootstrap (seed 20261025); `y2_analyze.py` computes the 10,000-resample p-values and Holm counts.
8. `orin_export.py`, `orin_bench_alloc.sh`, `orin_ap.py` export the YOLO26-l allocations to the Jetson Orin Nano Super (TensorRT 10.3) and measure latency and AP there.
9. Revision arms added in v1.1.0 (exploratory, post-freeze; rules `rules_11_11.md`, `rules_11_11b.md`, `rules_11_11c.md`): `fx_pandey.py` (per-image output SQNR, 256 tuning images) and `a11_analyze.py`; `fit_score.py` (block FIT score from the Fisher trace and the deployed quantization scales) and `c11_analyze.py`; `exdarkcal_sqnr.py`, `fx_exdarkcal.py` and `b11_bootcmd.py` (allocation on the ExDark-calibrated graph); `s6_table.py` writes Supplementary Table S8 from `metrics/bootstrap/fx_*_a11.json` and `fx_*_c11.json`. The queues `fx_queue_b11*.sh`, `fx_queue_c11.sh`, `fx_queue_p.sh`, `fx_boot_a11*.sh`, `fx_boot_c11.sh` and `boot_chains.sh` record the arguments.
10. Transformer-detector check (RT-DETR-l, exploratory; rules `rules_11_12d.md`, written before any RT-DETR number existed; addendum `rules_11_12d_add1.md`, written after the engine gate only; addendum `rules_11_12d_add2.md`, post hoc, written after the sweep and the proxy allocations had been read): `rd_common.py` adapts the unchanged pipeline (stretch preprocessing, `tools/rtdetr_decode.py`, Conv + MatMul + Gemm MAC cost, decoder block split), with `infer_eval_rd.py`, `rd_quant.py`, `rd_gate.py`, `rd_sens.py`, `rd_alloc.py`, `rd_queue.py` and the chains `rd_chain*.sh`; `rd_diag.py` (onnxruntime diagnostic), `rd_fx_bs.py` (output SQNR on boxes and scores), `rd_sensF.py`/`rd_allocF.py`/`rd_queueX.py` (secondary sweep on INT8 with the forced set restored), `rd_boot.py` (paired bootstrap and verdict: `metrics/bootstrap/rd_rtdetr-l.json`, `metrics/rd_rtdetr-l_verdict.json`), `rd_det.py` (four builds of each of four RT-DETR-l graphs: `metrics/det_rtdetr-l.json`) and `rd_det_yolo.py` (rules `rules_11_12e.md`; four builds of each of four YOLO graphs: `metrics/det_yolo.json`). The TensorRT 11.1 INT8 engine of RT-DETR-l was not reproducible between builds (up to 60.5 AP apart for one graph), so the verdict is reported as confounded and no claim on transformer detectors is made (Supplementary Section S7); the four YOLO graphs tested differed by at most 0.074 AP between builds. Prediction files (`*.pred.json.gz`) are not included.

Scripts named `queue_*.sh`, `fx_queue_*.sh`, `boot_*.sh` are the unattended job queues that were actually run, in order; they document the exact arguments. They contain absolute paths of the original machine (`/home/thuan/...`) that must be adapted.

## Protocol records

- `protocol/README_isci_method_frozen_20261025.md`: the confirmatory protocol as frozen on 2026-10-25 (internal planning notes, in Vietnamese); SHA-256 `934776253c13171a0612c2e9f4186d0972271866c18689fd5d6379bb0b5afa2e`, the digest printed in Supplementary Section S3.
- `protocol/README_isci_method_with_10_8a.md`: the same document with the clarification §10.8a appended on 2026-10-25 23:09 +07; SHA-256 `8773a8046a04dda8c122c0ab5ce42f8ed5bfaab023fb1702287195bb1eb3547c`, the digest in `FREEZE_RECORD.md`. Its first 375 lines are byte-identical to the frozen file.
- `FREEZE_RECORD.md`: SHA-256 digests of the confirmatory protocol at freeze time (2026-10-25) and after the documented clarification.
- `rules_11_9.md`, `rules_11_10.md`, `rules_11_11.md`, `rules_11_11b.md`, `rules_11_11c.md`, `rules_11_12d.md`: decision rules of the post-freeze exploratory experiments, written before their results existed (their digests and times are in `logs/fx.sha256`).
- `logs/fx.sha256`: digests of the result files of the post-freeze runs.

## Layout

| Path | Content |
|---|---|
| `*.py`, `tools/` | experiment, allocation, evaluation and bootstrap code |
| `configs/` | allocation plans and score files per detector, method and budget |
| `data/` | calibration list, tuning-split manifests (`tune2000`, `tune_d1`, `tune_d2`), ExDark split manifest |
| `metrics/` | AP and per-group AP of every evaluated engine, sensitivity sweeps, scores, bootstrap outputs (`metrics/bootstrap/`), Jetson latency (`metrics/orin_latency/`), Jetson AP check (`metrics/orin_e4/`), exploratory 25 W latency session (`metrics/orin_phaseB/`) |
| `results/` | summary tables collected during the runs |
| `tests/` | unit test of the Q/DQ restoration |
| `manuscript_inputs/` | `make_tables.py` and `make_vis_figs.py` with the inputs they read (`data/`): run summaries, allocation plans, Jetson records and bootstrap files. The bootstrap files `g_*.json` there are byte-identical copies of files in `metrics/bootstrap/` under the names used by the table script (e.g. `g_yolo26s_coco_sim256.json` = `k5_yolo26s_sim256.json`). The two dataset images of the qualitative figure are not redistributed; `data/vis/SHA256SUMS` identifies them. |

## Not included

- COCO 2017 (https://cocodataset.org) and ExDark (https://github.com/cs-chan/Exclusively-Dark-Image-Dataset): download from the official sources; `make_exdark_coco.py` converts ExDark to COCO format with the official split.
- Ultralytics YOLO11-s/l and YOLO26-s/l checkpoints: official COCO-pretrained releases (Ultralytics 8.4.95).
- ONNX graphs, TensorRT engines and raw prediction files: regenerated by the scripts; they are large and engine files are tied to the GPU and TensorRT version.

## Software and hardware

Ultralytics 8.4.95, PyTorch 2.11 (2.6 for YOLO11-s export), ONNX opset 17, NVIDIA ModelOpt 0.45, TensorRT 11.1 on an RTX 5090 (driver 610.43), onnxruntime for the simulated executor, pycocotools for AP. Latency: Jetson Orin Nano Super, TensorRT 10.3, MAXN_SUPER power mode.

## License

Code and metadata are released under the MIT License (see `LICENSE`). Third-party datasets and checkpoints keep their own licenses.

## Citation

All versions: [10.5281/zenodo.23983314](https://doi.org/10.5281/zenodo.23983314) (concept DOI). This version (1.1.0): [10.5281/zenodo.24013653](https://doi.org/10.5281/zenodo.24013653); version 1.0.0: [10.5281/zenodo.23983315](https://doi.org/10.5281/zenodo.23983315). See `CITATION.cff`.

## Versions

- 1.1.0 (2026-11-06): revision arms of item 9 (per-image output SQNR, block FIT, ExDark-calibrated allocation, Supplementary Table S8), their decision rules and bootstrap files; rules and adapters of the RT-DETR-l check (item 10), without results.
- 1.0.0 (2026-11-05): first release.

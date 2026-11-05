# Decision rules for README 11.10 experiments (written 2026-11-03 ~08:45 +07, BEFORE any Y1 allocation, engine or AP and before any 10k-resample p-value)

All runs are post-freeze and exploratory. New tags only; nothing is overwritten. Conventions of rules_11_9.md
(paired image bootstrap on val2017, pair `X:Y` = difference Y - X, significant = 95% percentile interval excludes 0,
Holm within the named family from two-sided p = 2 min(p_le0, 1 - p_le0), floor 2/(n+1)).

## Y1. Head-first SQNR hybrid (reviewer M5: "whole head, then fill the remaining budget with SQNR")

- Definition: restore the forced set F and every head block (YOLO11: head.cv2.*, head.cv3.*, head.dfl; YOLO26:
  head.one2one_cv2.*, head.one2one_cv3.*, head.rest; i.e. the convolutions under `^/model\.23/`), then solve the
  published output-SQNR knapsack (scores of fx_scores_<model>_int8ent_noise_n2000.json, criterion sqnrq, score/max,
  recon_select.knapsack) over the non-head blocks with budget B - c(F u head), c = MAC share of the convolutions matched.
- Budgets: 3/5/10/15/20%. A budget with c(F u head) > B is NOT reported (no forcing, no partial head).
- Engine: eval_alloc.py, desktop TensorRT 11.1, val2017 only, prefix `fxh_`.
- Pairs (fx_boot.py): `hyb:mean` (positive favours the measured allocation), `sqnrq:hyb` (positive favours the hybrid
  over raw SQNR), `head:hyb` (positive favours the hybrid over whole-head FP16 alone).
- Family for the rule: all feasible large-detector pairs (YOLO11-l, YOLO26-l) of `hyb:mean`; small detectors descriptive.
- Consequences:
  - (a) measured significantly better at >= half of the feasible large pairs: the SQNR claim stays; Supplementary adds
    the hybrid with its counts; Results gets one clause.
  - (b) measured significantly better at <= 1 feasible large pair, or the hybrid significantly better at >= 1: abstract,
    highlights, introduction and conclusion must say that a head-first SQNR hybrid matches (or beats) the measured
    allocation at the budgets where the head fits, and the SQNR claim is scoped to "raw output SQNR" and to the budgets
    below the head cost; Results states it.
  - (c) otherwise: Results and Supplementary report the hybrid with counts; the abstract SQNR clause names
    "raw output SQNR".

## Y2. 10,000-resample bootstrap of the claim-bearing SQNR pairs (reviewer M2)

- Pairs: `sqnrq:mean` (published SQNR arm vs measured), all 20 detector-budget pairs, same prediction files as the
  published F1/F1b bootstraps (fx_boot.py with the same --prefer mappings), n = 10000, seed as fx_boot.
- Report: exact two-sided p (floor 2/10001), Holm counts within the 20-pair family (primary) and the 10 large pairs.
- Consequence: the counts in abstract/introduction/conclusion are replaced by the 10k counts if they differ; if the
  20-pair Holm count is lower than the 10-pair one plus the small-detector count suggests, the 20-pair count is the
  one quoted. "the floor 2/1001 can make these counts conservative" is replaced by the 10k statement.

## Y3. Pre-identified worst group on ExDark (reviewer M3) -- re-analysis, NOT blind

- The per-group differences already exist in metrics/bootstrap/g_yolo11s_exdark_light*.json; their values for the group
  *Low* were seen on 2026-11-03 before this file was written, so this item is descriptive only and cannot change any claim
  upward. Reported: Low-group AP difference minimax - mean per budget with interval, uncorrected and Holm (5 budgets).
  Seen values: +0.77 [-0.43,+1.53], +0.76 [-0.58,+1.69], +1.39 [+0.01,+3.64], -0.21 [-1.41,+0.93], +0.34 [-0.56,+1.05]
  (3/5/10/15/20%): significant at 10% only, uncorrected; not after Holm.

## Y4. INT8 calibrated on ExDark train (reviewer M7) -- appended 2026-11-03 ~08:55 +07, BEFORE any Y4 graph, engine or AP

- fx_exdarkcal.py: published int8ent recipe (ModelOpt entropy, 512 images, Sigmoid excluded) with the 512 calibration
  images drawn (seed 20261103) from the official ExDark train split. Engines: whole-graph INT8 (`exdarkcal_base`) and
  INT8 + F (`exdarkcal_headout`). Evaluated on ExDark test (lighting groups) and COCO val2017.
- Reference: COCO-calibrated engines of the article, `exdark_base` (INT8, 30.11 AP, relative loss 13% of FP32 AP,
  Low-group loss 29.8%) and `exdark_headout` (INT8 + F).
- Pairs (bootstrap_groups.py, light groups, n = 1000): exdark_base:exdarkcal_base, exdark_headout:exdarkcal_headout
  (positive favours ExDark calibration). val2017 AP of the ExDark-calibrated engines is descriptive (cost on COCO).
- Consequences:
  - (a) ExDark-calibrated whole-graph INT8 has relative ExDark AP loss <= 6.5% (half of 13%) OR Low-group relative
    loss <= 15%: the Discussion paragraph "Integer deployment and capture conditions", contribution 4 and the cover
    letter must say that most of the ExDark loss is a calibration-domain effect, give the ExDark-calibrated numbers,
    and scope the "almost 30% of the darkest group" statement to COCO calibration.
  - (b) otherwise: the same places state that calibrating on ExDark train leaves the loss at the measured values
    (the loss is not only a calibration-domain effect).

## Y5. Fisher score through the training graph (reviewer M4) -- appended 2026-11-03 ~09:00 +07, BEFORE any Y5 score, allocation, engine or AP

- fx_fisher_o2m.py (copy of autoquant_select.py; the published script is untouched): YOLO26-s (512 tuning images, as the
  published main-table Fisher) and YOLO26-l (2000). BatchNorm is fused but the one-to-many head is kept, and the head runs
  as in Ultralytics training: one-to-many convolutions on the backbone features, one-to-one convolutions on detached
  features. Loss = one-to-many term (v8DetectionLoss, tal_topk=10) + one-to-one term (tal_topk=7, tal_topk2=1), unit
  weights, tuning-set ground truth. Backbone and neck scores therefore come from the one-to-many loss, i.e. the gradient
  path that trained them; scores of one-to-many convolutions (absent from the deployed graph) are discarded. Same
  per-convolution MAC knapsack, forced set F and budgets (3/5/10/15/20%) as the published Fisher knapsack.
- Engine: eval_alloc.py, val2017 only, prefix `fxo_`. Pairs (fx_boot2.py, n=1000): `fo2m:mean` (positive favours the
  measured allocation), `fisher:fo2m` (positive favours the training-graph score over the published one-to-one score).
- Family for the rule: the 10 YOLO26 pairs of `fo2m:mean`.
- Consequences:
  - (a) measured significantly better at >= 5 of 10: Fisher statements stay; Supplementary adds the variant; Results gets
    one clause.
  - (b) measured significantly better at <= 2, or fo2m significantly better than measured at >= 1: the Results statement
    that the gradient score "selects convolutions no better than chance" and the YOLO26 Fisher margins are scoped to the
    one-to-one-path score in Results, Discussion and Conclusion, and the training-graph result is stated next to every
    Fisher count (abstract/introduction/conclusion).
  - (c) otherwise: Results and Supplementary report the variant with its counts.

## M8. Sensitivity sweep with the forced set already restored (reviewer: surrogate omits F) -- appended 2026-11-04 ~00:05 +07, BEFORE any M8 graph, engine, sensitivity, allocation or AP

- Scope: YOLO26-l only (the confirmatory COCO detector), base int8ent, tuning split tune2000, budgets 3/5/10/15/20%.
- fx_sens_f.py (copy of block_sensitivity.py; the published script is untouched): the reference engine is INT8 + F
  (F = force_regex of metrics/alloc_yolo26l_int8ent_ho.json), and every block b is restored on top of F, so
  r_b = AP_g(F u b) - AP_g(F) and the gaps are AP_g(FP16) - AP_g(F). FP32 and all-FP16 rows reuse the published
  tuning evaluations. Block names and regexes must equal those of metrics/sens_yolo26l_int8ent.json.
  Output: metrics/sens_yolo26l_int8ent_fbase.json (the row keyed "int8ent" there is the INT8 + F engine, tag int8entF).
- allocate.py unchanged, same budgets and --force-regex -> configs/fx_yolo26l_int8ent_ho_fsweep.json. Only the mean
  objective is evaluated.
- Engine: eval_alloc.py, val2017 only, prefix `fxs_`, only for budgets whose mean selection differs from the published
  mean selection (identical selections share the published prediction file, difference exactly 0).
- Pair (fx_boot2.py, n = 1000, published seed, --prefer mean=mean): `fsw:mean` (positive favours the published
  allocation measured without F). Holm over the 5 budgets reported.
- Consequences:
  - (a) the F-base mean selection equals the published one at >= 4 of 5 budgets, or `fsw:mean` is significant at no
    budget: Supplementary Section S6 gets one sentence (budgets changed, largest |difference| with interval); the
    Method sentence on the surrogate omitting F stays as is.
  - (b) fsw significantly better than the published allocation at >= 1 budget: Method (surrogate sentence), Results
    (surrogate subsection) and Limitations state that measuring without F mis-ranks blocks at those budgets, with the
    margin; the abstract numbers are NOT changed (published allocation is the frozen one).
  - (c) published significantly better at >= 1 budget and (b) does not hold: Supplementary reports it with the counts.

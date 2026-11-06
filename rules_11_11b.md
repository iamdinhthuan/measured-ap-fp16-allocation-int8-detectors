# Decision rules for README 11.11 experiment B (written 2026-11-06 09:08 +0700, BEFORE any sensitivity, allocation, engine or AP on the ExDark-calibrated graph beyond the Y4 rows already reported)

Trigger: second independent review (2026-11-06): ExDark allocations used the COCO-calibrated graph only, so allocation
effects are not separated from calibration-domain mismatch. Post-freeze, exploratory; new tags only; conventions of
rules_11_9.md / rules_11_10.md. Known before writing: the Y4 rows (exdarkcal base and headout on ExDark test and val2017).

## B. Allocation on the ExDark-calibrated YOLO11-s graph

- Base graph: exports/yolo11s_int8ent_exdarkcal.onnx (Y4: ModelOpt entropy calibration on 512 ExDark train images,
  seed 20261103, otherwise the article's recipe).
- Sensitivities: block_sensitivity.py --model yolo11s --base int8ent_exdarkcal --split exdark (same ExDark train images,
  blocks and engine pipeline as the article's ExDark sweep).
- Allocations: allocate.py, groups attr:light with the article's 8 lighting types, forced set F of the article
  (headout regex), budgets 3/5/10/15/20%, objectives mean, minimax, mean_matched.
- Baselines on the same graph: base and F only (Y4 engines), stem (m0 + F), output SQNR knapsack in both forms of
  rules_11_11.md A (pooled and per-image) on the same 256 ExDark train images as tag _exdark_n256, same knapsack.
- Engines: eval_alloc.py, ExDark test, prefix exdarkcal_; per-group AP by bootstrap_groups.py as for the article.
- Pairs (1000 resamples, image-level paired bootstrap, worst-group relative loss vs FP32 as in the article):
  B1 mean_matched:minimax (worst_rel and AP) at every budget; B2 sqnr_pooled:mean, sqnr_pandey:mean, stem:mean (AP);
  B3 calibration effect at equal budget: COCO-calibrated exdark_mean_B : exdarkcal_mean_B (AP and worst_rel).
- Reporting: one new Supplementary subsection (all rows, intervals, residual Low-group relative loss); one sentence in
  Results (ExDark) and in Discussion.
- Consequences:
  - The failed confirmatory test is unchanged whatever B shows. If minimax beats mean_matched significantly in worst_rel
    at >= 1 budget, Results says so as exploratory evidence after domain calibration, with the AP cost.
  - If, on the ExDark-calibrated graph, the measured mean is significantly better than stem and than both SQNR forms at
    fewer budgets than on the COCO-calibrated graph, ExDark allocation statements in the abstract, introduction and
    conclusion are scoped "with COCO calibration".
  - If B3 shows the ExDark-calibrated mean allocation significantly better in worst_rel at >= 3 budgets, the Discussion
    states that domain calibration and allocation are complementary, quoting B3; otherwise it states the measured result.

# Decision rules for README 11.11 experiment A (written 2026-11-06 09:05 +0700, BEFORE any per-image SQNR score, allocation, engine or AP)

Trigger: second independent review (2026-11-06), CRITICAL finding: the output-SQNR knapsack of the article pools noise
and signal over images before the ratio, whereas Pandey et al. (arXiv 2302.05397, Eq. 3) define
Omega_b = 10 log10( (1/N) sum_i E[F(x_i)^2] / E[e_b(x_i)^2] ), the log of the MEAN OF PER-IMAGE ratios.
All runs are post-freeze and exploratory. New tags only; nothing is overwritten. Conventions of rules_11_9.md / rules_11_10.md
(paired image bootstrap on val2017 or ExDark test, pair X:Y = difference Y - X, significant = 95% percentile interval
excludes 0, Holm within the named family).

## A. Per-image (Pandey Eq. 3) output SQNR

- Data and executor: identical to fx_factorial.py --mode noise --n-img 256 (tag _n256): first 256 tune2000 images in
  file_name order (ExDark: the same 256 evenly spaced ExDark train images as tag _exdark_n256), Q/DQ graph lifted to FP32,
  onnxruntime CUDA EP, network output 0, only block b quantized.
- Per image i: p_i = sum of squared reference outputs, e_ib = sum of squared output error with only b quantized
  (E[.] in Eq. 3 averages over the same elements in numerator and denominator, so sums and means give the same ratio).
- Score: v_b = 1 / mean_i (p_i / max(e_ib, 1e-30)) = 10^(-Omega_b/10) (inverse SQNR, as the pooled arm uses inverse SQNR);
  normalised by its maximum; same block costs, forced set F, budgets 3/5/10/15/20% and recon_select.knapsack.
- Sanity gate (must pass before any engine is built): the pooled score sum_i e_ib / sum_i p_i recomputed in the same run
  reproduces sqnrq of fx_scores_<model>_int8ent_noise_n256.json (resp. _noise_exdark_n256) to relative error <= 1e-3 for
  every block AND gives identical selections at all five budgets. If not: stop, report, no claim change from A.
- Engines: selections identical to the pooled arm reuse its engines (fx_dedup); others eval_alloc.py, val2017
  (YOLO11-s ExDark: exdark_test), prefix fxp_.
- Pairs (fx_boot.py, 1000 resamples, seed as fx_boot): pandey:mean (positive favours the measured allocation) and
  sqnrfx:pandey (positive favours the per-image form over the pooled form), 20 COCO detector-budget pairs and 5 ExDark.
- Reporting: Supplementary Section S6 adds the per-image arm (selections changed vs pooled, AP, both pair sets);
  protocol.tex names both forms.
- Consequences (claims about output SQNR in abstract, highlights, introduction, conclusion, cover letter):
  - every count and margin "measured vs output SQNR" is quoted for the form LESS favourable to the measured allocation
    (per budget pair, the smaller measured-minus-SQNR difference; counts = pairs significant under both forms);
  - if the per-image form is significantly better than the pooled form at >= 1 large-detector pair, Results says so and
    Table 2 caption points to it;
  - if the measured allocation is significantly better than the per-image form at <= 1 of the 10 large pairs, the
    abstract and highlight SQNR sentences are removed or replaced by the per-image result as it stands.

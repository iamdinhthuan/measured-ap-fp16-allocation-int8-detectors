"""Paired image-level bootstrap of COCO AP differences between prediction sets.

COCOeval.evaluate() runs once per prediction set; each resample only re-runs
accumulate() on a re-indexed evalImgs list (same resampled images for every
set), so differences are paired. Reports mean, 95% percentile CI and the
share of resamples with difference <= 0 for AP, AP_S/M/L and the worst-group
relative loss (needs --ref, the FP32 prediction set).

Usage: python bootstrap_pair.py --gt GT.json --a A.pred.json.gz --b B.pred.json.gz [--ref FP32.pred.json.gz] --n 1000
"""
import argparse
import contextlib
import copy
import gzip
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

KEYS = {"AP": 0, "AP_small": 3, "AP_medium": 4, "AP_large": 5}
GROUPS = ("AP_small", "AP_medium", "AP_large")


def evaluated(gt, pred_path):
    with gzip.open(pred_path) as f:
        preds = json.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(gt, gt.loadRes(preds), iouType="bbox")
        e.evaluate()
    return e


def stats_on(e, idx):
    """COCO stats on the image multiset e.params.imgIds[idx] (duplicates allowed)."""
    p0 = e.params
    I0, A, K = len(p0.imgIds), len(p0.areaRng), len(p0.catIds)
    I = len(idx)
    ev = [None] * (K * A * I)
    for k in range(K):
        for a in range(A):
            src = e.evalImgs[k * A * I0 + a * I0: k * A * I0 + (a + 1) * I0]
            ev[k * A * I + a * I: k * A * I + (a + 1) * I] = [src[i] for i in idx]
    f = copy.copy(e)
    p = copy.deepcopy(p0)
    p.imgIds = list(range(I))
    f.params, f._paramsEval, f.evalImgs = p, copy.deepcopy(p), ev
    with contextlib.redirect_stdout(io.StringIO()):
        f.accumulate()
        f.summarize()
    return {k: 100 * f.stats[i] for k, i in KEYS.items()}


def worst_rel(s, ref):
    return max(100 * (ref[g] - s[g]) / ref[g] for g in GROUPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)
    sets = {"a": evaluated(gt, a.a), "b": evaluated(gt, a.b)}
    if a.ref:
        sets["ref"] = evaluated(gt, a.ref)
    n_img = len(sets["a"].params.imgIds)
    full = np.arange(n_img)
    point = {k: stats_on(e, full) for k, e in sets.items()}
    rng = np.random.default_rng(a.seed)
    diffs = {k: [] for k in list(KEYS) + (["worst_rel"] if a.ref else [])}
    for _ in range(a.n):
        idx = rng.integers(0, n_img, n_img)
        s = {k: stats_on(e, idx) for k, e in sets.items()}
        for k in KEYS:
            diffs[k].append(s["b"][k] - s["a"][k])
        if a.ref:
            # positive = b has the smaller worst-group relative loss
            diffs["worst_rel"].append(worst_rel(s["a"], s["ref"]) - worst_rel(s["b"], s["ref"]))
    res = {"a": a.a, "b": a.b, "ref": a.ref, "n": a.n, "seed": a.seed, "n_images": n_img, "point": point, "diff_b_minus_a": {}}
    for k, v in diffs.items():
        v = np.array(v)
        res["diff_b_minus_a"][k] = {"mean": float(v.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                    "p_le0": float((v <= 0).mean())}
        print(f"{k:10s} mean={v.mean():+.3f} CI95=[{np.percentile(v, 2.5):+.3f},{np.percentile(v, 97.5):+.3f}] P(<=0)={(v <= 0).mean():.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

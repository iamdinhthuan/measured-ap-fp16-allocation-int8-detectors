"""Paired image-level bootstrap with COCO size groups, vectorised (same output as bootstrap_groups.py).

bootstrap_groups.py keeps one COCOeval per prediction set and re-runs pycocotools' accumulate() on every
resample, which needs ~1.5 GB per set and hours per run. Here each set is evaluated once with COCOeval,
the per-image detection records that accumulate() reads (scores, matches and ignore flags at maxDet=100,
number of non-ignored ground truths) are packed into flat arrays per (category, area range), and the
COCOeval object is freed. A resample gathers the records of the drawn images in draw order, which is the
order accumulate() concatenates them, so the stable sort, precision envelope and recall-threshold lookup
reproduce pycocotools exactly (checked against bootstrap_groups.py output with --check).

Usage: python bootstrap_fast.py --gt GT.json --sets name=path.pred.json.gz,... --ref NAME --pairs a:b,c:d
         --n 1000 --out x.json [--check previous_bootstrap_groups_output.json]
"""
import argparse
import contextlib
import gc
import io
import json
import os

import numpy as np
from pycocotools.coco import COCO

from bootstrap_pair import evaluated

GROUPS = ("AP_small", "AP_medium", "AP_large")
AREAS = ("AP", *GROUPS)  # area ranges all, small, medium, large in COCOeval order


class Packed:
    """Per (category, area): concatenated per-image detection records and per-image offsets."""

    def __init__(self, e):
        p = e.params
        self.I, self.K, self.A = len(p.imgIds), len(p.catIds), len(p.areaRng)
        self.T = len(p.iouThrs)
        self.rec_thrs = np.asarray(p.recThrs)
        md = p.maxDets[-1]
        assert md == 100 and self.A == 4
        self.cells = []
        for k in range(self.K):
            row = []
            for a in range(self.A):
                ev = e.evalImgs[k * self.A * self.I + a * self.I: k * self.A * self.I + (a + 1) * self.I]
                present = np.array([x is not None for x in ev])
                nd = np.array([len(x["dtScores"][:md]) if x is not None else 0 for x in ev], dtype=np.int64)
                npig = np.array([int(np.count_nonzero(np.asarray(x["gtIgnore"]) == 0)) if x is not None else 0
                                 for x in ev], dtype=np.int64)
                start = np.concatenate([[0], np.cumsum(nd)[:-1]])
                kept = [x for x in ev if x is not None]
                if nd.sum():
                    sc = np.concatenate([np.asarray(x["dtScores"][:md], dtype=np.float64) for x in kept])
                    dm = np.concatenate([np.asarray(x["dtMatches"])[:, :md] for x in kept], axis=1) != 0
                    di = np.concatenate([np.asarray(x["dtIgnore"])[:, :md] for x in kept], axis=1).astype(bool)
                else:
                    sc, dm, di = np.zeros(0), np.zeros((self.T, 0), bool), np.zeros((self.T, 0), bool)
                row.append((present, nd, npig, start, sc, dm, di))
            self.cells.append(row)

    def stats(self, idx):
        """AP for area all/small/medium/large on the image multiset idx (draw order)."""
        out = {}
        eps = np.spacing(1)
        R = len(self.rec_thrs)
        for a, name in enumerate(AREAS):
            vals = []
            for k in range(self.K):
                present, nd, npig, start, sc, dm, di = self.cells[k][a]
                sel = idx[present[idx]]
                if len(sel) == 0:
                    continue  # accumulate() skips a cell with no evaluated images (precision stays -1)
                n_gt = int(npig[sel].sum())
                if n_gt == 0:
                    continue
                lens = nd[sel]
                tot = int(lens.sum())
                prec = np.zeros((self.T, R))
                if tot:
                    g = np.repeat(start[sel] - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(tot)
                    s = sc[g]
                    order = np.argsort(-s, kind="mergesort")
                    m = dm[:, g][:, order]
                    ig = di[:, g][:, order]
                    tp = np.cumsum(m & ~ig, axis=1).astype(float)
                    fp = np.cumsum(~m & ~ig, axis=1).astype(float)
                    rc = tp / n_gt
                    pr = tp / (fp + tp + eps)
                    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                    for t in range(self.T):
                        ins = np.searchsorted(rc[t], self.rec_thrs, side="left")
                        ok = ins < tot
                        prec[t, ok] = pr[t, ins[ok]]
                vals.append(prec)
            out[name] = 100 * float(np.mean(np.stack(vals))) if vals else -100.0
        return out


def worst_rel(s, ref):
    return max(100 * (ref[g] - s[g]) / ref[g] for g in GROUPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None, help="bootstrap_groups.py output with the same sets/pairs/seed")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)
    paths = dict(kv.split("=", 1) for kv in a.sets.split(","))
    pk, img_ids = {}, None
    for k, path in paths.items():
        e = evaluated(gt, path)
        if img_ids is None:
            img_ids = list(e.params.imgIds)
        assert list(e.params.imgIds) == img_ids, k
        pk[k] = Packed(e)
        del e
        gc.collect()
        print("packed", k, flush=True)
    pairs = [p.split(":") for p in a.pairs.split(",")]
    n_img = len(img_ids)
    full = np.arange(n_img)
    point = {k: p.stats(full) for k, p in pk.items()}
    keys = ["AP", *GROUPS]
    diffs = {f"{x}:{y}": {k: [] for k in keys + ["worst_rel"]} for x, y in pairs}
    rng = np.random.default_rng(a.seed)
    for it in range(a.n):
        idx = rng.integers(0, n_img, n_img)
        s = {k: p.stats(idx) for k, p in pk.items()}
        for x, y in pairs:
            d = diffs[f"{x}:{y}"]
            for k in keys:
                d[k].append(s[y][k] - s[x][k])
            d["worst_rel"].append(worst_rel(s[x], s[a.ref]) - worst_rel(s[y], s[a.ref]))
        if it % 100 == 99:
            print("resample", it + 1, flush=True)
    res = {"gt": a.gt, "sets": paths, "ref": a.ref, "groups": "size", "group_names": list(GROUPS),
           "n": a.n, "seed": a.seed, "n_images": n_img, "point": point,
           "point_worst_rel": {k: worst_rel(point[k], point[a.ref]) for k in pk},
           "diff_b_minus_a": {}, "impl": "bootstrap_fast.py"}
    for pkey, d in diffs.items():
        res["diff_b_minus_a"][pkey] = {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)),
                                                                             float(np.percentile(v, 97.5))],
                                           "p_le0": float(np.mean(np.array(v) <= 0))} for k, v in d.items()}
        w = res["diff_b_minus_a"][pkey]
        print(pkey, {k: f"{w[k]['mean']:+.2f} [{w[k]['ci95'][0]:+.2f},{w[k]['ci95'][1]:+.2f}]"
                     for k in ("AP", "worst_rel")}, flush=True)
    if a.check:
        ref = json.load(open(a.check))
        dev = 0.0
        for k in pk:
            for g in keys:
                dev = max(dev, abs(ref["point"][k][g] - point[k][g]))
        for pkey, d in res["diff_b_minus_a"].items():
            for g, v in d.items():
                r = ref["diff_b_minus_a"][pkey][g]
                dev = max(dev, abs(r["mean"] - v["mean"]), *(abs(x - y) for x, y in zip(r["ci95"], v["ci95"])))
        res["check"] = {"against": a.check, "max_abs_dev": dev}
        print("CHECK max abs deviation", dev, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("BOOT_DONE", a.out)


if __name__ == "__main__":
    main()

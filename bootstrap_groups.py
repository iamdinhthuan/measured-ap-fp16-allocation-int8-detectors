"""Paired image-level bootstrap for several comparisons at once, with size or attr groups.

Every prediction set is evaluated once (COCOeval.evaluate); each resample draws
one image multiset and re-accumulates AP of every set on it, so all comparisons
share the same resamples (paired). Groups are COCO size classes (--groups size)
or image attributes (--groups attr:light --group-names Ambient,...), in which
case group AP is re-accumulated on the resampled images of that group.
worst_rel = max_g 100*(ref_g - set_g)/ref_g; the reported difference for a pair
A:B is B - A for AP and groups, and worst_rel(A) - worst_rel(B) (positive = B
has the smaller worst-group relative loss), as in bootstrap_pair.py.

Usage: python bootstrap_groups.py --gt GT.json --sets name=path.pred.json.gz,... --ref NAME
         --pairs a:b,c:d --groups attr:light --group-names Ambient,Low --n 1000 --out x.json
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np
from pycocotools.coco import COCO

from bootstrap_pair import evaluated, stats_on


def group_stats(e, idx, groups, members):
    s = stats_on(e, idx)
    out = {"AP": s["AP"]}
    if groups == "size":
        out.update({g: s[g] for g in ("AP_small", "AP_medium", "AP_large")})
    else:
        for g, mask in members.items():
            out[g] = stats_on(e, idx[mask[idx]])["AP"]
    return out


def worst_rel(s, ref, names):
    return max(100 * (ref[g] - s[g]) / ref[g] for g in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--groups", default="size")
    ap.add_argument("--group-names", default="")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)
    paths = dict(kv.split("=", 1) for kv in a.sets.split(","))
    ev = {k: evaluated(gt, p) for k, p in paths.items()}
    img_ids = ev[a.ref].params.imgIds
    for k, e in ev.items():
        assert e.params.imgIds == img_ids, k
    members, names = {}, ("AP_small", "AP_medium", "AP_large")
    if a.groups != "size":
        key = a.groups.split(":", 1)[1]
        attr = {im["id"]: str(im[key]) for im in gt.dataset["images"]}
        names = tuple(a.group_names.split(","))
        members = {g: np.array([attr[i] == g for i in img_ids]) for g in names}
    pairs = [p.split(":") for p in a.pairs.split(",")]
    full = np.arange(len(img_ids))
    point = {k: group_stats(e, full, a.groups, members) for k, e in ev.items()}
    keys = ["AP", *names]
    diffs = {f"{x}:{y}": {k: [] for k in keys + ["worst_rel"]} for x, y in pairs}
    rng = np.random.default_rng(a.seed)
    for _ in range(a.n):
        idx = rng.integers(0, len(img_ids), len(img_ids))
        s = {k: group_stats(e, idx, a.groups, members) for k, e in ev.items()}
        for x, y in pairs:
            d = diffs[f"{x}:{y}"]
            for k in keys:
                d[k].append(s[y][k] - s[x][k])
            d["worst_rel"].append(worst_rel(s[x], s[a.ref], names) - worst_rel(s[y], s[a.ref], names))
    res = {"gt": a.gt, "sets": paths, "ref": a.ref, "groups": a.groups, "group_names": list(names),
           "n": a.n, "seed": a.seed, "n_images": len(img_ids), "point": point,
           "point_worst_rel": {k: worst_rel(point[k], point[a.ref], names) for k in ev},
           "diff_b_minus_a": {}}
    for pk, d in diffs.items():
        res["diff_b_minus_a"][pk] = {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)),
                                                                           float(np.percentile(v, 97.5))],
                                         "p_le0": float(np.mean(np.array(v) <= 0))} for k, v in d.items()}
        w = res["diff_b_minus_a"][pk]
        print(pk, {k: f"{w[k]['mean']:+.2f} [{w[k]['ci95'][0]:+.2f},{w[k]['ci95'][1]:+.2f}]"
                   for k in ("AP", "worst_rel")}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("BOOT_DONE", a.out)


if __name__ == "__main__":
    main()

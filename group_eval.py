"""AP per group for one prediction set, for three kinds of groups.

  size      : COCO AP_small / AP_medium / AP_large
  supercat  : mean per-category AP (IoU .50:.95, area all, 100 dets) over the
              categories of each COCO supercategory
  attr:KEY  : AP on the images whose GT image record has field KEY == value
              (e.g. attr:light on ExDark), via re-accumulation on the subset

Usage: python group_eval.py --gt GT.json --pred P.pred.json.gz --groups size,supercat [--out x.json]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from bootstrap_pair import evaluated, stats_on
from pycocotools.coco import COCO


def per_category_ap(e):
    """AP per category id from an evaluated+accumulated COCOeval (area all, maxDets 100)."""
    prec = e.eval["precision"][:, :, :, 0, -1]  # T, R, K
    out = {}
    for k, cid in enumerate(e.params.catIds):
        p = prec[:, :, k]
        p = p[p > -1]
        if p.size:
            out[cid] = 100 * float(p.mean())
    return out


def group_ap(gt, e, kind):
    if kind == "size":
        s = stats_on(e, np.arange(len(e.params.imgIds)))
        return {g: s[g] for g in ("AP_small", "AP_medium", "AP_large")}
    if kind == "supercat":
        e.accumulate()
        ap = per_category_ap(e)
        groups = defaultdict(list)
        for c in gt.loadCats(gt.getCatIds()):
            if c["id"] in ap:
                groups[c["supercategory"]].append(ap[c["id"]])
        return {g: float(np.mean(v)) for g, v in sorted(groups.items())}
    if kind.startswith("attr:"):
        key = kind.split(":", 1)[1]
        pos = {i: n for n, i in enumerate(e.params.imgIds)}
        buckets = defaultdict(list)
        for im in gt.dataset["images"]:
            buckets[str(im[key])].append(pos[im["id"]])
        return {f"{key}={v}": stats_on(e, np.array(idx))["AP"] for v, idx in sorted(buckets.items())}
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--groups", default="size,supercat")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gt = COCO(a.gt)
    e = evaluated(gt, a.pred)
    res = {"gt": a.gt, "pred": a.pred, "AP": stats_on(e, np.arange(len(e.params.imgIds)))["AP"], "groups": {}}
    for kind in a.groups.split(","):
        res["groups"][kind] = group_ap(gt, e, kind)
        print(kind, {k: round(v, 2) for k, v in res["groups"][kind].items()})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

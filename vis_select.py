"""Select qualitative examples by a fixed rule (written before looking at any image).

Per image and configuration: TP = detections with score >= 0.3 matched greedily (by score)
to a non-crowd GT box of the same class at IoU >= 0.5.
Success case: YOLO26-l, COCO val2017, B = 10%, images with 4..12 non-crowd GT;
  argmax TP(mean) - max(TP(sqnr), TP(fisher)); ties: larger TP(fp32), then smaller id.
Failure case: YOLO11-s, ExDark test, B = 15%, lighting 'Low', images with >= 2 GT;
  argmax TP(sqnr) - TP(minimax); same tie rule.
Also reports how often each comparison goes either way over all candidates.
"""
import gzip, json, sys
from collections import defaultdict

TH, IOU = 0.3, 0.5
R = "metrics/alloc_eval/"

def load(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return json.load(f)

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0

def by_img(preds):
    d = defaultdict(list)
    for p in preds:
        if p["score"] >= TH:
            d[p["image_id"]].append(p)
    return d

def tp(dets, gts):
    used, n = set(), 0
    for p in sorted(dets, key=lambda x: -x["score"]):
        best, bj = IOU, -1
        for j, g in enumerate(gts):
            if j in used or g["category_id"] != p["category_id"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj); n += 1
    return n

def run(gt_path, cfgs, cand_filter, score, out_key):
    gt = load(gt_path)
    gts = defaultdict(list)
    for a in gt["annotations"]:
        if not a.get("iscrowd", 0):
            gts[a["image_id"]].append(a)
    imgs = {im["id"]: im for im in gt["images"]}
    P = {k: by_img(load(v)) for k, v in cfgs.items()}
    rows = []
    for i, im in imgs.items():
        if not cand_filter(im, gts[i]):
            continue
        t = {k: tp(P[k].get(i, []), gts[i]) for k in cfgs}
        rows.append((i, t))
    best = max(rows, key=lambda r: (score(r[1]), r[1]["fp32"], -r[0]))
    i = best[0]
    return {"key": out_key, "n_candidates": len(rows), "image": imgs[i], "tp": best[1],
            "n_gt": len(gts[i]), "gt": gts[i],
            "dets": {k: P[k].get(i, []) for k in cfgs},
            "cats": {c["id"]: c["name"] for c in gt["categories"]}, "rows": rows}

def tally(rows, a, b):
    w = sum(r[1][a] > r[1][b] for r in rows); l = sum(r[1][a] < r[1][b] for r in rows)
    return {"a": a, "b": b, "a_more": w, "b_more": l, "equal": len(rows) - w - l}

y = R + "yolo26l_int8ent/"
coco = run("data/coco_src/annotations/instances_val2017.json",
           {"fp32": y + "fp32_val.pred.json.gz", "int8": "metrics/pred_yolo26l_int8ent.json",
            "sqnr": y + "sqnr_B0.1_val.pred.json.gz", "fisher": y + "fisher_B0.1_val.pred.json.gz",
            "mean": y + "mean_B0.1_val.pred.json.gz"},
           lambda im, g: 4 <= len(g) <= 12,
           lambda t: t["mean"] - max(t["sqnr"], t["fisher"]), "coco_yolo26l_B0.1")
coco["tally"] = [tally(coco["rows"], "mean", "sqnr"), tally(coco["rows"], "mean", "fisher"),
                 tally(coco["rows"], "mean", "int8")]
x = R + "yolo11s_int8ent/exdark_"
exd = run("data/exdark/exdark_test.json",
          {"fp32": "metrics/exdark/yolo11s_fp32_test.pred.json.gz", "int8": x + "base_exdark_test.pred.json.gz",
           "mean": x + "mean_B0.15_exdark_test.pred.json.gz", "minimax": x + "minimax_B0.15_exdark_test.pred.json.gz",
           "sqnr": x + "sqnr_B0.15_exdark_test.pred.json.gz"},
          lambda im, g: im.get("light") == "Low" and len(g) >= 2,
          lambda t: t["sqnr"] - t["minimax"], "exdark_yolo11s_B0.15_low")
exd["tally"] = [tally(exd["rows"], "sqnr", "minimax"), tally(exd["rows"], "minimax", "int8")]
for d in (coco, exd):
    d.pop("rows")
json.dump({"rule": __doc__, "coco": coco, "exdark": exd}, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: {kk: d[kk] for kk in ("n_candidates", "tp", "n_gt", "tally")} | {"file": d["image"]["file_name"]}
                  for k, d in (("coco", coco), ("exdark", exd))}, indent=1))

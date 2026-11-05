"""Visualization figures of Section 5.2, generated from stored data in data/vis/.

figs/selection_map.pdf  measured single-block recoveries and the blocks selected by each
                        method on YOLO26-l (tuning split sweep, allocations as evaluated).
figs/qualitative.pdf    detections of the compared configurations on two held-out images
                        chosen by the fixed rule recorded in data/vis/vis_select_20261101.json.

Usage: python make_vis_figs.py [path/to/csl-figure.mplstyle]
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "vis")
MM = 1 / 25.4
BLUE, ORANGE, GREEN, VERM, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#7F7F7F"

if len(sys.argv) > 1:
    plt.style.use(sys.argv[1])
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

HEAD = {"head.one2one_cv2.0": "box P3", "head.one2one_cv2.1": "box P4", "head.one2one_cv2.2": "box P5",
        "head.one2one_cv3.0": "cls P3", "head.one2one_cv3.1": "cls P4", "head.one2one_cv3.2": "cls P5",
        "head.rest": "head rest"}


def selection_map():
    plan = json.load(open(os.path.join(D, "alloc_yolo26l_int8ent_ho.json")))
    blocks = plan["blocks"]
    r0 = np.array([r[0] for r in plan["r"]])
    cost = 100 * np.array(plan["cost"])
    rows = [("mean", 0.05, "Measured, mean, 5%"), ("mean", 0.1, "Measured, mean, 10%"),
            ("mean", 0.2, "Measured, mean, 20%"), ("minimax", 0.1, "Measured, worst-group, 10%"),
            ("sqnr", 0.1, "Output SQNR, 10%"), ("sqnrfx", 0.2, "Output SQNR, 20%"),
            ("reconfx", 0.2, "Reconstruction, 20%")]
    sel = []
    for m, b, _ in rows:
        a = json.load(open(os.path.join(D, "y26l_alloc", f"{m}_B{b}.json")))
        sel.append([blk in a["blocks"] for blk in blocks])
    sel = np.array(sel)
    n = len(blocks)
    labels = [HEAD.get(b, b) for b in blocks]

    fig = plt.figure(figsize=(190 * MM, 78 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.55, 2.0], hspace=0.08, left=0.19, right=0.995, top=0.97, bottom=0.17)
    ax0 = fig.add_subplot(gs[0])
    x = np.arange(n)
    ax0.bar(x, r0, width=0.72, color=[VERM if v < 0 else BLUE for v in r0], linewidth=0)
    ax0.axhline(0, color="black", lw=0.6)
    ax0.set_ylabel("recovery\n(AP)")
    ax0.set_xlim(-0.6, n - 0.4)
    ax0.tick_params(axis="x", length=0, labelbottom=False)
    ax0.spines["bottom"].set_visible(False)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.bar(x, cost, width=0.72, color=GREY, linewidth=0)
    ax1.set_ylabel("MAC\n(%)")
    ax1.tick_params(axis="x", length=0, labelbottom=False)
    ax1.spines["bottom"].set_visible(False)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    colors = [BLUE, BLUE, BLUE, GREEN, ORANGE, ORANGE, SKY]
    for i, (row, c) in enumerate(zip(sel, colors)):
        for j, on in enumerate(row):
            ax2.add_patch(Rectangle((j - 0.4, i - 0.38), 0.8, 0.76, facecolor=c if on else "white",
                                    edgecolor=c if on else "#D9D9D9", linewidth=0.5))
    ax2.set_ylim(len(rows) - 0.5, -0.5)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([r[2] for r in rows])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90)
    for s in ("top", "right", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=0)
    for a in (ax0, ax1, ax2):
        a.axvline(22.5, color="black", lw=0.5, ls=(0, (2, 2)))
    top = ax0.get_ylim()[1]
    ax0.set_ylim(top=top * 1.25)
    ax0.text(11, top * 1.22, "backbone and neck", ha="center", va="top", fontsize=7)
    ax0.text(26, top * 1.22, "head branches", ha="center", va="top", fontsize=7)
    fig.savefig(os.path.join(HERE, "figs", "selection_map.pdf"))
    plt.close(fig)


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def match(dets, gts, th=0.5):
    """Greedy class-aware matching by score, as in the selection rule."""
    used, flags = set(), []
    for p in sorted(dets, key=lambda d: -d["score"]):
        best, bj = th, -1
        for j, g in enumerate(gts):
            if j in used or g["category_id"] != p["category_id"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
        flags.append((p, bj >= 0))
    return flags, used


def qualitative():
    sel = json.load(open(os.path.join(D, "vis_select_20261101.json")))
    rows = [("coco", "000000222299.jpg", [("fp32", "FP32"), ("int8", "INT8"), ("fisher", "Fisher, 10%"),
                                           ("sqnr", "Output SQNR, 10%"), ("mean", "Measured, mean, 10%")]),
            ("exdark", "2015_04402.png", [("fp32", "FP32"), ("int8", "INT8"), ("mean", "Measured, mean, 15%"),
                                          ("minimax", "Measured, worst-group, 15%"),
                                          ("sqnr", "Output SQNR, 15%")])]
    imgs = [np.asarray(Image.open(os.path.join(D, fn)).convert("RGB")) for _, fn, _ in rows]
    # Display only: the low-light image is brightened (gamma 0.4); the detectors saw the original.
    imgs[1] = (255 * (imgs[1] / 255.0) ** 0.4).astype(np.uint8)
    ratios = [im.shape[0] / im.shape[1] for im in imgs]
    fig, axes = plt.subplots(2, 5, figsize=(190 * MM, (36 * sum(ratios) + 24) * MM), constrained_layout=False,
                             gridspec_kw={"height_ratios": ratios})
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.10, wspace=0.04, hspace=0.32)
    for r, (key, fn, cfgs) in enumerate(rows):
        s = sel[key]
        img = imgs[r]
        gts = s["gt"]
        for c, (k, title) in enumerate(cfgs):
            ax = axes[r, c]
            ax.imshow(img)
            flags, used = match(s["dets"][k], gts)
            for j, g in enumerate(gts):
                if j not in used:
                    x, y, w, h = g["bbox"]
                    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white", lw=0.75, ls=(0, (1.5, 1.5))))
            for p, ok in flags:
                x, y, w, h = p["bbox"]
                ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=SKY if ok else VERM, lw=0.9))
            tp = s["tp"][k]
            ax.set_title(f"{title}\n{tp} of {s['n_gt']} objects found", fontsize=7, pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
        axes[r, 0].set_ylabel("(a) COCO, YOLO26-l" if r == 0 else "(b) ExDark, YOLO11-s", fontsize=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=SKY, lw=2.0, label="correct detection"),
         Line2D([], [], color=VERM, lw=2.0, label="false detection"),
         Line2D([], [], color="black", lw=0.9, ls=(0, (1.5, 1.5)), label="missed object")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(HERE, "figs", "qualitative.pdf"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    selection_map()
    qualitative()

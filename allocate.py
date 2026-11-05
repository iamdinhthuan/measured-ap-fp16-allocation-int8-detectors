"""Choose which blocks to restore to FP16 from measured restoration sensitivities.

Inputs: metrics/sens_<model>_<base>.json (block_sensitivity.py) and the FP32
ONNX for per-block MAC counts. For a budget B (fraction of convolution MACs
allowed in FP16) it solves, with the additive surrogate
    gap_g(S) = gap_g(empty) - sum_{b in S} r[b, g],   g in {S, M, L},
two exact 0/1 programs:
  mean   : minimise gap_AP(S)                       (mean-sensitivity allocation)
  minimax: minimise max_g gap_g(S) / AP_g(FP16)     (size-group-robust allocation)
  mean_matched: mean, forced to spend at least the MAC that minimax used, so the
           two objectives can be compared at equal cost
Both subject to sum_{b in S} macs[b] <= B * total_macs. Gaps are relative to the
all-FP16 graph on the tuning split; val2017 is not read here.

Groups other than size: --groups attr:light --group-names Ambient,Low,... takes
AP per image-attribute group, computed from each row's saved predictions on
the sensitivity split (group_eval.group_ap) and cached next to the sens file.

Usage: python allocate.py --sens metrics/sens_yolo11s_int8max.json --onnx exports/yolo11s.onnx
"""
import argparse
import json
import os
import re

import numpy as np
import onnx
from onnx import shape_inference
from scipy.optimize import Bounds, LinearConstraint, milp

GROUPS = ("AP_small", "AP_medium", "AP_large")


def conv_macs(onnx_path):
    m = shape_inference.infer_shapes(onnx.load(onnx_path))
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(m.graph.value_info) + list(m.graph.output)}
    inits = {i.name: list(i.dims) for i in m.graph.initializer}
    out = {}
    for n in m.graph.node:
        if n.op_type != "Conv" or n.output[0] not in shapes or n.input[1] not in inits:
            continue
        co, ci_g, kh, kw = inits[n.input[1]]
        _, _, ho, wo = shapes[n.output[0]]
        out[n.name] = co * ci_g * kh * kw * ho * wo
    return out


def block_macs(blocks, macs):
    return {b: sum(v for k, v in macs.items() if re.search(rx, k)) for b, rx in blocks.items()}


def attr_group_stats(s, sens_path, kind, names):
    """Per-row {'AP': .., group: AP} (fractions) for an attr:KEY grouping, from saved predictions."""
    cache = sens_path.replace(".json", f".groups_{kind.replace(':', '_')}.json")
    if os.path.exists(cache):
        out = json.load(open(cache))
    else:
        from bootstrap_pair import evaluated
        from group_eval import group_ap
        from pycocotools.coco import COCO
        root = os.path.dirname(os.path.dirname(os.path.abspath(sens_path)))
        mdir = {"tune": "tune", "exdark": "tune_exdark"}[s.get("split", "tune")]
        gt = COCO(s["gt"])
        out = {}
        for row, v in s["rows"].items():
            pred = os.path.join(root, "metrics", mdir, s["model"], f"{v['tag']}.pred.json.gz")
            out[row] = group_ap(gt, evaluated(gt, pred), kind)
        json.dump(out, open(cache, "w"), indent=1)
    key = kind.split(":", 1)[1]
    return {row: {"AP": s["rows"][row]["stats"]["AP"], **{n: g[f"{key}={n}"] / 100 for n in names}}
            for row, g in out.items()}


def solve(r, cost, budget, base_gap, scale, objective, min_spend=-np.inf):
    """r: (n_blocks, n_targets) recovery; returns 0/1 selection.

    min_spend forces the mean program to spend at least that MAC fraction
    (mean_matched: the mean objective at the MAC actually used by minimax)."""
    n = len(cost)
    if objective == "mean":
        c = -r[:, 0]
        A = [cost[None, :]]
        lb, ub = [min_spend], [budget]
        integrality = np.ones(n)
        res = milp(c, constraints=LinearConstraint(np.vstack(A), lb, ub), integrality=integrality,
                   bounds=Bounds(0, 1))
        return np.round(res.x).astype(int)
    # minimax over size groups: variables [x_1..x_n, t]; min t
    # s.t. (base_gap_g - r[:, g] . x) / scale_g <= t  for each group g
    G = r.shape[1] - 1
    c = np.zeros(n + 1)
    c[-1] = 1.0
    rows, lb, ub = [], [], []
    for g in range(G):
        row = np.zeros(n + 1)
        row[:n] = -r[:, g + 1] / scale[g]
        row[-1] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(-base_gap[g] / scale[g])
    row = np.zeros(n + 1)
    row[:n] = cost
    rows.append(row)
    lb.append(-np.inf)
    ub.append(budget)
    # tie-break toward larger mean recovery
    c[:n] = -1e-4 * r[:, 0]
    integrality = np.r_[np.ones(n), 0]
    res = milp(c, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=integrality,
               bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]))
    return np.round(res.x[:n]).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--budgets", default="0.05,0.1,0.15,0.2,0.3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--groups", default="size", help="size or attr:KEY")
    ap.add_argument("--group-names", default="", help="attr values used as groups (pre-specified)")
    ap.add_argument("--force-regex", default=None,
                    help="nodes always restored (e.g. head output convs); their MACs are taken from every budget")
    a = ap.parse_args()
    s = json.load(open(a.sens))
    if a.groups == "size":
        groups = GROUPS
        stats = {k: v["stats"] for k, v in s["rows"].items()}
    else:
        groups = tuple(a.group_names.split(","))
        stats = attr_group_stats(s, a.sens, a.groups, groups)
    base, fp16 = stats[s["base"]], stats["fp16_all"]
    names = [b for b in s["blocks"] if b in stats]
    keys = ("AP",) + groups
    r = np.array([[100 * (stats[b][k] - base[k]) for k in keys] for b in names])
    base_gap = np.array([100 * (fp16[k] - base[k]) for k in groups])
    scale = np.array([100 * fp16[k] for k in groups])  # gap_g / scale_g = relative gap of group g
    bm = block_macs({b: s["blocks"][b] for b in names}, conv_macs(a.onnx))
    total = sum(conv_macs(a.onnx).values())
    cost = np.array([bm[b] / total for b in names])
    force_cost = 0.0
    if a.force_regex:
        force_cost = sum(v for k, v in conv_macs(a.onnx).items() if re.search(a.force_regex, k)) / total
        print(f"forced set costs {100 * force_cost:.2f}% MACs (sensitivities were measured without it)")
    print(f"gap to FP16 on tuning split (AP pts): AP={100*(fp16['AP']-base['AP']):.2f} "
          + " ".join(f"{g}={x:.2f}" for g, x in zip(groups, base_gap)))
    print("top blocks by mean recovery:", sorted(zip(names, r[:, 0].round(2), cost.round(3)), key=lambda x: -x[1])[:8])
    plan = {"model": s["model"], "base": s["base"], "split": s.get("split", "tune"), "groups": a.groups,
            "group_names": list(groups), "blocks": names, "cost": cost.tolist(), "r": r.tolist(),
            "force_regex": a.force_regex, "force_cost": force_cost, "allocations": {}}
    for B in map(float, a.budgets.split(",")):
        spent = {}
        for obj in ("mean", "minimax", "mean_matched"):
            if obj == "mean_matched":
                x = solve(r, cost, B - force_cost, base_gap, scale, "mean", min_spend=spent["minimax"] - 1e-9)
            else:
                x = solve(r, cost, B - force_cost, base_gap, scale, obj)
            spent[obj] = float(cost[x == 1].sum())
            pred_gap = base_gap - r[x == 1, 1:].sum(0)
            pred_ap_gap = 100 * (fp16["AP"] - base["AP"]) - r[x == 1, 0].sum()
            sel = [names[i] for i in np.flatnonzero(x)]
            plan["allocations"][f"{obj}@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost),
                                                 "pred_gap_AP": float(pred_ap_gap), "pred_gap_SML": pred_gap.tolist()}
            print(f"B={B:.2f} {obj:7s} cost={cost[x==1].sum() + force_cost:.3f} pred gapAP={pred_ap_gap:.2f} "
                  f"groups={pred_gap.round(2).tolist()} rel_max={100*np.max(pred_gap/scale):.2f}% {sel}")
    if a.out:
        json.dump(plan, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

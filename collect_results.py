"""Collect every evaluated configuration into one summary file for the paper.

For each metrics/alloc_eval/<model>_<base>/<tag>.json it records the MAC share
actually restored (recomputed from the union regex, so block/forced-set overlaps
are counted once), AP and group AP per split, and, on the tuning split, the gap
between the additive prediction and the measured AP (the interaction of
Proposition 1, plus the effect of the forced set, which the prediction omits).
ExDark light-group AP is computed from the saved predictions and cached next to
them as <tag>_exdark_test.groups.json (written once, never overwritten).

It also computes, per allocation plan, the heterogeneity index delta of
Corollary 1 and the predicted benefit of the worst-group objective.

Usage: python collect_results.py [--out results/summary_<stamp>.json]
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np

from allocate import GROUPS, attr_group_stats, conv_macs
from block_sensitivity import ROOT

LIGHT = ["Ambient", "Strong", "Single", "Weak", "Twilight", "Object", "Low", "Window"]
_macs = {}


def mac_share(model, rx):
    if model not in _macs:
        _macs[model] = conv_macs(os.path.join(ROOT, "exports", f"{model}.onnx"))
    m = _macs[model]
    return sum(v for k, v in m.items() if rx and re.search(rx, k)) / sum(m.values())


def light_groups(gt_path, pred_path):
    cache = pred_path.replace(".pred.json.gz", ".groups.json")
    if not os.path.exists(cache):
        from bootstrap_pair import evaluated
        from group_eval import group_ap
        from pycocotools.coco import COCO
        gt = COCO(gt_path)
        json.dump(group_ap(gt, evaluated(gt, pred_path), "attr:light"), open(cache, "w"), indent=1)
    g = json.load(open(cache))
    return {n: g[f"light={n}"] for n in LIGHT}


def split_stats(sp, res, pred_path):
    s = {k: 100 * res["stats"][k] for k in ("AP", "AP50", "AP_small", "AP_medium", "AP_large")}
    if sp.startswith("exdark") and os.path.exists(pred_path):
        s["light"] = light_groups(res["gt"], pred_path)
    return s


def heterogeneity(plan_path):
    plan = json.load(open(plan_path))
    sfx = "_exdark" if plan.get("split") == "exdark" else ""
    sens_path = os.path.join(ROOT, "metrics", f"sens_{plan['model']}_{plan['base']}{sfx}.json")
    s = json.load(open(sens_path))
    if plan.get("groups", "size") == "size":
        groups = GROUPS
        stats = {k: v["stats"] for k, v in s["rows"].items()}
    else:
        groups = tuple(plan["group_names"])
        stats = attr_group_stats(s, sens_path, plan["groups"], groups)
    base, fp16 = stats[s["base"]], stats["fp16_all"]
    r = np.array(plan["r"])  # columns: overall, groups (AP points)
    gap = np.array([100 * (fp16[k] - base[k]) for k in groups])
    scale = np.array([100 * fp16[k] for k in groups])
    D = 100 * (fp16["AP"] - base["AP"])
    kappa = gap / (scale * D)
    sdev = r[:, 1:] / scale - np.outer(r[:, 0], kappa)  # signed departure from proportional recovery
    dev = np.abs(sdev)
    delta = float(dev.max())
    idx = {b: i for i, b in enumerate(plan["blocks"])}
    scale_ref = float(np.abs(r[:, 1:] / scale).max())
    cost = np.sort(np.array(plan["cost"]))
    out = {"plan": os.path.basename(plan_path), "model": plan["model"], "groups": list(groups),
           "D": D, "gap": gap.tolist(), "scale": scale.tolist(), "kappa": kappa.tolist(),
           "delta": delta, "delta_rel": delta / scale_ref, "per_budget": {}}
    for key, sel in plan["allocations"].items():
        obj, B = key.split("@")
        if obj not in ("mean", "minimax", "mean_matched"):
            continue
        B = float(B)
        k = int(np.searchsorted(np.cumsum(cost), B - plan.get("force_cost", 0.0), side="right"))
        pb = out["per_budget"].setdefault(str(B), {"k": k, "bound_2kdelta": 2 * k * delta})
        pb[f"{obj}_pred_rel_max"] = float(np.max(np.array(sel["pred_gap_SML"]) / scale))
        pb[f"{obj}_blocks"] = sel["blocks"]
        pb[f"{obj}_h"] = float(np.abs(sdev[[idx[b] for b in sel["blocks"]]].sum(0)).max()) if sel["blocks"] else 0.0
    for pb in out["per_budget"].values():
        if "mean_pred_rel_max" in pb and "minimax_pred_rel_max" in pb:
            pb["pred_benefit"] = pb["mean_pred_rel_max"] - pb["minimax_pred_rel_max"]
            pb["bound_h"] = pb["mean_h"] + pb["minimax_h"]
            pb["same_selection"] = sorted(pb["mean_blocks"]) == sorted(pb["minimax_blocks"])
        if "mean_matched_pred_rel_max" in pb and "minimax_pred_rel_max" in pb:
            # objective effect at equal spend (the comparison of the frozen protocol, README 10.8a)
            pb["pred_benefit_matched"] = pb["mean_matched_pred_rel_max"] - pb["minimax_pred_rel_max"]
            pb["bound_h_matched"] = pb["mean_matched_h"] + pb["minimax_h"]
            pb["same_selection_matched"] = sorted(pb["mean_matched_blocks"]) == sorted(pb["minimax_blocks"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    out = a.out or os.path.join("results", f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    summary = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": [], "references": {},
               "heterogeneity": [], "tune_refs": {}}
    for d in sorted(glob.glob("metrics/alloc_eval/*_*")):
        model, base = os.path.basename(d).split("_", 1)
        sens_path = f"metrics/sens_{model}_{base}.json"
        if os.path.exists(sens_path):
            rows = json.load(open(sens_path))["rows"]
            summary["tune_refs"][f"{model}_{base}"] = {k: 100 * rows[k]["stats"]["AP"]
                                                       for k in ("fp32", base, "fp16_all") if k in rows}
        for f in sorted(glob.glob(f"{d}/*.json")):
            tag = os.path.basename(f)[:-5]
            if tag.endswith((".groups",)) or re.search(r"_(tune|val|exdark_test|exdark_train|exdark_val)$", tag):
                if tag.startswith("fp32_"):
                    sp = tag[5:]
                    res = json.load(open(f))
                    summary["references"].setdefault(model, {})[f"fp32_{sp}"] = split_stats(
                        sp, res, f.replace(".json", ".pred.json.gz"))
                continue
            res = json.load(open(f))
            row = {"model": model, "base": base, "tag": tag, "key": res.get("key"), "blocks": res.get("blocks"),
                   "mac_planned": res.get("cost_macs_frac"), "mac_actual": mac_share(model, res.get("regex")),
                   "qdq_restored": res.get("qdq_inputs_restored"), "pred_gap_AP_tune": res.get("pred_gap_AP_tune"),
                   "build_s": res.get("build_s"), "latency_5090_ms": res.get("latency_ms"), "splits": {}}
            for sp, r in res.get("splits", {}).items():
                row["splits"][sp] = split_stats(sp, r, os.path.join(d, f"{tag}_{sp}.pred.json.gz"))
            tref = summary["tune_refs"].get(f"{model}_{base}", {})
            if "tune" in row["splits"] and "fp16_all" in tref and row["pred_gap_AP_tune"] is not None:
                meas = tref["fp16_all"] - row["splits"]["tune"]["AP"]
                row["meas_gap_AP_tune"] = meas
                row["interaction_AP_tune"] = row["pred_gap_AP_tune"] - meas  # >0: measured better than predicted
            summary["configs"].append(row)
    for model in sorted({c["model"] for c in summary["configs"]}):
        ref = summary["references"].setdefault(model, {})
        for name in ("fp32", "int8ent", "int8ent_headout", "int8ent_head"):
            p = f"metrics/{model}_{name}.json"
            if os.path.exists(p) and name not in ref:
                ref[name] = {k: 100 * v for k, v in json.load(open(p))["stats"].items()
                             if k in ("AP", "AP50", "AP_small", "AP_medium", "AP_large")}
        onnx = os.path.join(ROOT, "exports", f"{model}.onnx")
        if os.path.exists(onnx):  # MAC share of the whole-head heuristic (last top-level module)
            hi = max(int(k.split("/")[1].split(".")[1]) for k in conv_macs(onnx) if k.startswith("/model."))
            ref["head_mac"] = mac_share(model, rf"^/model\.{hi}/")
        p = f"metrics/exdark/{model}_fp32_test.json"
        if os.path.exists(p):
            ref["fp32_exdark_test"] = split_stats("exdark_test", json.load(open(p)), p.replace(".json", ".pred.json.gz"))
    for p in sorted(glob.glob("metrics/alloc_*_ho*.json")):
        try:
            summary["heterogeneity"].append(heterogeneity(p))
        except (KeyError, FileNotFoundError) as e:
            print(f"skip {p}: {e!r}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    print(f"{len(summary['configs'])} configs -> {out}")
    for h in summary["heterogeneity"]:
        print(h["plan"], f"delta={h['delta']:.4f} rel={h['delta_rel']:.3f}",
              {B: (round(v.get("pred_benefit", float('nan')) * 100, 2), round(v.get("bound_h", float('nan')) * 100, 2),
                   round(v["bound_2kdelta"] * 100, 1))
               for B, v in h["per_budget"].items()})


if __name__ == "__main__":
    main()

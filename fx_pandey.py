"""Per-image output SQNR of Pandey et al. (arXiv 2302.05397, Eq. 3) on the article's data and executor (rules_11_11.md, A).

Same images, executor, graphs, blocks, forced set, costs, budgets and knapsack as
fx_factorial.py --mode noise (criterion sqnrq). Only the aggregation over images differs:
  pooled  sum_i e_ib / sum_i p_i                       (the article's arm; recomputed here as a sanity gate)
  pandey  1 / mean_i (p_i / e_ib) = 10^(-Omega_b/10)   (Eq. 3: log of the mean of per-image ratios)

Usage: python fx_pandey.py --model yolo26l --n-img 256 --tag _n256 [--split exdark]
"""
import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from allocate import block_macs, conv_macs  # noqa: E402
from fx_factorial import Images, only_quantized, session, sq, variant  # noqa: E402
from recon_select import knapsack  # noqa: E402
from sim_sensitivity import OPT_LEVEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--n-img", type=int, default=256)
    ap.add_argument("--mac-budgets", default="0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="tune", choices=("tune", "exdark"))
    a = ap.parse_args()
    os.chdir(ROOT)
    sens = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))
    blocks = sens["blocks"]
    fcfg = os.path.join("configs", f"fisher_{a.model}_{a.base}_ho.json")
    if not os.path.exists(fcfg):
        fcfg = fcfg.replace("_ho.json", "_ho_n2000.json")
    ref_tag = f"_n{a.n_img}"
    if a.split == "exdark":
        fcfg = os.path.join("configs", f"recon_{a.model}_{a.base}_ho_exdark.json")
        ref_tag = f"_exdark_n{a.n_img}"
    force_rx = json.load(open(fcfg))["force_regex"]
    ref_sc = json.load(open(os.path.join("metrics", f"fx_scores_{a.model}_{a.base}_noise{ref_tag}.json")))
    ref_cfg = json.load(open(os.path.join("configs", f"fx_{a.model}_{a.base}_ho_noise{ref_tag}.json")))
    assert ref_cfg["force_regex"] == force_rx, "forced set differs from the pooled arm"
    out_sc = os.path.join("metrics", f"fx_scores_{a.model}_{a.base}_pandey{a.tag}.json")
    out_cfg = os.path.join("configs", f"fx_{a.model}_{a.base}_ho_pandey{a.tag}.json")
    for p in (out_sc, out_cfg):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; pass a new --tag")
    gt = os.path.join("data", "tune2000", "instances_tune.json")
    img_dir = os.path.join("data", "tune2000", "images")
    ims = sorted(json.load(open(gt))["images"], key=lambda im: im["file_name"])[: a.n_img]
    if a.split == "exdark":
        gt, img_dir = os.path.join("data", "exdark", "exdark_train.json"), os.path.join("data", "exdark_raw", "ExDark")
        ims = sorted(json.load(open(gt))["images"], key=lambda im: im["file_name"])
        ims = ims[:: max(1, len(ims) // a.n_img)][: a.n_img]
    assert len(ims) == ref_sc["n_img"], (len(ims), ref_sc["n_img"])
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    out_name = q.graph.output[0].name
    t0 = time.time()
    xs = Images([os.path.join(img_dir, im["file_name"]) for im in ims])
    s = session(variant(q, ".*"), [out_name])
    ref = [s.run([out_name], {"images": x})[0].astype(np.float32) for x in xs]
    del s
    p = np.array([sq(r) for r in ref])
    scores, per_image = {}, {}
    for b, rx in blocks.items():
        t1 = time.time()
        sb = session(only_quantized(q, rx), [out_name])
        e = np.array([sq(sb.run([out_name], {"images": x})[0], r) for x, r in zip(xs, ref)])
        del sb
        gc.collect()
        pooled = float(e.sum() / p.sum())
        ratio = p / np.maximum(e, 1e-30)
        pandey = float(1.0 / ratio.mean())
        scores[b] = {"pooled": pooled, "pandey": pandey, "omega_db": float(10 * np.log10(ratio.mean())),
                     "n_zero_err": int((e <= 0).sum()), "ref_sqnrq": ref_sc["scores"][b]["sqnrq"]}
        per_image[b] = e.tolist()
        print(f"{b:22s} pooled={pooled:.4e} ref={ref_sc['scores'][b]['sqnrq']:.4e} pandey={pandey:.4e} "
              f"({time.time() - t1:.0f}s)", flush=True)
    rel = {b: abs(v["pooled"] - v["ref_sqnrq"]) / max(abs(v["ref_sqnrq"]), 1e-30) for b, v in scores.items()}
    json.dump({"model": a.model, "base": a.base, "split": a.split, "n_img": len(ims), "gt": gt,
               "images": [im["file_name"] for im in ims],
               "executor": f"onnxruntime {ort.__version__} CUDAExecutionProvider, opt level {OPT_LEVEL.name}, graph lifted to FP32",
               "rules": "rules_11_11.md A", "p_out": p.tolist(), "e_out": per_image, "scores": scores,
               "max_rel_err_pooled_vs_ref": max(rel.values())}, open(out_sc, "w"))

    macs = conv_macs(os.path.join("exports", f"{a.model}.onnx"))
    total = sum(macs.values())
    force_cost = sum(v for k, v in macs.items() if re.search(force_rx, k)) / total
    names = list(blocks)
    bm = block_macs(blocks, macs)
    cost = np.array([bm[b] / total for b in names])
    plan = {"model": a.model, "base": a.base, "split": a.split, "criterion": "fx_pandey (rules_11_11 A)",
            "scores": out_sc, "force_regex": force_rx, "force_cost": force_cost, "allocations": {}}
    same = True
    for crit in ("pooled", "pandey"):
        sc = np.array([scores[b][crit] for b in names])
        sc = sc / max(sc.max(), 1e-30)
        for B in map(float, a.mac_budgets.split(",")):
            x = knapsack(sc, cost, B - force_cost)
            sel = [names[i] for i in np.flatnonzero(x)]
            plan["allocations"][f"{crit}@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost)}
            if crit == "pooled":
                refsel = ref_cfg["allocations"][f"sqnrq@{B}"]["blocks"]
                same &= sorted(sel) == sorted(refsel)
            print(f"{crit} B={B:.2f} cost={cost[x == 1].sum() + force_cost:.3f} {sel}", flush=True)
    gate = max(rel.values()) <= 1e-3 and same
    plan["sanity_gate"] = {"max_rel_err": max(rel.values()), "pooled_selections_identical": bool(same), "pass": bool(gate)}
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print(f"GATE {'PASS' if gate else 'FAIL'} max_rel_err={max(rel.values()):.2e} identical={same}", flush=True)
    print("PANDEY_DONE", out_cfg, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

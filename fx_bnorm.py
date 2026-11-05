"""X1 (README 11.9, rules_11_9.md): branch-normalized output SQNR, only block b quantized (exploratory, post-freeze).

Tensor: pre-decoding head output [1, 84, 8400] (rows 0-3 box, 4-83 class sigmoid): YOLO11 'output0',
YOLO26 '/model.23/Concat_3_output_0'. Executor, graphs, images, forced set, block costs and knapsack are
those of fx_factorial.py (noise mode, sqnrq direction).
  bnorm  = 0.5 * (N_box / S_box + N_cls / S_cls)
  chnorm = mean over the 84 rows of N_c / S_c
Outputs (never overwritten): metrics/fx_scores_<model>_<base>_bnorm<tag>.json, configs/fx_<model>_<base>_ho_bnorm<tag>.json
Usage: python fx_bnorm.py --model yolo26l --tag _n2000
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

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from allocate import block_macs, conv_macs  # noqa: E402
from fx_factorial import Images, only_quantized, session, variant  # noqa: E402
from recon_select import knapsack  # noqa: E402


def tap_name(model, q):
    return q.graph.output[0].name if model.startswith("yolo11") else "/model.23/Concat_3_output_0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--n-img", type=int, default=2000)
    ap.add_argument("--mac-budgets", default="0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    sens = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))
    blocks = sens["blocks"]
    fcfg = os.path.join("configs", f"fisher_{a.model}_{a.base}_ho.json")
    if not os.path.exists(fcfg):
        fcfg = fcfg.replace("_ho.json", "_ho_n2000.json")
    force_rx = json.load(open(fcfg))["force_regex"]
    out_sc = os.path.join("metrics", f"fx_scores_{a.model}_{a.base}_bnorm{a.tag}.json")
    out_cfg = os.path.join("configs", f"fx_{a.model}_{a.base}_ho_bnorm{a.tag}.json")
    for p in (out_sc, out_cfg):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; pass a new --tag")
    gt = os.path.join("data", "tune2000", "instances_tune.json")
    img_dir = os.path.join("data", "tune2000", "images")
    ims = sorted(json.load(open(gt))["images"], key=lambda im: im["file_name"])[: a.n_img]
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    tap = tap_name(a.model, q)
    t0 = time.time()
    xs = Images([os.path.join(img_dir, im["file_name"]) for im in ims])
    s = session(variant(q, ".*"), [tap])
    ref = None
    for i, x in enumerate(xs):
        y = s.run([tap], {"images": x})[0]
        assert y.shape == (1, 84, 8400), y.shape
        if ref is None:
            ref = np.empty((len(ims), 84, 8400), np.float32)
        ref[i] = y[0]
    del s
    S = np.array([float(np.sum(ref[:, c, :].astype(np.float64) ** 2)) for c in range(84)])
    assert np.all(S > 0), S
    print(f"reference done ({time.time() - t0:.0f}s): S_box={S[:4].sum():.4e} S_cls={S[4:].sum():.4e}", flush=True)
    scores = {}
    for b, rx in blocks.items():
        t1 = time.time()
        sb = session(only_quantized(q, rx), [tap])
        N = np.zeros(84)
        for i, x in enumerate(xs):
            d = sb.run([tap], {"images": x})[0][0].astype(np.float64) - ref[i]
            N += np.einsum("cj,cj->c", d, d)
        del sb
        gc.collect()
        scores[b] = {"bnorm": 0.5 * (N[:4].sum() / S[:4].sum() + N[4:].sum() / S[4:].sum()),
                     "chnorm": float(np.mean(N / S)), "raw": float(N.sum() / S.sum()),
                     "N_box": float(N[:4].sum()), "N_cls": float(N[4:].sum())}
        print(f"{b:22s} bnorm={scores[b]['bnorm']:.3e} chnorm={scores[b]['chnorm']:.3e} raw={scores[b]['raw']:.3e} "
              f"({time.time() - t1:.0f}s)", flush=True)
    json.dump({"model": a.model, "base": a.base, "mode": "bnorm", "n_img": len(ims), "gt": gt, "tap": tap,
               "S_rows": S.tolist(), "scores": scores}, open(out_sc, "w"), indent=1)
    macs = conv_macs(os.path.join("exports", f"{a.model}.onnx"))
    total = sum(macs.values())
    force_cost = sum(v for k, v in macs.items() if re.search(force_rx, k)) / total
    names = list(blocks)
    bm = block_macs(blocks, macs)
    cost = np.array([bm[b] / total for b in names])
    plan = {"model": a.model, "base": a.base, "split": "tune", "criterion": "fx_bnorm (rules_11_9.md X1)",
            "scores": out_sc, "force_regex": force_rx, "force_cost": force_cost, "allocations": {}}
    for crit in ("bnorm", "chnorm"):
        sc = np.array([scores[b][crit] for b in names])
        sc = sc / max(sc.max(), 1e-30)
        for B in map(float, a.mac_budgets.split(",")):
            x = knapsack(sc, cost, B - force_cost)
            sel = [names[i] for i in np.flatnonzero(x)]
            plan["allocations"][f"{crit}@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost)}
            print(f"{crit} B={B:.2f} cost={cost[x == 1].sum() + force_cost:.3f} {sel}", flush=True)
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print("BNORM_DONE", out_cfg, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

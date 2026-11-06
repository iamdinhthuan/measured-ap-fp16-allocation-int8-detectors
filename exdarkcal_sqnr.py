"""rules_11_11b.md B: output-SQNR knapsack (pooled and per-image forms) on the ExDark-calibrated YOLO11-s graph.

Same images (256 evenly spaced ExDark train images, as fx_factorial --split exdark), executor, blocks, forced set,
block costs, budgets and knapsack as the article's ExDark SQNR arm; only the base graph differs
(exports/yolo11s_int8ent_exdarkcal.onnx). Writes configs/fx_yolo11s_int8ent_exdarkcal_ho_sqnr_exdark_n256.json with
keys sqnrpool@B and sqnrpandey@B (split exdark, base int8ent_exdarkcal, so eval_alloc builds from the right graph).
Usage: python exdarkcal_sqnr.py
"""
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

MODEL, BASE, N_IMG = "yolo11s", "int8ent_exdarkcal", 256
BUDGETS = (0.03, 0.05, 0.1, 0.15, 0.2)


def main():
    os.chdir(ROOT)
    blocks = json.load(open(os.path.join("metrics", f"sens_{MODEL}_int8ent.json")))["blocks"]
    ref_cfg = json.load(open(os.path.join("configs", f"fx_{MODEL}_int8ent_ho_noise_exdark_n256.json")))
    force_rx = ref_cfg["force_regex"]
    out_sc = os.path.join("metrics", f"fx_scores_{MODEL}_{BASE}_sqnr_exdark_n256.json")
    out_cfg = os.path.join("configs", f"fx_{MODEL}_{BASE}_ho_sqnr_exdark_n256.json")
    for p in (out_sc, out_cfg):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists")
    gt, img_dir = os.path.join("data", "exdark", "exdark_train.json"), os.path.join("data", "exdark_raw", "ExDark")
    ims = sorted(json.load(open(gt))["images"], key=lambda im: im["file_name"])
    ims = ims[:: max(1, len(ims) // N_IMG)][:N_IMG]
    q = onnx.load(os.path.join("exports", f"{MODEL}_{BASE}.onnx"))
    out_name = q.graph.output[0].name
    t0 = time.time()
    xs = Images([os.path.join(img_dir, im["file_name"]) for im in ims])
    s = session(variant(q, ".*"), [out_name])
    ref = [s.run([out_name], {"images": x})[0].astype(np.float32) for x in xs]
    del s
    p = np.array([sq(r) for r in ref])
    scores, per_image = {}, {}
    for b, rx in blocks.items():
        sb = session(only_quantized(q, rx), [out_name])
        e = np.array([sq(sb.run([out_name], {"images": x})[0], r) for x, r in zip(xs, ref)])
        del sb
        gc.collect()
        ratio = p / np.maximum(e, 1e-30)
        scores[b] = {"sqnrpool": float(e.sum() / p.sum()), "sqnrpandey": float(1.0 / ratio.mean()),
                     "omega_db": float(10 * np.log10(ratio.mean()))}
        per_image[b] = e.tolist()
        print(f"{b:22s} pooled={scores[b]['sqnrpool']:.4e} pandey={scores[b]['sqnrpandey']:.4e}", flush=True)
    json.dump({"model": MODEL, "base": BASE, "split": "exdark", "n_img": len(ims), "gt": gt,
               "images": [im["file_name"] for im in ims],
               "executor": f"onnxruntime {ort.__version__} CUDAExecutionProvider, opt level {OPT_LEVEL.name}, graph lifted to FP32",
               "rules": "rules_11_11b.md B", "p_out": p.tolist(), "e_out": per_image, "scores": scores},
              open(out_sc, "w"))
    macs = conv_macs(os.path.join("exports", f"{MODEL}.onnx"))
    total = sum(macs.values())
    force_cost = sum(v for k, v in macs.items() if re.search(force_rx, k)) / total
    names = list(blocks)
    bm = block_macs(blocks, macs)
    cost = np.array([bm[b] / total for b in names])
    plan = {"model": MODEL, "base": BASE, "split": "exdark", "criterion": "exdarkcal_sqnr (rules_11_11b B)",
            "scores": out_sc, "force_regex": force_rx, "force_cost": force_cost, "allocations": {}}
    for crit in ("sqnrpool", "sqnrpandey"):
        sc = np.array([scores[b][crit] for b in names])
        sc = sc / max(sc.max(), 1e-30)
        for B in BUDGETS:
            x = knapsack(sc, cost, B - force_cost)
            sel = [names[i] for i in np.flatnonzero(x)]
            plan["allocations"][f"{crit}@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost)}
            print(f"{crit} B={B:.2f} cost={cost[x == 1].sum() + force_cost:.3f} {sel}", flush=True)
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print("EXDARKCAL_SQNR_DONE", out_cfg, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Criterion x direction factorial and executor-corrected proxies (README 11.8, exploratory, post-freeze).

All graphs are the Q/DQ export of the detector, lifted to FP32 and executed by onnxruntime CUDA EP
(sim_sensitivity.to_fp32), the executor that reproduces the engine within ~0.3 AP. Blocks, forced set,
MAC costs and the knapsack are those of recon_select.py.

mode noise   per block b, on the first --n-img tuning images (file_name order, as recon_select.py):
               reconq  ||y_b^q - y_b||^2 / ||y_b||^2 at the outputs of block b, only b quantized (HMPTQ-style)
               sqnrq   output noise power / output signal power, only b quantized (Pandey et al.-style)
               sqnrr   (noise of the all-INT8 graph - noise with only b restored) / signal power
             reference = every Q/DQ pair bypassed, same lifted graph.
mode apq     per block b, on all 2000 tuning images: AP(reference) - AP(only b quantized), decoded and scored
             with the same decoder and pycocotools as the engine sweep.

Outputs (never overwritten): metrics/fx_scores_<model>_<base>_<mode><tag>.json and
configs/fx_<model>_<base>_ho_<mode><tag>.json (allocations '<crit>@<B>' for eval_alloc.py).

Usage: python fx_factorial.py --model yolo26l --mode noise --n-img 256 --tag _n256
       python fx_factorial.py --model yolo26l --mode noise --n-img 2000 --no-recon --tag _n2000
       python fx_factorial.py --model yolo26l --mode apq --tag _n2000
"""
import argparse
import copy
import gc
import json
import os
import re
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from coco_common import preprocess  # noqa: E402
from allocate import block_macs, conv_macs  # noqa: E402
from infer_eval import evaluate  # noqa: E402
from qdq_restore import restore  # noqa: E402
from recon_select import boundary_outputs, knapsack  # noqa: E402
from sim_sensitivity import OPT_LEVEL, predict, to_fp32  # noqa: E402

PROV = ["CUDAExecutionProvider"]


def variant(q, rx):
    """Lifted copy of q with the operators matching rx restored (rx None: unchanged)."""
    m = copy.deepcopy(q)
    if rx is not None:
        restore(m, rx)
    return to_fp32(m)


def only_quantized(q, rx):
    return variant(q, f"^(?!{rx[1:]})")  # every node outside block b restored: only b stays quantized


def session(m, outs):
    m = copy.deepcopy(m)
    have = {o.name for o in m.graph.output}
    extra = [t for t in outs if t not in have]
    if extra:
        vi = {v.name: v for v in shape_inference.infer_shapes(m).graph.value_info}
        for t in extra:
            m.graph.output.append(vi[t] if t in vi else onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVEL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=PROV)
    assert s.get_providers()[0] == PROV[0], s.get_providers()
    return s


def sq(a, b=None):
    d = a.astype(np.float64) if b is None else a.astype(np.float64) - b
    return float(np.sum(d * d))


class Images:
    """Preprocessed tuning images kept as uint8 (exact: preprocess divides a uint8 letterbox by 255)."""

    def __init__(self, paths):
        self.u8 = []
        for i, p in enumerate(paths):
            x = preprocess(p, 640)[0]
            u = np.round(x * 255.0).astype(np.uint8)
            if i == 0:
                assert np.array_equal(u.astype(np.float32) / 255.0, x)
            self.u8.append(u)

    def __iter__(self):
        return (u.astype(np.float32) / 255.0 for u in self.u8)


def noise_scores(q, blocks, xs, out_name, with_recon):
    ref_m = variant(q, ".*")
    s = session(ref_m, [out_name])
    ref = [s.run([out_name], {"images": x})[0].astype(np.float32) for x in xs]
    del s
    p_out = sum(sq(r) for r in ref)
    s = session(variant(q, None), [out_name])
    e_base = sum(sq(s.run([out_name], {"images": x})[0], r) for x, r in zip(xs, ref))
    del s
    print(f"reference/base done: base noise/signal = {e_base / p_out:.4e}", flush=True)
    bout = {b: boundary_outputs(ref_m, rx) for b, rx in blocks.items()}
    scores = {}
    for b, rx in blocks.items():
        t0 = time.time()
        tn = [out_name] + ([t for t in bout[b] if t != out_name] if with_recon else [])
        sq_b = session(only_quantized(q, rx), tn)
        if with_recon and len(tn) > 1:
            sref = session(ref_m, tn)
        e_q = e_rc = p_rc = 0.0
        for x, r in zip(xs, ref):
            y = dict(zip(tn, sq_b.run(tn, {"images": x})))
            e_q += sq(y[out_name], r)
            if with_recon and len(tn) > 1:
                yr = dict(zip(tn, sref.run(tn, {"images": x})))
                for t in tn[1:]:
                    e_rc += sq(y[t], yr[t].astype(np.float64))
                    p_rc += sq(yr[t])
        del sq_b
        if with_recon and len(tn) > 1:
            del sref
        s_r = session(variant(q, rx), [out_name])
        e_r = sum(sq(s_r.run([out_name], {"images": x})[0], r) for x, r in zip(xs, ref))
        del s_r
        gc.collect()
        scores[b] = {"sqnrq": e_q / p_out, "sqnrr": (e_base - e_r) / p_out, "n_out": len(bout[b])}
        if with_recon:
            scores[b]["reconq"] = e_rc / max(p_rc, 1e-30)
        print(f"{b:22s} " + " ".join(f"{k}={v:.3e}" for k, v in scores[b].items() if k != "n_out")
              + f" ({time.time() - t0:.0f}s)", flush=True)
    return scores, {"noise_base": e_base / p_out, "boundary_outputs": bout}


def apq_scores(q, blocks, items, gt):
    stats_ref, n = evaluate(gt, predict(variant(q, ".*"), items))
    print(f"reference AP={100 * stats_ref['AP']:.2f} (n={n})", flush=True)
    scores, rows = {}, {"reference": stats_ref}
    for b, rx in blocks.items():
        t0 = time.time()
        st, _ = evaluate(gt, predict(only_quantized(q, rx), items))
        rows[b] = st
        scores[b] = {"apq": stats_ref["AP"] - st["AP"]}
        print(f"{b:22s} AP={100 * st['AP']:.2f} drop={100 * scores[b]['apq']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
        gc.collect()
    return scores, {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--mode", required=True, choices=("noise", "apq"))
    ap.add_argument("--n-img", type=int, default=2000)
    ap.add_argument("--no-recon", action="store_true")
    ap.add_argument("--mac-budgets", default="0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="tune", choices=("tune", "exdark"),
                    help="exdark: ExDark train images, evenly spaced subset and forced set as recon_select.py --split exdark")
    a = ap.parse_args()
    os.chdir(ROOT)
    sens = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))
    blocks = sens["blocks"]
    fcfg = os.path.join("configs", f"fisher_{a.model}_{a.base}_ho.json")
    if not os.path.exists(fcfg):
        fcfg = fcfg.replace("_ho.json", "_ho_n2000.json")
    if a.split == "exdark":
        assert a.mode == "noise"
        fcfg = os.path.join("configs", f"recon_{a.model}_{a.base}_ho_exdark.json")
    force_rx = json.load(open(fcfg))["force_regex"]
    out_sc = os.path.join("metrics", f"fx_scores_{a.model}_{a.base}_{a.mode}{a.tag}.json")
    out_cfg = os.path.join("configs", f"fx_{a.model}_{a.base}_ho_{a.mode}{a.tag}.json")
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
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    t0 = time.time()
    if a.mode == "noise":
        xs = Images([os.path.join(img_dir, im["file_name"]) for im in ims])
        scores, extra = noise_scores(q, blocks, xs, q.graph.output[0].name, not a.no_recon)
        crits = ["sqnrq", "sqnrr"] + ([] if a.no_recon else ["reconq"])
    else:
        assert a.n_img == 2000, "apq uses the full tuning set (its ground-truth file)"
        items = sorted((im["id"], os.path.join(img_dir, im["file_name"])) for im in ims)
        scores, extra = apq_scores(q, blocks, items, gt)
        crits = ["apq"]
    json.dump({"model": a.model, "base": a.base, "mode": a.mode, "n_img": len(ims), "gt": gt,
               "executor": f"onnxruntime {ort.__version__} CUDAExecutionProvider, opt level {OPT_LEVEL.name}, graph lifted to FP32",
               "scores": scores, **extra}, open(out_sc, "w"), indent=1)

    macs = conv_macs(os.path.join("exports", f"{a.model}.onnx"))
    total = sum(macs.values())
    force_cost = sum(v for k, v in macs.items() if re.search(force_rx, k)) / total
    names = list(blocks)
    bm = block_macs(blocks, macs)
    cost = np.array([bm[b] / total for b in names])
    plan = {"model": a.model, "base": a.base, "split": a.split, "criterion": f"fx_factorial {a.mode}",
            "scores": out_sc, "force_regex": force_rx, "force_cost": force_cost, "allocations": {}}
    for crit in crits:
        sc = np.array([scores[b][crit] for b in names])
        sc = sc / max(sc.max(), 1e-30)
        for B in map(float, a.mac_budgets.split(",")):
            x = knapsack(sc, cost, B - force_cost)
            sel = [names[i] for i in np.flatnonzero(x)]
            plan["allocations"][f"{crit}@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost)}
            print(f"{crit} B={B:.2f} cost={cost[x == 1].sum() + force_cost:.3f} {sel}", flush=True)
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print("FX_DONE", out_cfg, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

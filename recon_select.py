"""Reconstruction-error block selection (post-freeze baseline, outside the confirmatory rule).

Two published simulation-based sensitivities, computed on the same Q/DQ graph, blocks and
tuning images as the engine sweep, then fed to the same MAC knapsack as the Fisher baseline:

  recon  per-block reconstruction error (HMPTQ-style): only block b is quantized, all other
         nodes are restored; score = ||y_b^q - y_b||^2 / ||y_b||^2 at the outputs of block b.
  sqnr   output noise (Pandey et al.-style): same single-block graphs, score = noise power at
         the network output relative to the signal power, i.e. 1/SQNR, assumed additive.

The Q/DQ graph is executed in onnxruntime (simulated quantization), not by TensorRT.

Usage:
    python recon_select.py --model yolo11s --n-img 256 --mac-budgets 0.03,0.05,0.1,0.15,0.2 --out-tag ""
"""
import argparse
import copy
import json
import os
import re
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from coco_common import preprocess  # noqa: E402
from allocate import block_macs, conv_macs  # noqa: E402
from qdq_restore import restore  # noqa: E402


def boundary_outputs(model, rx):
    """Outputs of nodes in the block that leave the block (consumed outside it or graph outputs)."""
    g = model.graph
    inside = {n.name for n in g.node if re.search(rx, n.name)}
    outs = {o for n in g.node if n.name in inside for o in n.output}
    leave = {i for n in g.node if n.name not in inside for i in n.input if i in outs}
    leave |= {o.name for o in g.output if o.name in outs}
    return sorted(leave)


def session(model, extra, providers):
    m = copy.deepcopy(model)
    have = {o.name for o in m.graph.output}
    vi = {v.name: v for v in shape_inference.infer_shapes(m).graph.value_info}
    for t in extra:
        if t not in have:
            m.graph.output.append(vi[t] if t in vi else onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
    return ort.InferenceSession(m.SerializeToString(), so, providers=providers)


def knapsack(score, cost, budget):
    res = milp(-score, constraints=LinearConstraint(cost[None, :], [-np.inf], [budget]),
               integrality=np.ones(len(cost)), bounds=Bounds(0, 1))
    return np.round(res.x).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--n-img", type=int, default=256)
    ap.add_argument("--mac-budgets", default="0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--split", default="tune", help="tune (COCO tune2000) or exdark (ExDark train images)")
    ap.add_argument("--out-tag", default="", help="suffix for output files, so reruns never overwrite")
    a = ap.parse_args()
    os.chdir(ROOT)
    sens = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))
    blocks = sens["blocks"]
    # same forced set as the Fisher knapsack, so the two simulation baselines differ only in the score
    fcfg = os.path.join("configs", f"fisher_{a.model}_{a.base}_ho.json")
    if not os.path.exists(fcfg):  # confirmatory models only have the 2000-image Fisher plan
        fcfg = fcfg.replace("_ho.json", "_ho_n2000.json")
    force_rx = json.load(open(fcfg))["force_regex"]
    out_cfg = os.path.join("configs", f"recon_{a.model}_{a.base}_ho{a.out_tag}.json")
    out_sc = os.path.join("metrics", f"recon_scores_{a.model}_{a.base}{a.out_tag}.json")
    for p in (out_cfg, out_sc):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; pass a new --out-tag instead of overwriting")
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    out_name = q.graph.output[0].name
    gt, img_dir = {"tune": ("data/tune2000/instances_tune.json", "data/tune2000/images"),
                   "exdark": ("data/exdark/exdark_train.json", "data/exdark_raw/ExDark")}[a.split]
    tune = json.load(open(gt))
    ims = sorted(tune["images"], key=lambda im: im["file_name"])
    # COCO file names are random ids; ExDark names are grouped by class, so take an evenly spaced subset
    ims = ims[: a.n_img] if a.split == "tune" else ims[:: max(1, len(ims) // a.n_img)][: a.n_img]
    xs = [preprocess(os.path.join(img_dir, im["file_name"]), 640)[0] for im in ims]
    prov = [a.provider]
    ref_m = copy.deepcopy(q)
    restore(ref_m, ".*")  # same graph, every Q/DQ pair bypassed: tensor names are unchanged
    # boundaries on the float graph: these tensors also exist when only block b is quantized
    bout = {b: boundary_outputs(ref_m, rx) for b, rx in blocks.items()}
    t0 = time.time()
    names = sorted({t for v in bout.values() for t in v} | {out_name})
    s_ref = session(ref_m, names, prov)
    acc = {b: [0.0, 0.0, 0.0, 0.0] for b in blocks}  # block err, block power, output err, output power
    qm = {}
    for b, rx in blocks.items():
        m = copy.deepcopy(q)
        restore(m, f"^(?!{rx[1:]})")  # every node outside block b restored: only b stays quantized
        qm[b] = m
    chunk = int(os.environ.get("RECON_CHUNK", "32"))  # reference tensors of a chunk are held in memory
    for c0 in range(0, len(xs), chunk):
        ref_out = [dict(zip(names, s_ref.run(names, {"images": x}))) for x in xs[c0:c0 + chunk]]
        for b in blocks:
            if not bout[b]:
                continue
            tn = bout[b] + ([out_name] if out_name not in bout[b] else [])
            s = session(qm[b], tn, prov)
            for x, r in zip(xs[c0:c0 + chunk], ref_out):
                y = dict(zip(tn, s.run(tn, {"images": x})))
                for t in bout[b]:
                    acc[b][0] += float(np.sum((y[t].astype(np.float64) - r[t]) ** 2))
                    acc[b][1] += float(np.sum(r[t].astype(np.float64) ** 2))
                acc[b][2] += float(np.sum((y[out_name].astype(np.float64) - r[out_name]) ** 2))
                acc[b][3] += float(np.sum(r[out_name].astype(np.float64) ** 2))
            del s
        del ref_out
        print(f"images {min(c0 + chunk, len(xs))}/{len(xs)} {time.time() - t0:.0f}s", flush=True)
    scores = {}
    for b in blocks:
        e_b, p_b, e_o, p_o = acc[b]
        scores[b] = {"recon": e_b / max(p_b, 1e-30), "sqnr_inv": e_o / max(p_o, 1e-30), "n_out": len(bout[b])}
        print(f"{b:22s} recon={scores[b]['recon']:.3e} 1/sqnr={scores[b]['sqnr_inv']:.3e}", flush=True)
    json.dump({"model": a.model, "base": a.base, "split": a.split, "gt": gt, "n_img": a.n_img, "provider": a.provider,
               "boundary_outputs": bout, "scores": scores}, open(out_sc, "w"), indent=1)

    onnx_fp = os.path.join("exports", f"{a.model}.onnx")
    macs = conv_macs(onnx_fp)
    total = sum(macs.values())
    force_cost = sum(v for k, v in macs.items() if re.search(force_rx, k)) / total
    names = list(blocks)
    bm = block_macs(blocks, macs)
    cost = np.array([bm[b] / total for b in names])
    plan = {"model": a.model, "base": a.base, "split": a.split, "criterion": "recon/sqnr (onnxruntime simulation)",
            "scores": out_sc, "force_regex": force_rx, "force_cost": force_cost, "allocations": {}}
    for crit, key in (("recon", "recon"), ("sqnr", "sqnr_inv")):
        sc = np.array([scores[b][key] for b in names])
        sc = sc / max(sc.max(), 1e-30)
        for B in map(float, a.mac_budgets.split(",")):
            x = knapsack(sc, cost, B - force_cost)
            sel = [names[i] for i in np.flatnonzero(x)]
            plan["allocations"][f"{crit}@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost)}
            print(f"{crit} B={B:.2f} cost={cost[x == 1].sum() + force_cost:.3f} {sel}", flush=True)
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print("RECON_DONE", out_cfg, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

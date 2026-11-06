"""rules_11_11c.md (C): FIT score (Fisher trace x quantization-noise model) on the deployed INT8 quantizer.

FIT_l = sum_c s_wc^2/12 * sum_{k in c} E_i[g_ik^2] + s_a^2/12 * sum_j E_i[(dL_i/da_ij)^2], per-image gradients
(batch 1) of the main-text Fisher loss (autoquant_select.py), scales from exports/<m>_int8ent.onnx. Block score =
sum over the block's convs; allocation as fisher_block.py (score / max, recon_select.knapsack, forced set of the
main-text Fisher plan, block MAC costs). Outputs (never overwritten):
  metrics/fit_scores_<m>_int8ent.json, configs/fx_<m>_int8ent_ho_fit.json (keys fit@B, sanity_gate)
Usage: python fit_score.py --model yolo26l
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import onnx
import torch
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from allocate import block_macs, conv_macs  # noqa: E402
from autoquant_select import conv_map  # noqa: E402
from coco_common import preprocess  # noqa: E402
from fisher_block import FISHER_PLAN, SCORE_FILE  # noqa: E402
from qdq_restore import count_qdq  # noqa: E402
from recon_select import knapsack  # noqa: E402


def dq_scale(prod, inits, tensor):
    n = prod.get(tensor)
    assert n is not None and n.op_type == "DequantizeLinear", tensor
    return numpy_helper.to_array(inits[n.input[1]]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--mac-budgets", default="0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--smoke", type=int, default=0, help="N images, outputs under logs/ with _smoke (test only)")
    a = ap.parse_args()
    os.chdir(ROOT)
    out_sc = os.path.join("metrics", f"fit_scores_{a.model}_{a.base}.json")
    out_cfg = os.path.join("configs", f"fx_{a.model}_{a.base}_ho_fit.json")
    if a.smoke:
        out_sc = os.path.join("logs", f"fit_scores_{a.model}_{a.base}_smoke{a.smoke}.json")
        out_cfg = os.path.join("logs", f"fx_{a.model}_{a.base}_ho_fit_smoke{a.smoke}.json")
    for p in (out_sc, out_cfg):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists")
    n_img = a.smoke or json.load(open(os.path.join("configs", SCORE_FILE[a.model])))["n_img"]
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    dev = "cuda"
    fp = YOLO(os.path.join("exports", f"{a.model}.pt")).model.float().eval().fuse(verbose=False).to(dev)
    for mod in fp.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    for p in fp.parameters():
        p.requires_grad_(True)
    onnx_path = os.path.join("exports", f"{a.model}.onnx")
    cmap = conv_map(fp, onnx_path)  # torch name -> ONNX conv name; asserts names and shapes
    gate = {}
    # gate 1-2: every FP32 ONNX conv mapped, fused weights equal the initializers
    fo = onnx.load(onnx_path)
    finit = {i.name: i for i in fo.graph.initializer}
    fconv = {n.name: n for n in fo.graph.node if n.op_type == "Conv"}
    mods = dict(fp.named_modules())
    gate["all_convs_mapped"] = set(cmap.values()) == set(fconv)
    wdiff = 0.0
    for tn, on in cmap.items():
        w = mods[tn].weight.detach().cpu().double().numpy()
        wo = numpy_helper.to_array(finit[fconv[on].input[1]]).astype(np.float64)
        wdiff = max(wdiff, float(np.abs(w - wo).max() / max(np.abs(wo).max(), 1e-30)))
    gate["max_rel_weight_diff"] = wdiff
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    qinit = {i.name: i for i in q.graph.initializer}
    prod = {o: n for n in q.graph.node for o in n.output}
    qconv = {n.name: n for n in q.graph.node if n.op_type == "Conv"}
    gate["int8_convs_match"] = set(qconv) == set(fconv)
    s_a = {on: float(dq_scale(prod, qinit, qconv[on].input[0])) for on in cmap.values()}
    s_w = {on: dq_scale(prod, qinit, qconv[on].input[1]) for on in cmap.values()}

    fp.args = get_cfg()
    head = fp.model[-1]
    if getattr(head, "end2end", False):  # as autoquant_select.py: undetached one-to-one path and loss term
        from ultralytics.utils.loss import v8DetectionLoss

        class OneToOne(type(head)):
            def forward(self, x):
                return None, {"one2one": self.forward_head(x, **self.one2one)}

        head.__class__ = OneToOne
        o2o = v8DetectionLoss(fp, tal_topk=7, tal_topk2=1)

        def loss(y, b):
            return o2o.loss(y[1]["one2one"], b)[0].sum()
    else:
        criterion = fp.init_criterion()

        def loss(y, b):
            return criterion(y, b)[0].sum()

    tune = json.load(open(os.path.join("data", "tune2000", "instances_tune.json")))
    cls_of = {c: i for i, c in enumerate(sorted(c["id"] for c in tune["categories"]))}
    anns = {}
    for an in tune["annotations"]:
        if not an.get("iscrowd", 0):
            anns.setdefault(an["image_id"], []).append(an)
    ims = sorted(tune["images"], key=lambda im: im["file_name"])[:n_img]

    act = {tn: 0.0 for tn in cmap}
    wsq = {tn: torch.zeros_like(mods[tn].weight) for tn in cmap}

    def pre_hook(tn):
        def f(mod, inp):
            x = inp[0].view_as(inp[0])  # own graph node: the gradient through this conv only

            def acc(g):
                act[tn] += float((g.double() ** 2).sum())
            x.register_hook(acc)
            return (x,) + tuple(inp[1:])
        return f
    hooks = [mods[tn].register_forward_pre_hook(pre_hook(tn)) for tn in cmap]
    t0 = time.time()
    losses = []
    for k, im in enumerate(ims):
        x, gain, padx, pady, _, _ = preprocess(os.path.join("data", "tune2000", "images", im["file_name"]), 640)
        bi, cl, bb = [], [], []
        for an in anns.get(im["id"], []):
            bx, by, bw, bh = an["bbox"]
            bi.append(0)
            cl.append(cls_of[an["category_id"]])
            bb.append([(bx + bw / 2) * gain + padx, (by + bh / 2) * gain + pady, bw * gain, bh * gain])
        b = {"img": torch.from_numpy(x).to(dev).requires_grad_(True),
             "batch_idx": torch.tensor(bi, dtype=torch.float32, device=dev),
             "cls": torch.tensor(cl, dtype=torch.float32, device=dev),
             "bboxes": torch.tensor(bb, dtype=torch.float32, device=dev).view(-1, 4) / 640}
        fp.zero_grad(set_to_none=True)
        L = loss(fp(b["img"]), b)
        L.backward()
        losses.append(float(L))
        with torch.no_grad():
            for tn in cmap:
                g = mods[tn].weight.grad
                if g is not None:
                    wsq[tn] += g ** 2
        if k % 200 == 0:
            print(f"{k}/{len(ims)} loss={float(L):.3f} {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    n = len(ims)
    fit, wterm, aterm = {}, {}, {}
    for tn, on in cmap.items():
        per_c = wsq[tn].double().sum(dim=tuple(range(1, wsq[tn].dim()))).cpu().numpy() / n
        wterm[on] = float((s_w[on] ** 2 / 12 * per_c).sum())
        aterm[on] = float(s_a[on] ** 2 / 12 * act[tn] / n)
        fit[on] = wterm[on] + aterm[on]
    gate["finite"] = bool(np.all(np.isfinite(list(fit.values()))) and np.all(np.isfinite(losses)))

    blocks = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))["blocks"]
    convs = list(fconv)
    members = {bk: [c for c in convs if re.search(blocks[bk], c)] for bk in blocks}
    names = [bk for bk in blocks if members[bk]]
    under = {bk: {"qdq": count_qdq(q, blocks[bk]), "convs": len(members[bk])} for bk in blocks
             if count_qdq(q, blocks[bk]) != 2 * len(members[bk])}
    gate["pass"] = bool(gate["all_convs_mapped"] and gate["int8_convs_match"] and gate["max_rel_weight_diff"] <= 1e-4
                        and gate["finite"])
    score = np.array([sum(fit[c] for c in members[bk]) for bk in names])
    json.dump({"model": a.model, "base": a.base, "n_img": n, "rules": "rules_11_11c.md C",
               "loss": "main-text Fisher loss (autoquant_select.py), per-image gradients", "fit": fit,
               "weight_term": wterm, "act_term": aterm, "block_scores": dict(zip(names, score.tolist())),
               "losses": losses, "under_scored_blocks": under, "sanity_gate": gate,
               "wallclock_s": round(time.time() - t0, 1)}, open(out_sc, "w"), indent=1)
    force_rx = json.load(open(os.path.join("configs", FISHER_PLAN[a.model])))["force_regex"]
    macs = conv_macs(onnx_path)
    total = sum(macs.values())
    force_cost = sum(v for kk, v in macs.items() if re.search(force_rx, kk)) / total
    bm = block_macs({bk: blocks[bk] for bk in names}, macs)
    cost = np.array([bm[bk] / total for bk in names])
    sc = score / max(score.max(), 1e-30)
    plan = {"model": a.model, "base": a.base, "split": "tune", "criterion": "fit (rules_11_11c.md C)",
            "scores": out_sc, "force_regex": force_rx, "force_cost": force_cost, "sanity_gate": gate,
            "under_scored_blocks": under, "allocations": {}}
    for B in map(float, a.mac_budgets.split(",")):
        x = knapsack(sc, cost, B - force_cost)
        sel = [names[i] for i in np.flatnonzero(x)]
        plan["allocations"][f"fit@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost)}
        print(f"fit B={B:.2f} cost={cost[x == 1].sum() + force_cost:.3f} {sel}", flush=True)
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print(f"GATE {'PASS' if gate['pass'] else 'FAIL'} {gate} under={under}", flush=True)
    print("FIT_DONE", out_cfg, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Baseline: ModelOpt mtq.auto_quantize as the layer-selection criterion.

auto_quantize searches, per Conv layer, between INT8 and "do not quantize"
under an effective-bits constraint (weight-count cost model), scoring layers
with its gradient-based sensitivity on a label-free output-matching loss
(MSE between the fake-quant and the FP32 detector outputs) on the train tuning
split. Only its *selection* is kept: the unquantized Conv layers are mapped to
ONNX node names and restored to FP16 in our entropy-calibrated Q/DQ graph by
eval_alloc.py, so calibration and engine pipeline are identical to the MILP
allocations and only the selection criterion differs. The resulting MAC
fraction is recorded so that methods are compared on the same MAC axis.

Because auto_quantize's cost is weight count, not MACs, a second variant keeps
its per-layer Fisher scores but solves the same MAC-budget knapsack (scipy
milp) as allocate.py, with the same optional forced set; it is written to a
separate plan (fisher_*) carrying force_regex so eval_alloc applies it.

Usage: python autoquant_select.py --model yolo11s --base int8ent --bits 8.4,8.8,9.2 \
           --mac-budgets 0.05,0.1,0.15,0.2 --force-regex REGEX --force-tag ho
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
import torch
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from allocate import conv_macs  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402
from coco_common import preprocess  # noqa: E402


def onnx_to_torch_name(node_name):
    """'/model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv' -> 'model.23.cv3.2.1.1.conv'.

    Exporter scopes are dotted paths relative to the enclosing scope; a scope that
    extends the previous label (cv3.2 -> cv3.2.1) replaces it instead of nesting."""
    segs = node_name.strip("/").split("/")[:-1]
    path, last = [], None
    for lab in segs:
        if last is not None and lab.startswith(last + "."):
            path[-1] = lab
        else:
            path.append(lab)
        last = lab
    return ".".join(path)


def conv_map(torch_model, onnx_path):
    """torch Conv2d name -> ONNX Conv node name, matched by name and checked by weight shape."""
    tc = {n: tuple(m.weight.shape) for n, m in torch_model.named_modules() if isinstance(m, torch.nn.Conv2d)}
    om = onnx.load(onnx_path)
    inits = {i.name: i for i in om.graph.initializer}
    out = {}
    for n in om.graph.node:
        if n.op_type != "Conv":
            continue
        tn = onnx_to_torch_name(n.name)
        assert tn in tc, (n.name, tn)
        assert tc[tn] == tuple(numpy_helper.to_array(inits[n.input[1]]).shape), (tn, n.name)
        out[tn] = n.name
    assert len(out) == len(tc), (len(out), len(tc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--bits", default="8.4,8.8,9.2")
    ap.add_argument("--n-img", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mac-budgets", default="", help="total MAC fractions for the Fisher-score knapsack")
    ap.add_argument("--force-regex", default=None, help="ONNX nodes kept FP16 in every Fisher-knapsack plan")
    ap.add_argument("--force-tag", default="")
    ap.add_argument("--out-tag", default="", help="suffix for output files, so reruns never overwrite")
    a = ap.parse_args()
    os.chdir(ROOT)
    import modelopt.torch.quantization as mtq
    from ultralytics import YOLO

    dev = "cuda"
    fp = YOLO(os.path.join("exports", f"{a.model}.pt")).model.float().eval().fuse(verbose=False).to(dev)
    for mod in fp.modules():  # auto_quantize's backward hooks reject in-place activations
        if hasattr(mod, "inplace"):
            mod.inplace = False
    onnx_path = os.path.join("exports", f"{a.model}.onnx")
    cmap = conv_map(fp, onnx_path)
    macs = conv_macs(onnx_path)
    total = sum(macs.values())
    from ultralytics.cfg import get_cfg
    fp.args = get_cfg()  # default box/cls/dfl gains for the training loss
    tune = json.load(open(os.path.join("data", "tune2000", "instances_tune.json")))
    cls_of = {c: i for i, c in enumerate(sorted(c["id"] for c in tune["categories"]))}  # coco80 order
    anns = {}
    for an in tune["annotations"]:
        if not an.get("iscrowd", 0):
            anns.setdefault(an["image_id"], []).append(an)
    ims = sorted(tune["images"], key=lambda im: im["file_name"])[: a.n_img]
    batches = []
    for i in range(0, len(ims), a.batch):
        xs, bi, cl, bb = [], [], [], []
        for j, im in enumerate(ims[i:i + a.batch]):
            x, gain, padx, pady, _, _ = preprocess(os.path.join("data", "tune2000", "images", im["file_name"]), 640)
            xs.append(x)
            for an in anns.get(im["id"], []):
                bx, by, bw, bh = an["bbox"]
                # original pixels -> normalised xywh in the 640x640 letterboxed input
                bi.append(j)
                cl.append(cls_of[an["category_id"]])
                bb.append([(bx + bw / 2) * gain + padx, (by + bh / 2) * gain + pady, bw * gain, bh * gain])
        # kept on the host and moved per step: 2000 images on the device alone take ~10 GB
        batches.append({"img": torch.from_numpy(np.concatenate(xs)),
                        "batch_idx": torch.tensor(bi, dtype=torch.float32),
                        "cls": torch.tensor(cl, dtype=torch.float32),
                        "bboxes": torch.tensor(bb, dtype=torch.float32).view(-1, 4) / 640})

    def todev(batch):
        return {k: v.to(dev) for k, v in batch.items()}
    head = fp.model[-1]
    if getattr(head, "end2end", False):
        # The deployed YOLO26 path is the one2one head alone (fuse() drops one2many). Ultralytics feeds that head
        # detached features, because in training the one2many head trains the backbone; kept as is, every backbone
        # gradient, hence every backbone score, would be zero. Score through the undetached one2one path instead,
        # with the one2one loss term of E2ELoss (tal_topk=7, topk2=1).
        from ultralytics.utils.loss import v8DetectionLoss

        class OneToOne(type(head)):
            def forward(self, x):
                return None, {"one2one": self.forward_head(x, **self.one2one)}

        head.__class__ = OneToOne
        o2o = v8DetectionLoss(fp, tal_topk=7, tal_topk2=1)

        def loss(y, batch):
            return o2o.loss(y[1]["one2one"], todev(batch))[0].sum()
    else:
        criterion = fp.init_criterion()

        def loss(y, batch):
            return criterion(y, todev(batch))[0].sum()

    def out(model, batch):
        return model(batch["img"].to(dev))

    t_start = time.time()
    res = {"model": a.model, "base": a.base, "n_img": len(ims), "loss": "ultralytics train loss, tune GT", "allocations": {}}
    scores = None
    for bits in map(float, a.bits.split(",")):
        m = copy.deepcopy(fp)
        m, state = mtq.auto_quantize(m, constraints={"effective_bits": bits},
                                     quantization_formats=["INT8_DEFAULT_CFG"], data_loader=batches,
                                     forward_step=out, loss_func=loss, num_calib_steps=len(batches),
                                     num_score_steps=len(batches), verbose=False)
        unq = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Conv2d)
               and not (getattr(mod, "input_quantizer", None) is not None and mod.input_quantizer.is_enabled)]
        nodes = [cmap[n] for n in unq if n in cmap]
        rx = "|".join("^" + re.escape(n) + "$" for n in nodes)
        cost = sum(macs.get(n, 0) for n in nodes) / total
        res["allocations"][f"autoquant@{bits}"] = {"blocks": [], "regex": rx, "torch_layers": unq, "cost": cost,
                                                  "effective_bits": bits}
        print(f"bits={bits} unquantized={len(unq)} MAC%={100 * cost:.2f} {unq[:8]}", flush=True)
        if scores is None:  # Fisher score of INT8 per layer (the no-quant choice scores 0)
            res["score_wallclock_s"] = round(time.time() - t_start, 1)  # calibration + scoring, first call
            scores = {}
            for v in state["candidate_stats"].values():
                for n in v["module_names"]:
                    scores[n] = float(v["scores"][0]) / len(v["module_names"])
        del m
        torch.cuda.empty_cache()
    res["fisher_scores"] = scores
    out_path = os.path.join("configs", f"autoquant_{a.model}_{a.base}{a.out_tag}.json")
    json.dump(res, open(out_path, "w"), indent=1)
    if a.mac_budgets:
        forced = {n for n in scores if a.force_regex and re.search(a.force_regex, cmap[n])}
        force_cost = sum(macs[cmap[n]] for n in forced) / total
        free = sorted(set(scores) - forced)
        r = np.array([scores[n] for n in free])
        c = np.array([macs[cmap[n]] / total for n in free])
        plan = {"model": a.model, "base": a.base, "criterion": "modelopt auto_quantize Fisher score",
                "force_regex": a.force_regex, "force_cost": force_cost, "allocations": {}}
        for B in map(float, a.mac_budgets.split(",")):
            x = milp(-r / r.max(), constraints=LinearConstraint(c[None], 0, B - force_cost),
                     integrality=np.ones(len(r)), bounds=Bounds(0, 1)).x.round()
            sel = [n for n, xi in zip(free, x) if xi == 1]
            rx = "|".join("^" + re.escape(cmap[n]) + "$" for n in sel)
            cost = float(c[x == 1].sum() + force_cost)
            plan["allocations"][f"fisher@{B}"] = {"blocks": [], "regex": rx, "torch_layers": sel, "cost": cost}
            print(f"fisher B={B} layers={len(sel)} MAC%={100 * cost:.2f} {sel[:8]}", flush=True)
        tag = f"_{a.force_tag}" if a.force_tag else ""
        json.dump(plan, open(os.path.join("configs", f"fisher_{a.model}_{a.base}{tag}{a.out_tag}.json"), "w"), indent=1)
    print("AQ_DONE", out_path, flush=True)


if __name__ == "__main__":
    main()

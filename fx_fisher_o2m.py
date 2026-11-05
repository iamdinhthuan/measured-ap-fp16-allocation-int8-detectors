"""Y5 (README 11.10, rules_11_10.md): Fisher knapsack scored through the YOLO26 training graph (exploratory).

Copy of autoquant_select.py (published script untouched). Differences: BatchNorm is fused but the one-to-many head is
kept; the head runs as in Ultralytics training (one-to-many convolutions on the backbone features, one-to-one
convolutions on detached features) and the loss is the one-to-many term (tal_topk=10) plus the one-to-one term
(tal_topk=7, tal_topk2=1) with unit weights, so backbone and neck are scored through the gradient path that trained
them. Scores of one-to-many convolutions (absent from the deployed ONNX graph) are discarded before the knapsack.
Outputs (never overwritten): configs/autoquant_<model>_<base>_o2m*.json, configs/fisher_<model>_<base>_<tag>_o2m*.json
Usage: python fx_fisher_o2m.py --model yolo26l --n-img 2000 --bits 8.4 --mac-budgets 0.03,0.05,0.1,0.15,0.2 \
           --force-regex REGEX --force-tag ho
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
    extra = sorted(set(tc) - set(out))
    assert all(re.match(r"model\.23\.cv[23]\.", n) for n in extra), extra[:5]
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
    fp = YOLO(os.path.join("exports", f"{a.model}.pt")).model.float().eval()
    head = fp.model[-1]
    assert getattr(head, "end2end", False), "Y5 applies to end-to-end (YOLO26) detectors only"
    head.fuse = lambda: None  # keep the one-to-many head; BatchNorm is still fused
    fp = fp.fuse(verbose=False).to(dev)
    assert head.cv2 is not None and head.cv3 is not None
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
    from ultralytics.utils.loss import v8DetectionLoss

    class TrainGraph(type(head)):
        def forward(self, x):  # training-mode head: one2many on x, one2one on detached x
            o2m = self.forward_head(x, **self.one2many)
            o2o = self.forward_head([xi.detach() for xi in x], **self.one2one)
            return None, {"one2many": o2m, "one2one": o2o}

    head.__class__ = TrainGraph
    l_o2m = v8DetectionLoss(fp, tal_topk=10)
    l_o2o = v8DetectionLoss(fp, tal_topk=7, tal_topk2=1)

    def loss(y, batch):
        b = todev(batch)
        return l_o2m.loss(y[1]["one2many"], b)[0].sum() + l_o2o.loss(y[1]["one2one"], b)[0].sum()

    def out(model, batch):
        return model(batch["img"].to(dev))

    t_start = time.time()
    res = {"model": a.model, "base": a.base, "n_img": len(ims), "loss": "one2many + one2one train loss (training graph), tune GT", "allocations": {}}
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
            dropped = sorted(n for n in scores if n not in cmap)
            assert all(re.match(r"model\.23\.cv[23]\.", n) for n in dropped), dropped[:5]
            res["dropped_one2many"] = dropped
            scores = {n: s for n, s in scores.items() if n in cmap}
        del m
        torch.cuda.empty_cache()
    res["fisher_scores"] = scores
    out_path = os.path.join("configs", f"autoquant_{a.model}_{a.base}_o2m{a.out_tag}.json")
    fis_path = os.path.join("configs", f"fisher_{a.model}_{a.base}{('_' + a.force_tag) if a.force_tag else ''}_o2m{a.out_tag}.json")
    for p in (out_path, fis_path):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists")
    json.dump(res, open(out_path, "w"), indent=1)
    if a.mac_budgets:
        forced = {n for n in scores if a.force_regex and re.search(a.force_regex, cmap[n])}
        force_cost = sum(macs[cmap[n]] for n in forced) / total
        free = sorted(set(scores) - forced)
        r = np.array([scores[n] for n in free])
        c = np.array([macs[cmap[n]] / total for n in free])
        plan = {"model": a.model, "base": a.base, "criterion": "modelopt auto_quantize Fisher score, training graph (Y5)",
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
        json.dump(plan, open(fis_path, "w"), indent=1)
    print("AQ_DONE", out_path, flush=True)


if __name__ == "__main__":
    main()

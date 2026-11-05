"""Gate for the factorial control (exploratory, post-freeze): does the executor of the published
SQNR/recon baseline (recon_select.py: exported FP16-activation graph, onnxruntime CPU EP, no FP32
lift) reproduce detector accuracy?

Evaluates AP on the first N tuning images (file_name order, as recon_select.py) for
  ref   every Q/DQ pair bypassed (restore '.*'), i.e. the float reference graph of recon_select
  base  the all-INT8 Q/DQ graph
under (a) CPU EP without lift (as published) and (b) CUDA EP with the FP32 lift (as sim_sensitivity).

Output: metrics/fx_gate_<model>.json (refuses to overwrite).
Usage: python fx_gate_cpu16.py --model yolo11s --n-img 300
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from coco_common import COCO80_TO_91, CONF_FLOOR, decode_output, preprocess  # noqa: E402
from infer_eval import evaluate  # noqa: E402
from qdq_restore import restore  # noqa: E402
from sim_sensitivity import OPT_LEVEL, to_fp32  # noqa: E402


def predict(model, items, provider):
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVEL
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
    s = ort.InferenceSession(model.SerializeToString(), so, providers=[provider])
    assert s.get_providers()[0] == provider, s.get_providers()
    inp, out = s.get_inputs()[0].name, s.get_outputs()[0].name
    preds = []
    for img_id, path in items:
        x, gain, padx, pady, w0, h0 = preprocess(path, 640)
        buf = s.run([out], {inp: x})[0].astype(np.float32)
        for x1, y1, x2, y2, sc, c in decode_output(buf, CONF_FLOOR, gain, padx, pady, w0, h0):
            preds.append({"image_id": int(img_id), "category_id": COCO80_TO_91[c],
                          "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(sc)})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--n-img", type=int, default=300)
    a = ap.parse_args()
    os.chdir(ROOT)
    out = os.path.join("metrics", f"fx_gate_{a.model}.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    gt_path = os.path.join("data", "tune2000", "instances_tune.json")
    gt = json.load(open(gt_path))
    ims = sorted(gt["images"], key=lambda im: im["file_name"])[: a.n_img]
    keep = {im["id"] for im in ims}
    sub = {**gt, "images": ims, "annotations": [x for x in gt["annotations"] if x["image_id"] in keep]}
    sub_path = os.path.join("metrics", f"fx_gate_{a.model}_gt.json")
    json.dump(sub, open(sub_path, "w"))
    items = [(im["id"], os.path.join("data", "tune2000", "images", im["file_name"])) for im in ims]
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    res = {"model": a.model, "n_img": a.n_img, "ort": ort.__version__, "rows": {}}
    for row, rx in (("ref", ".*"), ("base", None)):
        for mode, prov, lift in (("cpu_fp16graph", "CPUExecutionProvider", False),
                                 ("cuda_fp32lift", "CUDAExecutionProvider", True)):
            t0 = time.time()
            m = copy.deepcopy(q)
            if rx is not None:
                restore(m, rx)
            if lift:
                to_fp32(m)
            stats, n = evaluate(sub_path, predict(m, items, prov))
            res["rows"][f"{row}/{mode}"] = {"n": n, "AP": stats["AP"], "AP_small": stats["AP_small"]}
            print(f"{row:5s} {mode:14s} AP={100 * stats['AP']:.2f} S={100 * stats['AP_small']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(out, "w"), indent=1)
    print("GATE_DONE", out, flush=True)


if __name__ == "__main__":
    main()

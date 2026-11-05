"""Simulated-quantization restoration sensitivity (post-freeze control, outside the confirmatory rule).

Same Q/DQ graphs as block_sensitivity.py (all-INT8 base, each block restored alone with
qdq_restore.restore), same tuning images, letterbox, decoder, confidence floor and pycocotools,
but executed by onnxruntime (CUDA EP, Q/DQ nodes run as quantize-dequantize operators) instead
of a compiled TensorRT engine. The exported graph carries FP16 activations; onnxruntime's FP16 path
lost about 10 AP even with every Q/DQ pair bypassed (45.1 against 55.5 for the engine on 300 tuning
images), so the graph is lifted to FP32 (initializers and the input cast) before execution: standard
FP32 fake quantization with the same scales and zero points. Only the executor differs, so allocate.py on the output isolates
what measuring on the engine adds over measuring AP in simulation.

Output: metrics/sens_sim_<model>_<base>.json in the format of block_sensitivity.py (rows -> stats).

Usage: python sim_sensitivity.py --model yolo26l --base int8ent [--only base,fp16_all]
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
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from coco_common import COCO80_TO_91, CONF_FLOOR, decode_output, preprocess  # noqa: E402
from infer_eval import evaluate  # noqa: E402
from qdq_restore import restore  # noqa: E402

OPT_LEVEL = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # constant folding only; no Q/DQ-to-integer-kernel fusion


def to_fp32(m):
    """FP16 initializers and the input cast -> FP32; Q/DQ scales become FP32 with the same values."""
    for t in m.graph.initializer:
        if t.data_type == onnx.TensorProto.FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    for n in m.graph.node:
        if n.op_type == "Cast":
            for at in n.attribute:
                if at.name == "to" and at.i == onnx.TensorProto.FLOAT16:
                    at.i = onnx.TensorProto.FLOAT
    del m.graph.value_info[:]
    return m


def predict(model, items):
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVEL
    s = ort.InferenceSession(model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
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
    ap.add_argument("--only", default="", help="comma list of rows (sanity gate), e.g. int8ent,fp16_all")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--n-img", type=int, default=0, help="first N tuning images by file name (recon_select.py subset)")
    a = ap.parse_args()
    os.chdir(ROOT)
    sens = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))
    out = os.path.join("metrics", f"sens_sim_{a.model}_{a.base}{a.out_tag}.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; pass a new --out-tag")
    gt = sens.get("gt", os.path.join(ROOT, "data", "tune2000", "instances_tune.json"))
    img_dir = os.path.join(ROOT, "data", "tune2000", "images")
    ims = json.load(open(gt))["images"]
    if a.n_img:  # same subset as the output-SQNR and reconstruction scores, scored against its own annotation file
        full = json.load(open(gt))
        ims = sorted(full["images"], key=lambda im: im["file_name"])[: a.n_img]
        keep = {im["id"] for im in ims}
        gt = os.path.join(ROOT, "data", "tune2000", f"instances_tune_first{a.n_img}.json")
        if not os.path.exists(gt):
            json.dump(dict(full, images=ims, annotations=[x for x in full["annotations"] if x["image_id"] in keep]), open(gt, "w"))
    items = sorted((im["id"], os.path.join(img_dir, im["file_name"])) for im in ims)
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    variants = {a.base: None, "fp16_all": "."}
    variants.update(sens["blocks"])
    if a.only:
        variants = {k: v for k, v in variants.items() if k in a.only.split(",")}
    summary = {"model": a.model, "base": a.base, "split": "tune", "gt": gt, "blocks": sens["blocks"],
               "executor": f"onnxruntime {ort.__version__} CUDAExecutionProvider, opt level {OPT_LEVEL.name}, graph lifted to FP32",
               "rows": {}}
    for name, rx in variants.items():
        t0 = time.time()
        m = copy.deepcopy(q)
        if rx is not None:
            restore(m, rx)
        to_fp32(m)
        stats, n = evaluate(gt, predict(m, items))
        eng = sens["rows"][name]["stats"]  # engine on all 2000 tuning images, for reference only
        summary["rows"][name] = {"tag": f"sim_{name}", "n_images": n, "stats": stats, "engine_stats": eng}
        print(f"{name:20s} sim AP={100*stats['AP']:.2f} S={100*stats['AP_small']:.2f} | engine AP={100*eng['AP']:.2f} "
              f"S={100*eng['AP_small']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(summary, open(out, "w"), indent=1)
    print("SIM_DONE", out, flush=True)


if __name__ == "__main__":
    main()

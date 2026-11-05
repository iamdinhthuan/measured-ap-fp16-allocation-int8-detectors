"""Export measured allocations for latency on the Jetson Orin (TensorRT 10.3).

The Jetson parser needs Q/DQ graphs whose high-precision dtype is FP32 (the
FP16 flavour asserts in weightsPtr.h on TRT 10.3), so the base is quantized
again with the same recipe (entropy, the same 512 train2017 calibration images,
Sigmoid excluded) and high_precision_dtype=fp32; the per-tensor scales are
checked against the FP16-flavour base used for every accuracy number. Each
allocation's regex, read from its eval_alloc result, is then restored in this
base. On the device the graphs are built with --int8 --fp16, so Q/DQ-wrapped
layers run INT8 and restored layers FP16.

Usage: python orin_export.py --model yolo11s --tags headout,mean_B0.1,fisher_B0.1
Writes exports/orin/{model}/{tag}.onnx and exports/orin/{model}/manifest.json.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import onnx
from onnx import numpy_helper

from qdq_restore import count_qdq, restore

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from coco_common import COCO_CALIB_DIR, preprocess  # noqa: E402


def scales(path):
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    out = {}
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            out[n.input[0]] = numpy_helper.to_array(inits[n.input[1]]).astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--tags", required=True, help="eval_alloc result tags (metrics/alloc_eval/{model}_{base}/{tag}.json)")
    a = ap.parse_args()
    os.chdir(ROOT)
    src = os.path.join("exports", f"{a.model}.onnx")
    hp32 = os.path.join("exports", f"{a.model}_{a.base}_hp32.onnx")
    if not os.path.exists(hp32):
        from modelopt.onnx.quantization import quantize as mq
        assert a.base == "int8ent"
        files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, "*.jpg")))
        assert len(files) == 512, len(files)
        calib = np.stack([preprocess(f, 640)[0][0] for f in files]).astype(np.float32)
        inp = onnx.load(src, load_external_data=False).graph.input[0].name
        mq(src, quantize_mode="int8", calibration_data={inp: calib}, calibration_method="entropy",
           calibration_eps=["cpu"], output_path=hp32, op_types_to_exclude=["Sigmoid"], high_precision_dtype="fp32")
    s16, s32 = scales(os.path.join("exports", f"{a.model}_{a.base}.onnx")), scales(hp32)
    common = sorted(set(s16) & set(s32))
    rel = max(float(np.max(np.abs(s16[k] - s32[k]) / np.maximum(np.abs(s16[k]), 1e-12))) for k in common)
    outdir = os.path.join("exports", "orin", a.model)
    os.makedirs(outdir, exist_ok=True)
    man_path = os.path.join(outdir, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    man["_base"] = {"hp32": hp32, "quantizers_fp16_flavour": len(s16), "quantizers_hp32": len(s32),
                    "matched": len(common), "max_rel_scale_diff": rel}
    print(f"scales: fp16-flavour {len(s16)} hp32 {len(s32)} matched {len(common)} max rel diff {rel:.2e}", flush=True)
    evdir = os.path.join("metrics", "alloc_eval", f"{a.model}_{a.base}")
    for tag in a.tags.split(","):
        out = os.path.join(outdir, f"{tag}.onnx")
        if tag == "base":
            rx, cost = None, 0.0
        elif tag == "head":  # whole Detect head (last top-level module), the gate's int8ent_head variant
            from allocate import conv_macs
            hi = max(int(k.split("/")[1].split(".")[1]) for k in conv_macs(src) if k.startswith("/model."))
            rx = rf"/model\.{hi}/"
            macs = conv_macs(src)
            cost = sum(v for k, v in macs.items() if re.search(rx, k)) / sum(macs.values())
        else:
            r = json.load(open(os.path.join(evdir, f"{tag}.json")))
            rx, cost = r["regex"], r["cost_macs_frac"]
            if cost is None:  # manual configs did not store their cost: MAC share of the restored Conv nodes
                from allocate import conv_macs
                macs = conv_macs(src)
                cost = sum(v for k, v in macs.items() if re.search(rx, k)) / sum(macs.values())
        m = onnx.load(hp32)
        n_q = count_qdq(m, rx) if rx else 0
        if rx:
            restore(m, rx)
        onnx.save(m, out)
        man[tag] = {"regex": rx, "cost_macs_frac": cost, "qdq_inputs_restored": n_q,
                    "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
        print(f"{tag}: restored {n_q} DQ inputs, MAC {100 * cost:.1f}% -> {out}", flush=True)
    json.dump(man, open(man_path, "w"), indent=1)


if __name__ == "__main__":
    main()

"""RT-DETR adapters for experiment D (rules_11_12d.md). Import this module BEFORE any pipeline script.

The published pipeline (block_sensitivity, allocate, eval_alloc, fx_factorial, sim_sensitivity) is left unchanged;
this module rebinds, in place, the names those scripts look up so that they handle RT-DETR-l:
  coco_common.preprocess     -> stretch resize to 640x640, INTER_LINEAR, no padding (Ultralytics RTDETRPredictor)
  coco_common.decode_output  -> tools/rtdetr_decode.decode_rtdetr(mode="stretch") on the (1,300,6) export
                                [cx,cy,w,h normalised, score, class]; asserts the shape
  allocate.conv_macs         -> MACs of Conv + MatMul + Gemm (transformer layers are MatMul/Gemm)
  block_sensitivity.blocks   -> RT-DETR block split (backbone/neck /model.K/, decoder sub-modules)
  block_sensitivity.run      -> infer_eval.py replaced by infer_eval_rd.py (writes arch="rtdetr")
"""
import os
import re
import sys

import numpy as np
import onnx
from onnx import shape_inference

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import coco_common  # noqa: E402
from rtdetr_decode import decode_rtdetr  # noqa: E402

ARCH = "rtdetr"
OUT_SHAPE = (1, 300, 6)
H = r"^/model\.28/"
# forced set F: the layers that produce the outputs (analogue of the last box/class convs + DFL of the YOLO head)
FORCE_RX = (r"^/model\.28/(?:enc_score_head/|decoder/dec_score_head\.\d+/|"
            r"enc_bbox_head/layers\.2/|decoder/dec_bbox_head\.\d+/layers\.2/)")


def preprocess_stretch(img_path, imgsz):
    import cv2
    im0 = cv2.imread(img_path)
    h0, w0 = im0.shape[:2]
    im = cv2.resize(im0, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    im = im[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0, 1.0, 0.0, 0.0, w0, h0


def decode_rd(out, conf_thr, gain, padx, pady, ow, oh):
    assert tuple(out.shape) == OUT_SHAPE, f"RT-DETR output shape {out.shape} != {OUT_SHAPE}"
    return decode_rtdetr(out, conf_thr, ow, oh, mode="stretch")


def lin_macs(onnx_path):
    """Per-node MACs of Conv, MatMul and Gemm on the shape-inferred static graph."""
    m = shape_inference.infer_shapes(onnx.load(onnx_path))
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
    inits = {i.name: list(i.dims) for i in m.graph.initializer}
    shapes.update(inits)
    out, missing = {}, []
    for n in m.graph.node:
        if n.op_type == "Conv":
            if n.output[0] not in shapes or n.input[1] not in inits:
                missing.append(n.name)
                continue
            co, ci_g, kh, kw = inits[n.input[1]]
            _, _, ho, wo = shapes[n.output[0]]
            out[n.name] = co * ci_g * kh * kw * ho * wo
        elif n.op_type in ("MatMul", "Gemm"):
            o, a = shapes.get(n.output[0]), shapes.get(n.input[0])
            if not o or not a or 0 in o or 0 in a:
                missing.append(n.name)
                continue
            trans_a = any(at.name == "transA" and at.i for at in n.attribute)
            k = a[-2] if (n.op_type == "Gemm" and trans_a) else a[-1]
            out[n.name] = int(np.prod(o)) * k
    if missing:
        raise RuntimeError(f"{len(missing)} Conv/MatMul/Gemm nodes without static shapes, e.g. {missing[:3]}")
    return out


def blocks_rd(model):
    from qdq_restore import count_qdq
    names = [n.name for n in model.graph.node if n.op_type not in ("QuantizeLinear", "DequantizeLinear")]
    ks = sorted({int(m.group(1)) for n in names if (m := re.match(r"/model\.(\d+)/", n))})
    assert ks[-1] == 28, ks
    out = {f"m{k}": rf"^/model\.{k}/" for k in ks if k != 28}
    out["dec.input_proj"] = H + r"input_proj\."
    out["dec.enc_query"] = H + r"(?:enc_output/|enc_bbox_head/layers\.[01]/)"
    out["dec.query_pos"] = H + r"decoder/query_pos_head"
    for i in range(6):
        out[f"dec.layer{i}"] = H + rf"decoder/layers\.{i}/"
    out["dec.box_mlp"] = H + r"decoder/dec_bbox_head\.\d+/layers\.[01]/"
    out["dec.rest"] = (H + r"(?!input_proj\.|enc_output/|enc_score_head/|enc_bbox_head/|decoder/query_pos_head|"
                       r"decoder/layers\.\d/|decoder/dec_score_head|decoder/dec_bbox_head)")
    return {k: v for k, v in out.items() if count_qdq(model, v) > 0}


coco_common.preprocess = preprocess_stretch
coco_common.decode_output = decode_rd

import allocate  # noqa: E402
allocate.conv_macs = lin_macs

import block_sensitivity  # noqa: E402
block_sensitivity.blocks = blocks_rd
_run = block_sensitivity.run


def run_rd(cmd, log):
    if " infer_eval.py " in cmd:
        cmd = cmd.replace(" infer_eval.py ", " infer_eval_rd.py ")
    return _run(cmd, log)


block_sensitivity.run = run_rd

"""COCO predictions from a TensorRT engine ON the Jetson (post-freeze E4, README 11.6).

Same letterbox preprocessing, decoder, NMS and confidence floor as infer_eval.py (tools/coco_common.py
copied verbatim), so val2017 AP of the Orin engines is comparable with the desktop engines. The Jetson
has no cuda-python, so the CUDA runtime is called through ctypes. pycocotools is not needed here:
predictions are written as .pred.json.gz and scored on the server with the same evaluate().

Usage (on the Jetson):
    venv/bin/python orin_ap.py --engine yolo26l/engines/mean_B0.1.plan --gt GT --images DIR --out preds/x.pred.json.gz
    venv/bin/python orin_ap.py --hash-only --gt GT --images DIR --n 50     # preprocessing fingerprint
"""
import argparse
import ctypes
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from coco_common import COCO80_TO_91, CONF_FLOOR, decode_output, preprocess  # noqa: E402

H2D, D2H = 1, 2


def cuda():
    rt = ctypes.CDLL("libcudart.so")
    rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    rt.cudaFree.argtypes = [ctypes.c_void_p]
    rt.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    rt.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    return rt


def check(rc):
    if rc != 0:
        raise RuntimeError(f"CUDA error {rc}")


def predict(engine_path, items):
    import tensorrt as trt
    rt = cuda()
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as r:
        engine = r.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    dev, host = {}, {}
    for n in names:
        shape = tuple(1 if d == -1 else d for d in engine.get_tensor_shape(n))
        host[n] = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(n)))
        p = ctypes.c_void_p()
        check(rt.cudaMalloc(ctypes.byref(p), host[n].nbytes))
        dev[n] = p
        ctx.set_tensor_address(n, p.value)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(n, shape)
    inp = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    out = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
    stream = ctypes.c_void_p()
    check(rt.cudaStreamCreate(ctypes.byref(stream)))
    preds = []
    for k, (img_id, path) in enumerate(items):
        x, gain, padx, pady, w0, h0 = preprocess(path, 640)
        x = np.ascontiguousarray(x, dtype=host[inp].dtype)
        check(rt.cudaMemcpy(dev[inp], x.ctypes.data, x.nbytes, H2D))
        if not ctx.execute_async_v3(stream_handle=stream.value):
            raise RuntimeError("execute failed")
        check(rt.cudaStreamSynchronize(stream))
        buf = host[out]
        check(rt.cudaMemcpy(buf.ctypes.data, dev[out], buf.nbytes, D2H))
        for x1, y1, x2, y2, s, c in decode_output(buf, CONF_FLOOR, gain, padx, pady, w0, h0):
            preds.append({"image_id": int(img_id), "category_id": COCO80_TO_91[c],
                          "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
        if k % 1000 == 0:
            print(f"{k}/{len(items)}", flush=True)
    for p in dev.values():
        rt.cudaFree(p)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out")
    ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    gt = json.load(open(a.gt))
    items = sorted((im["id"], os.path.join(a.images, im["file_name"])) for im in gt["images"])
    if a.n:
        items = items[: a.n]
    if a.hash_only:
        h = hashlib.sha256()
        for _, p in items:
            x, *meta = preprocess(p, 640)
            h.update(x.tobytes())
            h.update(json.dumps([float(v) for v in meta]).encode())
        print("PREPROC_SHA256", len(items), h.hexdigest())
        return
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    preds = predict(a.engine, items)
    blob = json.dumps(preds).encode()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with gzip.open(a.out, "wb") as f:
        f.write(blob)
    print("ORIN_PRED_DONE", a.out, len(preds), hashlib.sha256(blob).hexdigest(), flush=True)


if __name__ == "__main__":
    main()

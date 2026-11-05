"""TensorRT engine -> COCO size-stratified AP on an arbitrary GT json, in one process.

Same letterbox preprocessing, decoder, NMS and confidence floor as
tools/coco_infer_trt.py (imported, not copied), so numbers are comparable to
the canonical benchmark; only the image list and GT file are parameters.

Usage:
    python infer_eval.py --engine x.plan --gt data/tune2000/instances_tune.json \
        --images data/tune2000/images --out metrics/tune/x.json [--save-pred]
"""
import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import sys

import numpy as np
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from coco_common import COCO80_TO_91, CONF_FLOOR, decode_output, preprocess  # noqa: E402
from coco_infer_trt import cuda_check, cudart  # noqa: E402

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def predict(engine_path, items, imgsz=640, conf=CONF_FLOOR):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    dev, host = {}, {}
    for n in names:
        shape = tuple(1 if d == -1 else d for d in engine.get_tensor_shape(n))
        host[n] = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(n)))
        err, ptr = cudart.cudaMalloc(host[n].nbytes)
        cuda_check(err)
        dev[n] = ptr
        ctx.set_tensor_address(n, ptr)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(n, shape)
    inp = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    out = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)
    preds = []
    for img_id, path in items:
        x, gain, padx, pady, w0, h0 = preprocess(path, imgsz)
        cuda_check(cudart.cudaMemcpyAsync(dev[inp], x.ctypes.data, x.nbytes,
                                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)[0])
        ctx.execute_async_v3(stream_handle=stream)
        buf = host[out]
        cuda_check(cudart.cudaMemcpyAsync(buf.ctypes.data, dev[out], buf.nbytes,
                                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)[0])
        cudart.cudaStreamSynchronize(stream)
        for x1, y1, x2, y2, s, c in decode_output(buf, conf, gain, padx, pady, w0, h0):
            preds.append({"image_id": int(img_id), "category_id": COCO80_TO_91[c],
                          "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    for p in dev.values():
        cudart.cudaFree(p)
    return preds


def evaluate(gt_path, preds):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
        e = COCOeval(gt, gt.loadRes(preds), iouType="bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    return {k: float(v) for k, v in zip(STAT_NAMES, e.stats)}, len(e.params.imgIds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-pred", action="store_true")
    a = ap.parse_args()
    gt = json.load(open(a.gt))
    items = sorted((im["id"], os.path.join(a.images, im["file_name"])) for im in gt["images"])
    missing = [p for _, p in items if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[0]}")
    preds = predict(a.engine, items)
    stats, n = evaluate(a.gt, preds)
    blob = json.dumps(preds).encode()
    res = {"engine": a.engine, "gt": a.gt, "n_images": n, "n_det": len(preds), "stats": stats,
           "prediction_sha256": hashlib.sha256(blob).hexdigest()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.save_pred:
        with gzip.open(a.out.replace(".json", ".pred.json.gz"), "wb") as f:
            f.write(blob)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(100 * stats[k], 2) for k in ("AP", "AP_small", "AP_medium", "AP_large")}))


if __name__ == "__main__":
    main()

"""Engine-in-the-loop restoration sensitivity per block and per object-size group.

Starting from a fully quantized INT8 Q/DQ graph, each block is restored to
FP16 alone (qdq_restore.restore), compiled to a native TensorRT engine and
evaluated on the train2017 tuning split. Output per block: AP, AP_S/M/L and
engine latency, so that r[b, g] = AP_g(restore b) - AP_g(all INT8).

Usage (workdir root):
    python block_sensitivity.py --model yolo11s --base int8ent [--split exdark]

--split exdark measures on the ExDark train split instead (zero-shot COCO
detector, same Q/DQ graphs); outputs go to metrics/tune_exdark/ and
metrics/sens_<model>_<base>_exdark.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import onnx
try:
    import tensorrt as trt
except ImportError:  # blocks() is usable without TensorRT (tests, local inspection)
    trt = None

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from qdq_restore import count_qdq, restore  # noqa: E402

PY = sys.executable
SPLITS = {
    "tune": (os.path.join(ROOT, "data", "tune2000", "instances_tune.json"),
             os.path.join(ROOT, "data", "tune2000", "images"), "tune", ""),
    "exdark": (os.path.join(ROOT, "data", "exdark", "exdark_train.json"),
               os.path.join(ROOT, "data", "exdark_raw", "ExDark"), "tune_exdark", "_exdark"),
}


def blocks(model):
    """Top-level /model.K/ modules; the Detect head is split per level and branch."""
    names = [n.name for n in model.graph.node if n.op_type not in ("QuantizeLinear", "DequantizeLinear")]
    ks = sorted({int(m.group(1)) for n in names if (m := re.match(r"/model\.(\d+)/", n))})
    head = ks[-1]
    out = {f"m{k}": rf"^/model\.{k}/" for k in ks if k != head}
    subs = sorted({m.group(1) for n in names if (m := re.match(rf"/model\.{head}/((?:one2one_)?cv[23]\.\d|dfl)/", n))})
    for s in subs:
        out[f"head.{s}"] = rf"^/model\.{head}/{re.escape(s)}/"
    rest = rf"^/model\.{head}/(?!(?:one2one_)?cv[23]\.\d/|dfl/)"
    out["head.rest"] = rest
    return {k: v for k, v in out.items() if count_qdq(model, v) > 0}


def latency_ms(plan, iters=300, warmup=50):
    """Median GPU latency of one batch-1 inference (cudaEvents around execute)."""
    from coco_infer_trt import cuda_check, cudart
    logger = trt.Logger(trt.Logger.WARNING)
    with open(plan, "rb") as f, trt.Runtime(logger) as rt:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shape = tuple(1 if d == -1 else d for d in eng.get_tensor_shape(n))
        size = int(np.prod(shape)) * np.dtype(trt.nptype(eng.get_tensor_dtype(n))).itemsize
        err, ptr = cudart.cudaMalloc(size)
        cuda_check(err)
        ctx.set_tensor_address(n, ptr)
    _, stream = cudart.cudaStreamCreate()
    _, e0 = cudart.cudaEventCreate()
    _, e1 = cudart.cudaEventCreate()
    ts = []
    for k in range(warmup + iters):
        cudart.cudaEventRecord(e0, stream)
        ctx.execute_async_v3(stream_handle=stream)
        cudart.cudaEventRecord(e1, stream)
        cudart.cudaEventSynchronize(e1)
        if k >= warmup:
            ts.append(cudart.cudaEventElapsedTime(e0, e1)[1])
    return float(np.median(ts))


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        if subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode:
            raise SystemExit(f"FAILED: {cmd}")


def evaluate_variant(tag, onnx_path, a, log, keep_plan=False):
    gt, img, mdir, _ = SPLITS[a.split]
    out = os.path.join(ROOT, "metrics", mdir, a.model, f"{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        return json.load(open(out))
    plan = os.path.join(ROOT, "exports", "tmp", f"{a.model}_{tag}.plan")
    os.makedirs(os.path.dirname(plan), exist_ok=True)
    t0 = time.time()
    run(f"{PY} tools/build_engine.py --onnx {onnx_path} --engine {plan}", log)
    build_s = time.time() - t0
    run(f"{PY} infer_eval.py --engine {plan} --gt {gt} --images {img} --out {out} --save-pred", log)
    res = json.load(open(out))
    res.update({"tag": tag, "onnx": onnx_path, "build_s": round(build_s, 1), "latency_ms": latency_ms(plan),
                "engine_bytes": os.path.getsize(plan)})
    json.dump(res, open(out, "w"), indent=1)
    keep_plan or os.remove(plan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent", help="fully quantized variant from gate_headroom.py")
    ap.add_argument("--split", default="tune", choices=sorted(SPLITS))
    a = ap.parse_args()
    os.chdir(ROOT)
    sfx = SPLITS[a.split][3]
    log = os.path.join(ROOT, "logs", f"sens_{a.model}_{a.base}{sfx}.log")
    base_onnx = os.path.join(ROOT, "exports", f"{a.model}_{a.base}.onnx")
    fp32_onnx = os.path.join(ROOT, "exports", f"{a.model}.onnx")
    tmp = os.path.join(ROOT, "exports", "tmp")
    os.makedirs(tmp, exist_ok=True)
    m = onnx.load(base_onnx)
    bl = blocks(m)
    summary = {"model": a.model, "base": a.base, "split": a.split, "gt": SPLITS[a.split][0], "blocks": bl, "rows": {}}
    summary["rows"]["fp32"] = evaluate_variant("fp32", fp32_onnx, a, log)
    summary["rows"][a.base] = evaluate_variant(a.base, base_onnx, a, log)
    allf = os.path.join(tmp, f"{a.model}_{a.base}_restore_all.onnx")
    if not os.path.exists(allf):
        mm = onnx.load(base_onnx)
        restore(mm, ".")
        onnx.save(mm, allf)
    summary["rows"]["fp16_all"] = evaluate_variant(f"{a.base}_restore_all", allf, a, log)
    for name, rx in bl.items():
        tag = f"{a.base}_restore_{name}"
        path = os.path.join(tmp, f"{a.model}_{tag}.onnx")
        if not os.path.exists(path):
            mm = onnx.load(base_onnx)
            n = restore(mm, rx)
            onnx.save(mm, path)
        r = evaluate_variant(tag, path, a, log)
        r["qdq_inputs"] = count_qdq(m, rx)
        summary["rows"][name] = r
        os.remove(path)
        b = summary["rows"][a.base]["stats"]
        print(f"{name:14s} dAP={100*(r['stats']['AP']-b['AP']):+.2f} dS={100*(r['stats']['AP_small']-b['AP_small']):+.2f} "
              f"dL={100*(r['stats']['AP_large']-b['AP_large']):+.2f} lat={r['latency_ms']:.3f}", flush=True)
        json.dump(summary, open(os.path.join(ROOT, "metrics", f"sens_{a.model}_{a.base}{sfx}.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(ROOT, "metrics", f"sens_{a.model}_{a.base}{sfx}.json"), "w"), indent=1)
    print("SENS_DONE", flush=True)


if __name__ == "__main__":
    main()

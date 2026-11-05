"""Gate 1b: how much INT8 headroom is left once the INT8 recipe is strong?

Native TensorRT 11.1 strongly-typed engines on the RTX 5090, ModelOpt ONNX
explicit Q/DQ, the 512 train2017 calibration images used by the canonical
benchmark, full val2017. Exploratory only: val2017 is used to see headroom,
never to pick the recipe that a later method is compared against.

Usage (from the workdir root):
    python gate_headroom.py --model yolo11s [--limit 0]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import onnx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from coco_common import COCO_CALIB_DIR, preprocess  # noqa: E402

PY = sys.executable


def head_nodes(onnx_path, pattern):
    g = onnx.load(onnx_path, load_external_data=False).graph
    return [n.name for n in g.node if re.search(pattern, n.name)]


# name -> (calibration_method, node-name regex to keep in high precision or None)
VARIANTS = {
    "int8max": ("max", None),
    "int8ent": ("entropy", None),
    # last box/cls convs of every level + DFL (the head's output-producing layers)
    "int8max_headout": ("max", r"/model\.\d+/(cv2\.\d/cv2\.\d\.2|cv3\.\d/cv3\.\d\.2|dfl)/"),
    "int8ent_headout": ("entropy", r"/model\.\d+/(cv2\.\d/cv2\.\d\.2|cv3\.\d/cv3\.\d\.2|dfl)/"),
    # whole Detect head (last top-level module) in high precision
    "int8ent_head": ("entropy", "HEAD"),
}


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from modelopt.onnx.quantization import quantize as mq

    src = os.path.join(ROOT, "exports", f"{a.model}.onnx")
    g = onnx.load(src, load_external_data=False).graph
    inp = g.input[0].name
    head_idx = max(int(m.group(1)) for n in g.node if (m := re.match(r"/model\.(\d+)/", n.name)))
    files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, "*.jpg")))
    assert len(files) == 512, len(files)
    calib = np.stack([preprocess(f, 640)[0][0] for f in files]).astype(np.float32)
    log = os.path.join(ROOT, "logs", f"gate_{a.model}.log")
    summary_path = os.path.join(ROOT, "metrics", f"gate_{a.model}.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    for tag in ["fp32"] + a.variants.split(","):
        onnx_out = src if tag == "fp32" else os.path.join(ROOT, "exports", f"{a.model}_{tag}.onnx")
        if tag != "fp32" and not os.path.exists(onnx_out):
            method, pat = VARIANTS[tag]
            if pat == "HEAD":
                pat = rf"/model\.{head_idx}/"
            excl = head_nodes(src, pat) if pat else []
            t0 = time.time()
            mq(src, quantize_mode="int8", calibration_data={inp: calib}, calibration_method=method,
               calibration_eps=["cpu"], output_path=onnx_out, op_types_to_exclude=["Sigmoid"],
               nodes_to_exclude=excl or None)
            summary.setdefault(tag, {})["excluded_nodes"] = len(excl)
            summary[tag]["quantize_s"] = round(time.time() - t0, 1)
        plan = onnx_out.replace(".onnx", ".plan") if tag != "fp32" else os.path.join(ROOT, "exports", f"{a.model}_fp32.plan")
        os.path.exists(plan) or run(f"{PY} tools/build_engine.py --onnx {onnx_out} --engine {plan}", log)
        pred = os.path.join(ROOT, "metrics", f"pred_{a.model}_{tag}.json")
        os.path.exists(pred) or run(f"{PY} tools/coco_infer_trt.py --engine {plan} --out {pred} --limit {a.limit}", log)
        ev = os.path.join(ROOT, "metrics", f"{a.model}_{tag}.json")
        extra = "--img-ids-from-dt" if a.limit else ""
        os.path.exists(ev) or run(f"{PY} tools/coco_eval_pilot.py --dt {pred} --model {a.model} --precision {tag} --out {ev} {extra}", log)
        st = json.load(open(ev))["stats"]
        summary.setdefault(tag, {}).update({k: st[k] for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")})
        summary[tag]["n_images"] = json.load(open(ev))["n_images"]
        os.path.exists(plan) and tag != "fp32" and os.remove(plan)  # engines are cheap to rebuild; keep disk low
        json.dump(summary, open(summary_path, "w"), indent=1)
        print(tag, {k: round(v * 100, 2) for k, v in summary[tag].items() if k.startswith("AP")}, flush=True)

    ref = summary["fp32"]
    for tag, s in summary.items():
        if tag == "fp32":
            continue
        loss = {k: ref[k] - s[k] for k in ("AP", "AP_small", "AP_large")}
        print(f"{tag:18s} loss={loss['AP']*100:+.2f} DIFF={(loss['AP_small']-loss['AP_large'])*100:+.2f} (AP points)")


if __name__ == "__main__":
    main()

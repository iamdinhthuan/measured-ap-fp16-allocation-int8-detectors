"""X3 (README 11.9, rules_11_9.md): simulated AP x r sweep on another tuning draw (exploratory, post-freeze).

Identical to sim_sensitivity.py (FP32-lifted Q/DQ graph, onnxruntime CUDA, each block restored alone,
same decoder and pycocotools) except that the images and labels come from --tune-dir.
Output (never overwritten): metrics/sens_sim_<model>_<base>_<tag>.json (format of sens_sim_*.json; allocate.py reads it).
Usage: python sim_sens_draw.py --model yolo26l --tune-dir data/tune_d1 --tag d1
"""
import argparse
import copy
import json
import os
import sys
import time

import onnx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from infer_eval import evaluate  # noqa: E402
from qdq_restore import restore  # noqa: E402
from sim_sensitivity import OPT_LEVEL, predict, to_fp32  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--tune-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    out = os.path.join("metrics", f"sens_sim_{a.model}_{a.base}_{a.tag}.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    sens = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))
    gt = os.path.join(a.tune_dir, "instances_tune.json")
    img_dir = os.path.join(a.tune_dir, "images")
    items = sorted((im["id"], os.path.join(img_dir, im["file_name"])) for im in json.load(open(gt))["images"])
    missing = [p for _, p in items if not os.path.exists(p)]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:2]}"
    q = onnx.load(os.path.join("exports", f"{a.model}_{a.base}.onnx"))
    variants = {a.base: None, "fp16_all": "."}
    variants.update(sens["blocks"])
    summary = {"model": a.model, "base": a.base, "split": f"tune_{a.tag}", "gt": gt, "blocks": sens["blocks"],
               "executor": f"onnxruntime CUDAExecutionProvider, opt level {OPT_LEVEL.name}, graph lifted to FP32",
               "rows": {}}
    for name, rx in variants.items():
        t0 = time.time()
        m = copy.deepcopy(q)
        if rx is not None:
            restore(m, rx)
        to_fp32(m)
        stats, n = evaluate(gt, predict(m, items))
        summary["rows"][name] = {"tag": f"sim_{a.tag}_{name}", "n_images": n, "stats": stats}
        print(f"{name:20s} sim AP={100 * stats['AP']:.2f} S={100 * stats['AP_small']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(summary, open(out, "w"), indent=1)
    print("SIMDRAW_DONE", out, flush=True)


if __name__ == "__main__":
    main()

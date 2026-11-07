"""rules_11_12e.md: TensorRT build reproducibility on YOLO11-l / YOLO26-l (4 builds per graph, first 200 tuning images)."""
import copy
import json
import os
import subprocess
import sys

import onnx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from qdq_restore import restore  # noqa: E402

P = sys.executable
OUT = os.path.join("metrics", "det_yolo.json")
GT = os.path.join("data", "tune2000", "instances_tune_first200.json")
IMG = os.path.join("data", "tune2000", "images")
TMP = os.path.join("exports", "tmp", "dety")
os.makedirs(TMP, exist_ok=True)
assert os.path.exists(GT)
res = json.load(open(OUT)) if os.path.exists(OUT) else {"rules": "rules_11_12e.md", "graphs": {}}
graphs = {}
for m in ("yolo11l", "yolo26l"):
    base = onnx.load(os.path.join("exports", f"{m}_int8ent.onnx"))
    graphs[f"{m}_int8"] = os.path.join("exports", f"{m}_int8ent.onnx")
    p = os.path.join(TMP, f"{m}_mean01.onnx")
    if not os.path.exists(p):
        mm = copy.deepcopy(base)
        restore(mm, json.load(open(f"metrics/alloc_eval/{m}_int8ent/mean_B0.1.json"))["regex"])
        onnx.save(mm, p)
    graphs[f"{m}_mean01"] = p


def sh(cmd):
    print("$", cmd, flush=True)
    if subprocess.call(cmd, shell=True):
        raise SystemExit(f"FAILED: {cmd}")


for name, p in graphs.items():
    g = res["graphs"].setdefault(name, {"onnx": p, "builds": {}})
    for k in range(1, 5):
        if str(k) in g["builds"]:
            continue
        eng = os.path.join(TMP, f"{name}_b{k}.plan")
        out = os.path.join(TMP, f"{name}_b{k}.json")
        sh(f"{P} tools/build_engine.py --onnx {p} --engine {eng}")
        if not os.path.exists(out):
            sh(f"{P} infer_eval.py --engine {eng} --gt {GT} --images {IMG} --out {out}")
        g["builds"][str(k)] = 100 * json.load(open(out))["stats"]["AP"]
        os.remove(eng)
        json.dump(res, open(OUT, "w"), indent=1)
    v = list(g["builds"].values())
    g["range"] = max(v) - min(v)
res["max_range"] = max(g["range"] for g in res["graphs"].values())
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({n: [round(x, 2) for x in g["builds"].values()] for n, g in res["graphs"].items()}),
      "max_range", round(res["max_range"], 3), flush=True)

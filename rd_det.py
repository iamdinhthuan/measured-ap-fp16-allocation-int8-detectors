"""rules_11_12d_add2.md A5: TensorRT build reproducibility on RT-DETR-l (4 builds per graph, first 200 tuning images)."""
import copy
import json
import os
import subprocess
import sys

import onnx

import rd_common
from qdq_restore import restore

os.chdir(rd_common.ROOT)
P = sys.executable
OUT = os.path.join("metrics", "det_rtdetr-l.json")
GT = os.path.join("data", "tune2000", "instances_tune_first200.json")
IMG = os.path.join("data", "tune2000", "images")
TMP = os.path.join("exports", "tmp", "det")
os.makedirs(TMP, exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {"rules": "rules_11_12d_add2.md A5", "graphs": {}}
base = onnx.load(os.path.join("exports", "rtdetr-l_int8ent.onnx"))
rx_F = json.load(open("metrics/alloc_eval/rtdetr-l_int8ent/F_B0.json"))["regex"]
rx_s = json.load(open("metrics/alloc_eval/rtdetr-l_int8ent/fx2000_sqnrq_B0.03.json"))["regex"]
graphs = {"g1_fp32": os.path.join("exports", "rtdetr-l.onnx")}
for name, rx in (("g2_int8F", rx_F), ("g3_sqnr003", rx_s), ("g4_int8", None)):
    p = os.path.join(TMP, f"{name}.onnx")
    if not os.path.exists(p):
        m = copy.deepcopy(base)
        if rx:
            restore(m, rx)
        onnx.save(m, p)
    graphs[name] = p


def sh(cmd):
    print("$", cmd, flush=True)
    if subprocess.call(cmd, shell=True):
        raise SystemExit(f"FAILED: {cmd}")


def ev(eng, tag):
    out = os.path.join(TMP, f"{tag}.json")
    if not os.path.exists(out):
        sh(f"{P} infer_eval_rd.py --engine {eng} --gt {GT} --images {IMG} --out {out}")
    r = json.load(open(out))
    assert r["arch"] == rd_common.ARCH
    return 100 * r["stats"]["AP"] if "stats" in r else 100 * r["AP"]


for name, p in graphs.items():
    g = res["graphs"].setdefault(name, {"onnx": p, "builds": {}})
    for k in range(1, 5):
        if str(k) in g["builds"]:
            continue
        eng = os.path.join(TMP, f"{name}_b{k}.plan")
        sh(f"{P} tools/build_engine.py --onnx {p} --engine {eng}")
        ap = ev(eng, f"{name}_b{k}")
        rep = ev(eng, f"{name}_b{k}_rep") if k == 1 else None
        g["builds"][str(k)] = {"AP": ap, "runtime_repeat_AP": rep}
        os.remove(eng)
        json.dump(res, open(OUT, "w"), indent=1)
    aps = [b["AP"] for b in g["builds"].values()]
    g["range"] = max(aps) - min(aps)
    json.dump(res, open(OUT, "w"), indent=1)
q = [res["graphs"][n]["range"] for n in ("g2_int8F", "g3_sqnr003", "g4_int8")]
res["reproducible"] = max(q) <= 1.0
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({n: [round(b["AP"], 2) for b in g["builds"].values()] for n, g in res["graphs"].items()}),
      "reproducible:", res["reproducible"], flush=True)

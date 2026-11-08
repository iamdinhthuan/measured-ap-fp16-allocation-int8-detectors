"""X1 (rules_20261108_x1.md, written before any result): exact incremental cost c_b^F = MAC(b minus F).
Solver-only re-analysis of published sensitivities. Output (never overwritten): metrics/x1_exactcost.json."""
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from allocate import conv_macs, solve  # noqa: E402

OUT = os.path.join("metrics", "x1_exactcost.json")
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists")
TMP = os.path.join("exports", "x1_tmp")
BUDGETS = (0.03, 0.05, 0.1, 0.15, 0.2)
GROUPS = ("AP_small", "AP_medium", "AP_large")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = {"rules": "rules_20261108_x1.md", "models": {}}


def onnx_for(m):
    src = os.path.join("exports", f"{m}.onnx")
    if os.path.exists(src):
        return src, False
    from ultralytics import YOLO
    d = os.path.join(TMP, m)
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join("exports", f"{m}.pt"), d)
    f = YOLO(os.path.join(d, f"{m}.pt")).export(format="onnx", opset=17, simplify=True, batch=1, dynamic=False,
                                                nms=False, imgsz=640)
    return f, True


def gaps(m):
    s = json.load(open(os.path.join("metrics", f"sens_{m}_int8ent.json")))
    st = {k: v["stats"] for k, v in s["rows"].items()}
    base, fp16 = st[s["base"]], st["fp16_all"]
    return (np.array([100 * (fp16[k] - base[k]) for k in GROUPS]), np.array([100 * fp16[k] for k in GROUPS]), s["blocks"])


for m in ("yolo11s", "yolo26s", "yolo11l", "yolo26l"):
    R = {"pt_sha256": sha(os.path.join("exports", f"{m}.pt"))}
    pub = json.load(open(os.path.join("metrics", f"alloc_{m}_int8ent_ho.json")))
    base_gap, scale, rx = gaps(m)
    names = pub["blocks"]
    path, regenerated = onnx_for(m)
    R["onnx_regenerated"] = regenerated
    macs = conv_macs(path)
    total = sum(macs.values())
    frx = pub["force_regex"]
    cost = np.array([sum(v for k, v in macs.items() if re.search(rx[b], k)) / total for b in names])
    fcost = sum(v for k, v in macs.items() if re.search(frx, k)) / total
    cexact = np.array([sum(v for k, v in macs.items() if re.search(rx[b], k) and not re.search(frx, k)) / total
                       for b in names])
    g1 = bool(np.max(np.abs(cost - np.array(pub["cost"]))) < 1e-9 and abs(fcost - pub["force_cost"]) < 1e-9)
    R["G1"] = g1
    R["G1_max_abs_err"] = float(max(np.max(np.abs(cost - np.array(pub["cost"]))), abs(fcost - pub["force_cost"])))
    R["overlap"] = {b: float(cost[i] - cexact[i]) for i, b in enumerate(names) if cost[i] - cexact[i] > 0}
    R["force_cost"] = fcost
    print(m, "G1", g1, R["G1_max_abs_err"], "overlap blocks", sorted(R["overlap"]), flush=True)
    if not g1:
        rec["models"][m] = R
        continue
    arms = [("published", np.array(pub["r"]), pub["allocations"])]
    fsw = os.path.join("configs", f"fx_{m}_int8ent_ho_fsweep.json")
    if os.path.exists(fsw):
        f = json.load(open(fsw))
        assert f["blocks"] == names
        arms.append(("fsweep", np.array(f["r"]), f["allocations"]))
    R["G2"] = True
    R["arms"] = {}
    for arm, r, allocs in arms:
        A = {}
        for B in BUDGETS:
            for obj in ("mean", "minimax"):
                key = f"{obj}@{B}"
                if key not in allocs:
                    continue
                x0 = solve(r, cost, B - fcost, base_gap, scale, obj)
                sel0 = [names[i] for i in np.flatnonzero(x0)]
                ok = sorted(sel0) == sorted(allocs[key]["blocks"])
                R["G2"] &= ok
                x1 = solve(r, cexact, B - fcost, base_gap, scale, obj)
                sel1 = [names[i] for i in np.flatnonzero(x1)]
                A[key] = {"reproduced": ok, "changed": sorted(sel1) != sorted(sel0), "published": sel0, "exact": sel1,
                          "exact_mac_published": float(fcost + cexact[x0 == 1].sum()),
                          "exact_mac_x1": float(fcost + cexact[x1 == 1].sum()),
                          "pred_dAP_tune": float(r[x1 == 1, 0].sum() - r[x0 == 1, 0].sum())}
                print(m, arm, key, "repro" if ok else "NOT_REPRO", "CHANGED" if A[key]["changed"] else "same",
                      f"mac {A[key]['exact_mac_published']:.4f}->{A[key]['exact_mac_x1']:.4f}",
                      f"pred dAP {A[key]['pred_dAP_tune']:+.3f}", flush=True)
        R["arms"][arm] = A
    rec["models"][m] = R
shutil.rmtree(TMP, ignore_errors=True)
json.dump(rec, open(OUT, "w"), indent=1)
print("X1_DONE", OUT, flush=True)

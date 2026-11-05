"""Phase B pilot on yolo26l (README 11.7, rule written before any engine was built).

Inputs (copied from the Orin / server):
  bench.jsonl      same-session timings (3 interleaved rounds) of sb_*, random_*, held-out, headout, base, fp16
  profiles/        per-layer profiles of sb_* and headout
  features.json    metrics/phaseB_pilot_features_yolo26l.json (boundary_features on every graph)
  plans.json       {config: [blocks]} for the held-out configurations
  old_bench.jsonl  yolo26l/bench.jsonl of 26-10 (drift check)
Usage: python fit_pilot.py DIR
"""
import collections
import json
import os
import re
import sys

import numpy as np

D = sys.argv[1]
HELD = ["head", "mean_B0.05", "mean_B0.1", "mean_B0.2", "minimax_B0.1",
        "random_B0.1_s1", "random_B0.1_s2", "random_B0.2_s1", "random_B0.2_s2"]


def medians(path):
    d = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("median_ms") is not None:
            d[r["engine"]].append(r["median_ms"])
    return d


reps = medians(os.path.join(D, "bench.jsonl"))
lat = {k: float(np.median(v)) for k, v in reps.items()}
spread = [100 * (max(v) - min(v)) / np.median(v) for v in reps.values() if len(v) >= 2]
print(f"engines {len(lat)}; reps per engine {sorted(set(len(v) for v in reps.values()))}; "
      f"noise floor (range/median) median {np.median(spread):.2f}% p90 {np.percentile(spread, 90):.2f}%")
base, ho, fp16 = lat["base"], lat["headout"], lat["fp16"]
y = {k: 100 * (v - base) / base for k, v in lat.items()}

old = {k: float(np.median(v)) for k, v in medians(os.path.join(D, "old_bench.jsonl")).items()}
print("drift vs 26-10 (same engine files, % of lat):")
for k in ["base", "headout", "fp16", "stem_headout"] + HELD[:5]:
    if k in old:
        print(f"  {k:14s} {old[k]:.3f} -> {lat[k]:.3f} ms ({100 * (lat[k] - old[k]) / old[k]:+.2f}%)")
print(f"build-to-build: sb_m0 {lat['sb_m0']:.3f} vs stem_headout {lat['stem_headout']:.3f} ms "
      f"({100 * (lat['sb_m0'] - lat['stem_headout']) / lat['stem_headout']:+.2f}%)")

feat = json.load(open(os.path.join(D, "features.json")))
plans = json.load(open(os.path.join(D, "plans.json")))
sb = sorted(k[3:] for k in lat if k.startswith("sb_"))
a = {b: lat[f"sb_{b}"] - ho for b in sb}


def bnd_time(name):
    p = [r for r in json.load(open(os.path.join(D, "profiles", f"{name}.profile.json"))) if "name" in r]
    t = 0.0
    for r in p:
        n = r["name"]
        if "Reformatting" in n or (re.search(r"(Quantize|Dequantize)Linear", n) and "Conv" not in n
                                   and not n.endswith(" copy")):
            t += r["averageMs"]
    return t


B_ho = bnd_time("headout")
dB = np.array([bnd_time(f"sb_{b}") - B_ho for b in sb])
dX = np.array([feat[f"sb_{b}"]["Bytes"] - feat["headout"]["Bytes"] for b in sb])
kappa = float(dX @ dB / (dX @ dX)) if dX @ dX > 0 else 0.0
r_k = np.corrcoef(dX, dB)[0, 1] if np.std(dX) > 0 and np.std(dB) > 0 else float("nan")
print(f"kappa = {kappa:.4f} ms per unit Bytes fraction (r = {r_k:.2f}, {len(sb)} sb engines)")

# M_mac: y ~ mac on sb_* and headout
X = np.array([[1, feat[f"sb_{b}"]["mac"]] for b in sb] + [[1, feat["headout"]["mac"]]])
Y = np.array([y[f"sb_{b}"] for b in sb] + [y["headout"]])
cm = np.linalg.lstsq(X, Y, rcond=None)[0]


def pct(ms):
    return 100 * (ms - base) / base


rows = []
for k in HELD:
    S = plans[k]
    add = ho + sum(a[b] for b in S)
    bnd = add - kappa * sum(feat[f"sb_{b}"]["Bytes"] - feat["headout"]["Bytes"] for b in S) \
        + kappa * (feat[k]["Bytes"] - feat["headout"]["Bytes"])
    rows.append((k, y[k], pct(bnd), pct(add), cm[0] + cm[1] * feat[k]["mac"], y["headout"]))
print(f"\n{'config':15s} {'meas':>7s} {'M_bnd':>7s} {'M_add':>7s} {'M_mac':>7s} {'const':>7s}   (y, % of INT8 base; FP16 = {y['fp16']:+.2f})")
for r in rows:
    print(f"{r[0]:15s} " + " ".join(f"{v:+7.2f}" for v in r[1:]))
R = np.array([r[1:] for r in rows])
mae = {m: float(np.mean(np.abs(R[:, i] - R[:, 0]))) for i, m in enumerate(["M_bnd", "M_add", "M_mac", "const"], 1)}
fp = y["fp16"]
cls = {m: int(np.sum((R[:, i] < fp) == (R[:, 0] < fp))) for i, m in enumerate(["M_bnd", "M_add", "M_mac", "const"], 1)}
print("\nMAE (points of %):", {m: round(v, 2) for m, v in mae.items()})
print("faster-than-FP16 classified correctly (of 9):", cls)
ref = min(mae["const"], mae["M_mac"])
for m in ("M_bnd", "M_add"):
    ok = mae[m] <= 1.0 and mae[m] <= 0.5 * ref and cls[m] == len(HELD)
    print(f"{m}: MAE {mae[m]:.2f} <= 1.0? {mae[m] <= 1.0}; <= 0.5 x {ref:.2f}? {mae[m] <= 0.5 * ref}; "
          f"classification {cls[m]}/{len(HELD)} -> {'PASS' if ok else 'FAIL'}")
json.dump({"lat": lat, "y": y, "a": a, "kappa": kappa, "kappa_r": r_k, "mae": mae, "cls": cls,
           "rows": [list(r) for r in rows]}, open(os.path.join(D, "pilot_result.json"), "w"), indent=1)

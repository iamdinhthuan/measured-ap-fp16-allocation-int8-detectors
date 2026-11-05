import json, sys, numpy as np, collections
Q = sys.argv[1]
F = json.load(open(f"{Q}/bench/phaseB_boundary_features.json"))
rows = []
noise = []
for m in ["yolo11s", "yolo26s", "yolo26l"]:
    lat = collections.defaultdict(list)
    for l in open(f"{Q}/bench/{m}.jsonl"):
        d = json.loads(l)
        if "median_ms" in d: lat[d["engine"]].append(d["median_ms"])
    med = {k: float(np.median(v)) for k, v in lat.items()}
    for k, v in lat.items():
        noise.append(100 * (max(v) - min(v)) / np.median(v))
    b = med["base"]
    for k, v in med.items():
        if k == "fp16": continue
        f = F.get(f"{m}/{k}")
        if f is None: print("nofeat", m, k); continue
        rows.append((m, k, 100 * (v - b) / b, f["mac"], f["T"], f["Bytes"]))
print("n =", len(rows), " noise (max-min)/med %: median", round(np.median(noise), 2), "p90", round(np.percentile(noise, 90), 2))
y = np.array([r[2] for r in rows]); det = [r[0] for r in rows]
Xs = {"M_mac": np.array([[1, r[3]] for r in rows]), "M_bnd": np.array([[1, r[3], r[4], r[5]] for r in rows])}
from scipy.stats import spearmanr
res = {}
for name, X in Xs.items():
    e = []
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        beta = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
        e.append(X[i] @ beta - y[i])
    e = np.array(e)
    ed = []
    for dd in sorted(set(det)):
        mask = np.array([x != dd for x in det])
        beta = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
        ed += list(X[~mask] @ beta - y[~mask])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res[name] = np.mean(np.abs(e))
    print(name, "LOO MAE", round(np.mean(np.abs(e)), 2), " LODO MAE", round(np.mean(np.abs(ed)), 2),
          " Spearman(pred_LOO,y)", round(spearmanr(y + e, y)[0], 3), " beta", np.round(beta, 2))
r = res["M_bnd"] / res["M_mac"]
print("ratio", round(r, 3), "PASS" if (r <= 0.6 and res["M_bnd"] <= 4) else "FAIL")
for row, eb in zip(rows, y):
    pass
for rr in sorted(rows, key=lambda r: -r[2])[:8]:
    print([rr[0], rr[1], round(rr[2], 1), round(rr[3], 3), round(rr[4], 3), round(rr[5], 3)])
print("--- EXPLORATORY (post hoc, not the pre-registered test) ---")
for name, cols in [("mac", [3]), ("mac+T+Bytes", [3, 4, 5]), ("mac+Bytes", [3, 5])]:
    errs = {}
    for dd in sorted(set(det)):
        R = [r for r in rows if r[0] == dd]
        X = np.array([[1] + [r[c] for c in cols] for r in R]); yy = np.array([r[2] for r in R])
        e = []
        for i in range(len(yy)):
            mk = np.arange(len(yy)) != i
            b = np.linalg.lstsq(X[mk], yy[mk], rcond=None)[0]; e.append(X[i] @ b - yy[i])
        errs[dd] = round(float(np.mean(np.abs(e))), 2)
    print("per-detector LOO MAE", name, errs)
X = Xs["M_bnd"]
e = []
for i in range(len(y)):
    mk = np.arange(len(y)) != i
    b = np.linalg.lstsq(X[mk], y[mk], rcond=None)[0]; e.append(X[i] @ b - y[i])
for i in np.argsort(-np.abs(e))[:6]:
    print("worst pooled residual", rows[i][0], rows[i][1], "y", round(y[i], 1), "err", round(e[i], 1))

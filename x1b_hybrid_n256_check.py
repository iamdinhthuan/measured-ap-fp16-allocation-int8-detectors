"""Post-hoc check (exploratory, after the head-first hybrid had been evaluated): would the hybrid of fx_hybrid.py
select the same blocks if it used the 256-image output-SQNR scores of the main-text knapsack instead of the
2000-image scores? Solver-only, no engine. Block costs are the published block MAC shares of
metrics/alloc_<model>_int8ent_ho.json (non-head blocks do not overlap the forced set); c(F u head) is taken from
the hybrid configuration. Output: metrics/x1b_hybrid_n256_check.json (never overwritten).
Usage: python x1b_hybrid_n256_check.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from recon_select import knapsack  # noqa: E402


def main():
    os.chdir(ROOT)
    out = os.path.join("metrics", "x1b_hybrid_n256_check.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    res = {}
    for m in ["yolo11s", "yolo26s", "yolo11l", "yolo26l"]:
        hyb = json.load(open(os.path.join("configs", f"fx_{m}_int8ent_ho_hyb.json")))
        alloc = json.load(open(os.path.join("metrics", f"alloc_{m}_int8ent_ho.json")))
        cost = dict(zip(alloc["blocks"], alloc["cost"]))
        rest = [b for b in alloc["blocks"] if not b.startswith("head.")]
        c = np.array([cost[b] for b in rest])
        res[m] = {}
        for n in ["n2000", "n256"]:
            src = json.load(open(os.path.join("configs", f"fx_{m}_int8ent_ho_noise_{n}.json")))
            sc = json.load(open(src["scores"]))["scores"]
            s = np.array([sc[b]["sqnrq"] for b in rest])
            s = s / max(s.max(), 1e-30)
            for key, a in hyb["allocations"].items():
                B = float(key.split("@")[1])
                x = knapsack(s, c, B - hyb["cost_F_u_head"])
                sel = sorted(hyb["head_blocks"] + [rest[i] for i in np.flatnonzero(x)])
                res[m].setdefault(key, {})[n] = sel
                res[m][key][f"{n}_equals_published"] = sel == sorted(a["blocks"])
        for key, v in res[m].items():
            print(m, key, "n2000==published:", v["n2000_equals_published"], "n256==published:", v["n256_equals_published"])
    json.dump(res, open(out, "w"), indent=1)
    print("X1B_DONE", out)


if __name__ == "__main__":
    main()

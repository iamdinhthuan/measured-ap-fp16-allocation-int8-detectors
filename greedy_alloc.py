"""Greedy sensitivity fallback at matched MAC cost (post-freeze baseline, outside the confirmatory rule).

The partial-quantization recipe of Wu et al. (2020) and YOLOv6: rank layers by the accuracy recovered
when each one alone is kept in higher precision, then keep them in that order. Here the ranking uses the
same engine-measured single-block recoveries r[b, AP] as allocate.py, the same blocks, MAC accounting and
forced set; blocks are added in decreasing r while they fit in the budget (blocks that do not fit are
skipped). It differs from the 'mean' program only in replacing the exact 0/1 solution by the greedy order.

Usage: python greedy_alloc.py --alloc metrics/alloc_yolo26l_int8ent_ho.json --out configs/greedy_yolo26l_int8ent_ho.json
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True, help="plan written by allocate.py (blocks, cost, r, forced set)")
    ap.add_argument("--budgets", default="0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; choose a new name instead of overwriting")
    p = json.load(open(a.alloc))
    names, cost, r = p["blocks"], np.array(p["cost"]), np.array(p["r"])[:, 0]
    order = [i for i in np.argsort(-r, kind="stable") if r[i] > 0]
    plan = {k: p[k] for k in ("model", "base", "split", "force_regex", "force_cost")}
    plan.update({"criterion": "greedy (engine single-block AP recovery)", "source_alloc": a.alloc, "allocations": {}})
    for B in map(float, a.budgets.split(",")):
        sel, used = [], p["force_cost"]
        for i in order:
            if used + cost[i] <= B + 1e-12:
                sel.append(names[i])
                used += cost[i]
        mean = p["allocations"].get(f"mean@{B}", {}).get("blocks")
        same = mean is not None and sorted(mean) == sorted(sel)
        plan["allocations"][f"greedy@{B}"] = {"blocks": sel, "cost": float(used),
                                              "pred_recovery_AP": float(sum(r[names.index(b)] for b in sel)),
                                              "same_as_mean": same}
        print(f"B={B:.2f} cost={used:.3f} pred_rec={sum(r[names.index(b)] for b in sel):.2f} "
              f"mean_pred_rec={sum(r[names.index(b)] for b in (mean or [])):.2f} same_as_mean={same} {sel}")
    json.dump(plan, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

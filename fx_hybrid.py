"""Y1 (README 11.10, rules_11_10.md): head-first output-SQNR hybrid (exploratory, post-freeze).

Forced set F and every head block (convolutions under ^/model\\.23/) are restored; the published output-SQNR scores
(metrics/fx_scores_<model>_<base>_noise_n2000.json, criterion sqnrq) fill the remaining budget over the non-head blocks
with the knapsack of recon_select.py. Budgets where c(F u head) > B are skipped (recorded as infeasible).
Output (never overwritten): configs/fx_<model>_<base>_ho_hyb.json, allocation keys hyb@<B>
Usage: python fx_hybrid.py --model yolo26l
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from allocate import block_macs, conv_macs  # noqa: E402
from recon_select import knapsack  # noqa: E402

HEAD_RX = r"^/model\.23/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--mac-budgets", default="0.03,0.05,0.1,0.15,0.2")
    a = ap.parse_args()
    os.chdir(ROOT)
    out_cfg = os.path.join("configs", f"fx_{a.model}_{a.base}_ho_hyb.json")
    if os.path.exists(out_cfg):
        raise SystemExit(f"{out_cfg} exists")
    blocks = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))["blocks"]
    src = json.load(open(os.path.join("configs", f"fx_{a.model}_{a.base}_ho_noise_n2000.json")))
    force_rx = src["force_regex"]
    scores = json.load(open(src["scores"]))["scores"]
    head = [b for b in blocks if b.startswith("head.")]
    rest = [b for b in blocks if not b.startswith("head.")]
    assert all(re.match(r"\^/model\\\.23/", blocks[b]) for b in head), head
    assert not any(blocks[b].startswith("^/model\\.23/") for b in rest), rest
    macs = conv_macs(os.path.join("exports", f"{a.model}.onnx"))
    total = sum(macs.values())
    fh_cost = sum(v for k, v in macs.items() if re.search(force_rx, k) or re.search(HEAD_RX, k)) / total
    head_union = sum(v for k, v in macs.items() if any(re.search(blocks[b], k) for b in head)) / total
    assert abs(head_union - sum(v for k, v in macs.items() if re.search(HEAD_RX, k)) / total) < 1e-12
    bm = block_macs({b: blocks[b] for b in rest}, macs)
    cost = np.array([bm[b] / total for b in rest])
    sc = np.array([scores[b]["sqnrq"] for b in rest])
    sc = sc / max(sc.max(), 1e-30)
    plan = {"model": a.model, "base": a.base, "split": "tune", "criterion": "fx_hybrid head-first sqnrq (rules_11_10.md Y1)",
            "scores": src["scores"], "force_regex": force_rx, "force_cost": src["force_cost"],
            "head_blocks": head, "cost_F_u_head": fh_cost, "infeasible": [], "allocations": {}}
    print(f"{a.model}: c(F u head) = {fh_cost:.4f}", flush=True)
    for B in map(float, a.mac_budgets.split(",")):
        if fh_cost > B:
            plan["infeasible"].append(B)
            print(f"hyb B={B:.2f} infeasible", flush=True)
            continue
        x = knapsack(sc, cost, B - fh_cost)
        sel = [rest[i] for i in np.flatnonzero(x)]
        c = float(cost[x == 1].sum() + fh_cost)
        plan["allocations"][f"hyb@{B}"] = {"blocks": head + sel, "cost": c}
        print(f"hyb B={B:.2f} cost={c:.3f} extra={sel}", flush=True)
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print("HYB_DONE", out_cfg, flush=True)


if __name__ == "__main__":
    main()

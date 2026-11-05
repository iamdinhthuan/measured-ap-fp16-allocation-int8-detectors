"""X2 (README 11.9, rules_11_9.md): block-aggregated Fisher score (exploratory, post-freeze).

Block score = sum of the published per-convolution auto_quantize Fisher scores over the convolutions
matched by the block regex. Allocation identical to fx_factorial.py (score / max, recon_select.knapsack,
forced set and block MAC costs). Score file = the one behind the published Fisher arm:
autoquant_<model>_<base>.json (512 images) for the small detectors, ..._n2000.json for the large ones.
Output (never overwritten): configs/fx_<model>_<base>_ho_fisherblk.json
Usage: python fisher_block.py --model yolo26l
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import onnx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from allocate import block_macs, conv_macs  # noqa: E402
from autoquant_select import onnx_to_torch_name  # noqa: E402
from recon_select import knapsack  # noqa: E402

SCORE_FILE = {"yolo11s": "autoquant_yolo11s_int8ent.json", "yolo26s": "autoquant_yolo26s_int8ent.json",
              "yolo11l": "autoquant_yolo11l_int8ent_n2000.json", "yolo26l": "autoquant_yolo26l_int8ent_n2000.json"}
FISHER_PLAN = {"yolo11s": "fisher_yolo11s_int8ent_ho.json", "yolo26s": "fisher_yolo26s_int8ent_ho.json",
               "yolo11l": "fisher_yolo11l_int8ent_ho_n2000.json", "yolo26l": "fisher_yolo26l_int8ent_ho_n2000.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--mac-budgets", default="0.03,0.05,0.1,0.15,0.2")
    a = ap.parse_args()
    os.chdir(ROOT)
    out_cfg = os.path.join("configs", f"fx_{a.model}_{a.base}_ho_fisherblk.json")
    if os.path.exists(out_cfg):
        raise SystemExit(f"{out_cfg} exists")
    blocks = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))["blocks"]
    fs = json.load(open(os.path.join("configs", SCORE_FILE[a.model])))["fisher_scores"]
    force_rx = json.load(open(os.path.join("configs", FISHER_PLAN[a.model])))["force_regex"]
    convs = [n.name for n in onnx.load(os.path.join("exports", f"{a.model}.onnx")).graph.node if n.op_type == "Conv"]
    tn = {c: onnx_to_torch_name(c) for c in convs}
    assert set(tn.values()) == set(fs), (len(set(tn.values()) ^ set(fs)))
    members = {b: [c for c in convs if re.search(blocks[b], c)] for b in blocks}
    names = [b for b in blocks if members[b]]  # blocks without convolutions carry no Fisher score: never selected
    print("blocks without convolutions (excluded):", [b for b in blocks if not members[b]], flush=True)
    score = [sum(fs[tn[c]] for c in members[b]) for b in names]
    score = np.array(score)
    macs = conv_macs(os.path.join("exports", f"{a.model}.onnx"))
    total = sum(macs.values())
    force_cost = sum(v for k, v in macs.items() if re.search(force_rx, k)) / total
    bm = block_macs({b: blocks[b] for b in names}, macs)
    cost = np.array([bm[b] / total for b in names])
    sc = score / max(score.max(), 1e-30)
    plan = {"model": a.model, "base": a.base, "split": "tune", "criterion": "fisher_block (rules_11_9.md X2)",
            "score_file": SCORE_FILE[a.model], "block_scores": dict(zip(names, score.tolist())),
            "n_convs": {b: len(members[b]) for b in blocks}, "force_regex": force_rx, "force_cost": force_cost,
            "allocations": {}}
    for B in map(float, a.mac_budgets.split(",")):
        x = knapsack(sc, cost, B - force_cost)
        sel = [names[i] for i in np.flatnonzero(x)]
        plan["allocations"][f"fisherblk@{B}"] = {"blocks": sel, "cost": float(cost[x == 1].sum() + force_cost)}
        print(f"fisherblk B={B:.2f} cost={cost[x == 1].sum() + force_cost:.3f} {sel}", flush=True)
    json.dump(plan, open(out_cfg, "w"), indent=1)
    print("FISHERBLK_DONE", out_cfg, flush=True)


if __name__ == "__main__":
    main()

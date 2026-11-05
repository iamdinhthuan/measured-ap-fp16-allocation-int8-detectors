"""Random-selection baseline at matched MAC cost.

Same blocks, same MAC accounting and the same forced set as allocate.py, but
blocks are taken in a random order (seeded) and added while they fit in the
budget. Writes a plan readable by eval_alloc.py (keys random@B_sSEED).

Usage: python random_alloc.py --sens metrics/sens_yolo11s_int8ent.json --onnx exports/yolo11s.onnx \
           --budgets 0.05,0.1,0.2 --seeds 1,2,3,4,5 --force-regex REGEX --out metrics/random_yolo11s_int8ent_ho.json
"""
import argparse
import json
import re

import numpy as np

from allocate import block_macs, conv_macs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--budgets", default="0.05,0.1,0.2")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--force-regex", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = json.load(open(a.sens))
    names = [b for b in s["blocks"] if b in s["rows"]]
    macs = conv_macs(a.onnx)
    total = sum(macs.values())
    bm = block_macs({b: s["blocks"][b] for b in names}, macs)
    cost = {b: bm[b] / total for b in names}
    force_cost = sum(v for k, v in macs.items() if a.force_regex and re.search(a.force_regex, k)) / total
    plan = {"model": s["model"], "base": s["base"], "split": s.get("split", "tune"), "criterion": "random",
            "force_regex": a.force_regex, "force_cost": force_cost, "allocations": {}}
    for B in map(float, a.budgets.split(",")):
        for seed in map(int, a.seeds.split(",")):
            order = np.random.default_rng(20261025 + seed).permutation(names)
            sel, used = [], force_cost
            for b in order:
                if used + cost[b] <= B:
                    sel.append(str(b))
                    used += cost[b]
            plan["allocations"][f"random@{B}_s{seed}"] = {"blocks": sel, "cost": used}
            print(f"B={B} seed={seed} cost={used:.3f} {sel}")
    json.dump(plan, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

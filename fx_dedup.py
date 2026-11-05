"""For each allocation of a plan, find an already evaluated configuration with the identical restored regex
(same blocks and forced set, built exactly as eval_alloc.py builds it), so identical engines are not rebuilt.

Prints one line per key: '<key> eval' or '<key> same <existing tag>'.
Usage: python fx_dedup.py --alloc configs/fx_yolo26l_int8ent_ho_noise_n256.json --keys sqnrq@0.1,reconq@0.1
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--keys", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    plan = json.load(open(a.alloc))
    blocks = json.load(open(os.path.join("metrics", f"sens_{plan['model']}_{plan['base']}.json")))["blocks"]
    known = {}
    for f in sorted(glob.glob(os.path.join("metrics", "alloc_eval", f"{plan['model']}_{plan['base']}", "*.json"))):
        if f.endswith(("_val.json", "_tune.json")):
            continue
        r = json.load(open(f))
        if "val" in r.get("splits", {}) and os.path.exists(f[:-5] + "_val.pred.json.gz"):
            known.setdefault(r["regex"], os.path.basename(f)[:-5])
    for key in a.keys.split(","):
        sel = plan["allocations"][key]
        rx = "|".join(f"(?:{blocks[b]})" for b in sel["blocks"])
        if plan.get("force_regex"):
            rx = f"(?:{plan['force_regex']})" + (f"|{rx}" if rx else "")
        print(key, f"same {known[rx]}" if rx in known else "eval")


if __name__ == "__main__":
    main()

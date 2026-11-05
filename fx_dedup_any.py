"""Like fx_dedup.py but against evaluated configurations of any split (prefix filter), e.g. ExDark.
Usage: python fx_dedup_any.py --alloc CFG --keys k1,k2 --tag-prefix exdark_ --split exdark_test"""
import argparse, glob, json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--alloc", required=True); ap.add_argument("--keys", required=True)
ap.add_argument("--tag-prefix", default=""); ap.add_argument("--split", required=True)
a = ap.parse_args(); os.chdir(ROOT)
plan = json.load(open(a.alloc))
blocks = json.load(open(os.path.join("metrics", f"sens_{plan['model']}_{plan['base']}.json")))["blocks"]
known = {}
for f in sorted(glob.glob(os.path.join("metrics", "alloc_eval", f"{plan['model']}_{plan['base']}", a.tag_prefix + "*.json"))):
    if f.endswith((f"_{a.split}.json", "_val.json", "_tune.json")) or ".groups." in f:
        continue
    r = json.load(open(f))
    if "regex" in r and a.split in r.get("splits", {}) and os.path.exists(f[:-5] + f"_{a.split}.pred.json.gz"):
        known.setdefault(r["regex"], os.path.basename(f)[:-5])
for key in a.keys.split(","):
    sel = plan["allocations"][key]
    rx = "|".join(f"(?:{blocks[b]})" for b in sel["blocks"])
    if plan.get("force_regex"):
        rx = f"(?:{plan['force_regex']})" + (f"|{rx}" if rx else "")
    print(key, f"same {known[rx]}" if rx in known else "eval")

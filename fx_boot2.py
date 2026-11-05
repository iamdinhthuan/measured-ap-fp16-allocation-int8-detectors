"""Copy of fx_boot.py (README 11.10) with --n and fixed-file arms 'label=!path' (same file at every budget).

Paired bootstrap for the README 11.8 arms (exploratory, post-freeze).

Each arm is 'label=cfg:crit' (allocation keys '<crit>@<B>' of an allocation config) or 'label=@tag'
(an already evaluated tag '<tag>_B<B>'). For a config arm, the prediction file of each budget is the
evaluated configuration with the identical restored regex (built as eval_alloc.py builds it);
published tags are preferred over fx-prefixed ones, so identical allocations share one file.

Usage: python fx_boot.py --model yolo26l --out fx_yolo26l_f1 \
         --arm mean=@mean --arm sqnrfx=configs/fx_yolo26l_int8ent_ho_noise_n256.json:sqnrq \
         --pairs sqnrfx:mean,reconfx:mean
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
P = sys.executable
GT = "data/coco_src/annotations/instances_val2017.json"


def evaluated(model, base):
    d = os.path.join("metrics", "alloc_eval", f"{model}_{base}")
    known = {}
    files = sorted(glob.glob(os.path.join(d, "*.json")), key=lambda f: (os.path.basename(f).startswith("fx"), f))
    for f in files:
        if f.endswith(("_val.json", "_tune.json")):
            continue
        r = json.load(open(f))
        pred = f[:-5] + "_val.pred.json.gz"
        if "val" in r.get("splits", {}) and os.path.exists(pred):
            known.setdefault(r["regex"], pred)
    return d, known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--pairs", required=True, help="a:b,... over arm labels; each budget gives a_B:b_B (b - a)")
    ap.add_argument("--budgets", default="0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--n", default="1000")
    ap.add_argument("--prefer", action="append", default=[],
                    help="label=tag: if '<tag>_B<B>' was evaluated with the identical regex, use it (keeps an arm on its own published engine)")
    a = ap.parse_args()
    os.chdir(ROOT)
    out = os.path.join("metrics", "bootstrap", a.out + ".json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    blocks = json.load(open(os.path.join("metrics", f"sens_{a.model}_{a.base}.json")))["blocks"]
    d, known = evaluated(a.model, a.base)
    budgets = a.budgets.split(",")
    sets = {"fp32": os.path.join(d, "fp32_val.pred.json.gz")}
    for spec in a.arm:
        label, src = spec.split("=", 1)
        for B in budgets:
            if src.startswith("!"):
                path = src[1:]
            elif src.startswith("@"):
                path = os.path.join(d, f"{src[1:]}_B{B}_val.pred.json.gz")
            else:
                cfg, crit = src.rsplit(":", 1)
                plan = json.load(open(cfg))
                sel = plan["allocations"][f"{crit}@{B}"]
                rx = "|".join(f"(?:{blocks[b]})" for b in sel["blocks"])
                if plan.get("force_regex"):
                    rx = f"(?:{plan['force_regex']})" + (f"|{rx}" if rx else "")
                if rx not in known:
                    raise SystemExit(f"{label} {B}: allocation not evaluated yet")
                path = known[rx]
                for pl, pt in (x.split("=", 1) for x in a.prefer):
                    own = os.path.join(d, f"{pt}_B{B}")
                    if pl == label and os.path.exists(own + "_val.pred.json.gz") and json.load(open(own + ".json"))["regex"] == rx:
                        path = own + "_val.pred.json.gz"
            assert os.path.exists(path), path
            sets[f"{label}_B{B}"] = path
            print(f"{label}_B{B} -> {os.path.basename(path)}", flush=True)
    pairs = ",".join(f"{x}_B{B}:{y}_B{B}" for p in a.pairs.split(",") for x, y in [p.split(":")] for B in budgets)
    if a.dry:
        return
    cmd = [P, "bootstrap_fast.py", "--gt", GT, "--sets", ",".join(f"{k}={v}" for k, v in sets.items()),
           "--ref", "fp32", "--pairs", pairs, "--n", a.n, "--out", out]
    sys.exit(subprocess.call(["nice", "-n", "10"] + cmd))


if __name__ == "__main__":
    main()

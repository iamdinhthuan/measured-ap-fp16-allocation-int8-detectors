"""Y2 (rules_11_10.md): 10,000-resample paired bootstrap of the published SQNR pairs (sqnrfx:mean),
using exactly the prediction files recorded in the published F1/F1b outputs (no re-mapping).
Usage: python y2_driver.py <model> <published_tag>   -> metrics/bootstrap/fx_<model>_y2_10k.json"""
import json, subprocess, sys
m, tag = sys.argv[1], sys.argv[2]
src = json.load(open(f"metrics/bootstrap/fx_{m}_{tag}.json"))
sets = {k: v for k, v in src["sets"].items() if k == "fp32" or k.startswith(("mean_B", "sqnrfx_B"))}
pairs = [p for p in src["diff_b_minus_a"] if p.startswith("sqnrfx_B")]
assert len(pairs) == 5 and len(sets) == 11, (pairs, list(sets))
for k, v in sets.items():
    print(k, "->", v, flush=True)
out = f"metrics/bootstrap/fx_{m}_y2_10k.json"
cmd = ["nice", "-n", "10", sys.executable, "bootstrap_fast.py", "--gt", src["gt"],
       "--sets", ",".join(f"{k}={v}" for k, v in sets.items()), "--ref", src["ref"],
       "--pairs", ",".join(pairs), "--n", "10000", "--seed", str(src["seed"]), "--out", out]
sys.exit(subprocess.call(cmd))

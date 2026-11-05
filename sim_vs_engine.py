"""E2 descriptive statistics: single-block recoveries measured in simulation vs on the engine (README 11.6)."""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

KEYS = ("AP", "AP_small", "AP_medium", "AP_large")
out = {}
for m in sys.argv[1:]:
    d = json.load(open(f"metrics/sens_sim_{m}_int8ent.json"))
    rows, base = d["rows"], d["base"]
    blocks = [b for b in d["blocks"] if b in rows]
    res = {"n_blocks": len(blocks)}
    for k in KEYS:
        s = np.array([100 * (rows[b]["stats"][k] - rows[base]["stats"][k]) for b in blocks])
        e = np.array([100 * (rows[b]["engine_stats"][k] - rows[base]["engine_stats"][k]) for b in blocks])
        top = lambda v: set(np.argsort(-v)[:5])
        res[k] = {"spearman": float(spearmanr(s, e)[0]), "pearson": float(pearsonr(s, e)[0]),
                  "mean_abs_diff": float(np.abs(s - e).mean()), "max_abs_diff": float(np.abs(s - e).max()),
                  "top5_overlap": len(top(s) & top(e))}
    res["gap"] = {"sim": 100 * (rows["fp16_all"]["stats"]["AP"] - rows[base]["stats"]["AP"]),
                  "engine": 100 * (rows["fp16_all"]["engine_stats"]["AP"] - rows[base]["engine_stats"]["AP"])}
    out[m] = res
    print(m, json.dumps(res["AP"]), "gap", res["gap"])
fn = "metrics/sim_vs_engine_" + "_".join(sys.argv[1:]) + ".json"
if not os.path.exists(fn):
    json.dump(out, open(fn, "w"), indent=1)

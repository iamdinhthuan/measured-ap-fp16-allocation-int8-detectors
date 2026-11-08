"""X2: exact-cost plans from metrics/x1_exactcost.json. Writes configs/x2_<m>_exact.json (published arm) and
configs/x2_<m>_fexact.json (F-restored arm) and prints 'model arm keys' lines for the queue."""
import copy
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
X = json.load(open("metrics/x1_exactcost.json"))["models"]
SRC = {"published": "metrics/alloc_{m}_int8ent_ho.json", "fsweep": "configs/fx_{m}_int8ent_ho_fsweep.json"}
OUTN = {"published": "configs/x2_{m}_exact.json", "fsweep": "configs/x2_{m}_fexact.json"}
for m, d in X.items():
    for arm, A in d["arms"].items():
        keys = [k for k, v in A.items() if k.startswith("mean@") and v["changed"]]
        if not keys:
            continue
        plan = json.load(open(SRC[arm].format(m=m)))
        new = copy.deepcopy(plan)
        new["x2_source"] = {"plan": SRC[arm].format(m=m), "x1": "metrics/x1_exactcost.json", "rules": "rules_20261108_x2.md"}
        for k in keys:
            v = A[k]
            assert sorted(plan["allocations"][k]["blocks"]) == sorted(v["published"]), (m, arm, k)
            new["allocations"][k] = {"blocks": v["exact"], "cost": v["exact_mac_x1"], "pred_gap_AP": None,
                                     "exact_mac_published": v["exact_mac_published"], "pred_dAP_tune": v["pred_dAP_tune"]}
        out = OUTN[arm].format(m=m)
        if os.path.exists(out):
            assert json.load(open(out))["allocations"] == new["allocations"], f"{out} differs"
        else:
            json.dump(new, open(out, "w"), indent=1)
        print(m, arm, ",".join(keys))

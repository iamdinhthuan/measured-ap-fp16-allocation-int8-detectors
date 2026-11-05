"""Val AP and restored MAC share per budget for allocation arms (resolved as fx_boot.py), for Table 3 rows.
Usage: python fx_table.py --model yolo26l --arm sqnrfx=configs/...:sqnrq [--arm ...]"""
import argparse, json, os
from fx_boot import evaluated, ROOT
from collect_results import mac_share
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--arm", action="append")
ap.add_argument("--budgets", default="0.03,0.05,0.1,0.15,0.2"); ap.add_argument("--prefer", action="append", default=[])
a = ap.parse_args(); os.chdir(ROOT)
blocks = json.load(open(os.path.join("metrics", f"sens_{a.model}_int8ent.json")))["blocks"]
d, known = evaluated(a.model, "int8ent")
for spec in a.arm:
    label, src = spec.split("=", 1); cfg, crit = src.rsplit(":", 1); plan = json.load(open(cfg)); cells = []
    for B in a.budgets.split(","):
        sel = plan["allocations"][f"{crit}@{B}"]
        rx = "|".join(f"(?:{blocks[b]})" for b in sel["blocks"])
        if plan.get("force_regex"):
            rx = f"(?:{plan['force_regex']})" + (f"|{rx}" if rx else "")
        pred = known[rx]
        for pl, pt in (x.split("=", 1) for x in a.prefer):
            own = os.path.join(d, f"{pt}_B{B}")
            if pl == label and os.path.exists(own + "_val.pred.json.gz") and json.load(open(own + ".json"))["regex"] == rx:
                pred = own + "_val.pred.json.gz"
        r = json.load(open(pred.replace("_val.pred.json.gz", ".json")))
        cells.append(f"{100 * r['splits']['val']['stats']['AP']:.2f} ({100 * mac_share(a.model, rx):.1f})[{os.path.basename(pred)[:-17]}]")
    print(a.model, label, " & ".join(cells))

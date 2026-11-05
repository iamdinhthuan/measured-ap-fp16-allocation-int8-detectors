"""Audit for README 11.8: (1) twin builds (same regex, several evaluated tags) with their val AP;
(2) whether SQNRxq selections changed from 256 to 2000 images; (3) FX_DONE times are read from logs separately."""
import glob, json, os
from collections import defaultdict
from fx_boot import ROOT
os.chdir(ROOT)
for M in ("yolo11s", "yolo26s", "yolo11l", "yolo26l"):
    by = defaultdict(list)
    for f in sorted(glob.glob(f"metrics/alloc_eval/{M}_int8ent/*.json")):
        if f.endswith(("_val.json", "_tune.json")) or ".groups." in f or "exdark" in f:
            continue
        r = json.load(open(f))
        if "val" in r.get("splits", {}):
            by[r["regex"]].append((os.path.basename(f)[:-5], round(100 * r["splits"]["val"]["stats"]["AP"], 2)))
    tw = [v for v in by.values() if len(v) > 1 and any(t.startswith("fx") for t, _ in v)]
    for v in tw:
        print(M, "TWIN", v)
    a = json.load(open(f"configs/fx_{M}_int8ent_ho_noise_n256.json"))["allocations"]
    b = json.load(open(f"configs/fx_{M}_int8ent_ho_noise_n2000.json"))["allocations"]
    for B in ("0.03", "0.05", "0.1", "0.15", "0.2"):
        k = f"sqnrq@{B}"
        print(M, k, "n256==n2000" if sorted(a[k]["blocks"]) == sorted(b[k]["blocks"]) else f"DIFF {sorted(set(a[k]['blocks']) ^ set(b[k]['blocks']))}")

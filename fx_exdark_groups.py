"""ExDark lighting-group AP rows (as tables/exdark.tex) for evaluated tags; caches via collect_results.light_groups.
Usage: python fx_exdark_groups.py exdark_sqnr_B0.03 fxex_sqnrq_B0.03 ..."""
import json, os, sys
from collect_results import LIGHT, light_groups, mac_share, ROOT
D = os.path.join(ROOT, "metrics", "alloc_eval", "yolo11s_int8ent")
GT = os.path.join(ROOT, "data", "exdark", "exdark_test.json")
ref = json.load(open(os.path.join(ROOT, "metrics", "exdark", "yolo11s_fp32_test.groups.json")))
ref = {n: 100 * ref[f"light={n}"] for n in LIGHT} if max(ref[f"light={n}"] for n in LIGHT) < 1 else {n: ref[f"light={n}"] for n in LIGHT}
for tag in sys.argv[1:]:
    r = json.load(open(os.path.join(D, tag + ".json")))
    g = light_groups(GT, os.path.join(D, tag + "_exdark_test.pred.json.gz"))
    g = {n: (100 * v if v < 1 else v) for n, v in g.items()}
    ap = 100 * r["splits"]["exdark_test"]["stats"]["AP"]
    loss = {n: 100 * (ref[n] - g[n]) / ref[n] for n in LIGHT}
    w = max(loss, key=loss.get)
    print(f"{tag} & {100 * mac_share('yolo11s', r['regex']):.1f} & {ap:.2f} & " + " & ".join(f"{g[n]:.1f}" for n in LIGHT)
          + f" & {loss[w]:.1f} ({w})")

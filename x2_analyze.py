"""Apply the X2 decision rule (rules_20261108_x2.md section 2) to the fetched X2 outputs.
Usage: python3 x2_analyze.py <root with metrics/> <x1_exactcost.json>
Writes metrics/x2/x2_summary.json under <root> and prints a table."""
import json, sys, os

root, x1p = sys.argv[1], sys.argv[2]
x1 = json.load(open(x1p))["models"]
ALPHA = 0.05


def p2(ple0):
    return max(2 * min(ple0, 1 - ple0), 2 / 1001)


def holm(rows):
    rows = sorted(rows, key=lambda r: r["p"])
    m, stop = len(rows), False
    for k, r in enumerate(rows):
        thr = ALPHA / (m - k)
        r["holm_thr"] = thr
        if not stop and r["p"] <= thr + 1e-12:
            r["holm_reject"] = True
        else:
            stop = True
            r["holm_reject"] = False


g3 = {}
g3p = os.path.join(root, "metrics/x2/g3.txt")
if os.path.exists(g3p):
    for line in open(g3p):
        t = line.split()
        if t and t[0] == "G3":
            g3[(t[1], t[2])] = line.strip()

fam = {"published": [], "fsweep": []}
for m in ["yolo26l", "yolo26s", "yolo11s", "yolo11l"]:
    bp = os.path.join(root, f"metrics/bootstrap/x2_{m}.json")
    if not os.path.exists(bp):
        print("MISSING", bp)
        continue
    bj = json.load(open(bp)); bs, pt = bj["diff_b_minus_a"], bj["point"]
    for pair, v in bs.items():
        a, b = pair.split(":")
        arm = "published" if a.startswith("r_") else "fsweep"
        B = a.split("_", 1)[1]
        key = f"mean@{float(B):g}"
        x = x1[m]["arms"][arm][key]
        assert x["changed"], (m, arm, key)
        ap = v["AP"]
        pfx_r, pfx_e = ("x2r_", "x2e_") if arm == "published" else ("x2fr_", "x2fe_")
        d = os.path.join(root, f"metrics/alloc_eval/{m}_int8ent/")
        tag = key.replace("@", "_B")
        apr = 100 * json.load(open(d + f"{pfx_r}{tag}_val.json"))["stats"]["AP"]
        ape = 100 * json.load(open(d + f"{pfx_e}{tag}_val.json"))["stats"]["AP"]
        fam[arm].append(dict(model=m, arm=arm, B=key, AP_rebuilt=apr, AP_exact=ape,
                             diff=pt[b]["AP"] - pt[a]["AP"], boot_mean=ap["mean"], ci95=ap["ci95"], p_le0=ap["p_le0"], p=p2(ap["p_le0"]),
                             mac_published=100 * x["exact_mac_published"], mac_exact=100 * x["exact_mac_x1"],
                             g3=g3.get((m, f"{pfx_r}{tag}"), "")))
for arm, rows in fam.items():
    holm(rows)
    print(f"== {arm} family (m={len(rows)})")
    for r in sorted(rows, key=lambda r: (r["model"], r["B"])):
        print(f"{r['model']:8s} {r['B']:10s} rebuilt={r['AP_rebuilt']:.2f} exact={r['AP_exact']:.2f} "
              f"diff={r['diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p={r['p']:.4f} "
              f"holm={'REJ' if r['holm_reject'] else '-'} MAC {r['mac_published']:.1f}->{r['mac_exact']:.1f}"
              f"{'  ' + r['g3'].split('diff=')[1] if r['g3'] else ''}")
pub = fam["published"]
better = [r for r in pub if r["holm_reject"] and r["diff"] > 0]
worse = [r for r in pub if r["holm_reject"] and r["diff"] < 0]
outcome = "b" if better else ("c" if worse else "a")
rng = (min(r["diff"] for r in pub), max(r["diff"] for r in pub)) if pub else None
print("OUTCOME", outcome, "range", rng)
json.dump(dict(outcome=outcome, range=rng, families=fam), open(os.path.join(root, "metrics/x2/x2_summary.json"), "w"), indent=1)

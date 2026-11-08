"""Supplementary table of X2 (rules_20261108_x2.md) from metrics/x2/x2_summary.json.
Usage: python3 x2_table.py metrics/x2/x2_summary.json > tables/x2.tex"""
import json, sys

s = json.load(open(sys.argv[1]))
NAME = {"yolo11s": "YOLO11-s", "yolo26s": "YOLO26-s", "yolo11l": "YOLO11-l", "yolo26l": "YOLO26-l"}
ORDER = ["yolo11s", "yolo26s", "yolo11l", "yolo26l"]


def f2(x):
    return f"{x:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")


out = [r"\begin{tabular}{@{}lllllll@{}}", r"\toprule",
       r"Arm & Model & $B$ & Published (MAC, \%) & Exact cost (MAC, \%) & Exact$-$published & G3\\", r"\midrule"]
for arm, label in (("published", "Main"), ("fsweep", r"$\Fset$ restored")):
    rows = sorted(s["families"][arm], key=lambda r: (ORDER.index(r["model"]), float(r["B"].split("@")[1])))
    first = True
    for r in rows:
        star = ("^{**}" if r["holm_reject"] else "^{*}") if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else ""
        g3 = r["g3"].split("diff=")[1].split()[0] if r["g3"] else "--"
        B = round(100 * float(r["B"].split("@")[1]))
        out.append(f"{label if first else ''} & {NAME[r['model']]} & {B} & "
                   f"{r['AP_rebuilt']:.2f} ({r['mac_published']:.1f}) & {r['AP_exact']:.2f} ({r['mac_exact']:.1f}) & "
                   f"${f2(r['diff'])}{star}$ {{\\tiny$[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]$}} & ${f2(float(g3))}$\\\\")
        first = False
    if arm == "published":
        out.append(r"\midrule")
out += [r"\bottomrule", r"\end{tabular}"]
print("\n".join(out))

"""Supplementary table of JHF2 (rules_20261109_jhf2.md) from metrics/orin_latency/yolo26l_jhf2_summary.json.
Usage: python3 jhf2_table.py metrics/orin_latency/yolo26l_jhf2_summary.json > tables/jhf2.tex
AP of INT8, INT8 + F and whole head FP16 is the desktop AP of Table 5 of the article (no *_val.json for these graphs)."""
import json, sys
s = json.load(open(sys.argv[1])); E, C = s["engines"], s["consistency"]
AP_T5 = {"base": 48.48, "headout": 50.05, "head": 53.08}
ROWS = [("base", "INT8 (entropy)"), ("headout", r"INT8 + $\mathcal{F}$"), ("head", "whole head FP16"),
        ("mean_B0.05", r"Measured, mean $B=5\%$"), ("mean_B0.1", r"Measured, mean $B=10\%$"), ("rb_mean_B0.1", r"\quad rebuild"),
        ("mean_B0.2", r"Measured, mean $B=20\%$"), ("rb_mean_B0.2", r"\quad rebuild"),
        ("hf_B0.1", r"Head-first SQNR $B=10\%$"), ("hf_B0.15", r"Head-first SQNR $B=15\%$"), ("hf_B0.2", r"Head-first SQNR $B=20\%$")]
def pct(x): return f"{x:+.1f}".replace("-", "$-$")
out = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
       r"Configuration & AP & latency (ms) & vs.\ INT8 (\%) & TRT~10.3 (\%) & power (W) & energy (mJ) & engine (MB)\\", r"\midrule"]
for k, lab in ROWS:
    e = E[k]; ap = e.get("AP_val") or AP_T5[k]
    t5 = pct(C[k]["table5_vs_int8_pct"]) if k in C else "--"
    out.append(f"{lab} & {ap:.2f} & {e['ms']:.2f} & {pct(e['vs_int8_pct'])} & {t5} & {e['W']:.1f} & {e['mJ']:.0f} & {e['MiB']:.1f}\\\\")
out += [r"\bottomrule", r"\end{tabular}"]
print("\n".join(out))

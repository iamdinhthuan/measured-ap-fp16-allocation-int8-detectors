"""Tables and figures for the Information Sciences manuscript, from the server summary.

Inputs : data/summary_*.json (collect_results.py), data/bootstrap/*.json, data/orin/*.jsonl
Outputs: tables/*.tex, figs/*.pdf. Every number in the results section comes from here.

Usage: python make_tables.py data/summary_YYYYMMDD_HHMMSS.json
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "tables")
F = os.path.join(HERE, "figs")
MODELS = ["yolo11s", "yolo26s", "yolo11l", "yolo26l"]
NAME = {"yolo11s": "YOLO11-s", "yolo26s": "YOLO26-s", "yolo11l": "YOLO11-l", "yolo26l": "YOLO26-l"}
BUDGETS = ["0.03", "0.05", "0.1", "0.15", "0.2"]
BITS = ["8.02", "8.04", "8.08", "8.15", "8.3"]
LIGHT = ["Ambient", "Strong", "Single", "Weak", "Twilight", "Object", "Low", "Window"]
SIZE = ["AP_small", "AP_medium", "AP_large"]


def load(path):
    d = json.load(open(path))
    rows = defaultdict(dict)
    for c in d["configs"]:
        if c["base"] != "int8ent":
            continue
        rows[c["model"]][c["tag"]] = c
    return d, rows


def worst_rel(s, ref, groups):
    return max(100 * (ref[g] - s[g]) / ref[g] for g in groups)


def method_points(rows, split="val", prefix=""):
    """{method: [(mac%, AP, cfg)]} for one model and split."""
    out = defaultdict(list)
    for tag, c in rows.items():
        if split not in c["splits"]:
            continue
        t = tag[len(prefix):] if tag.startswith(prefix) else None
        if t is None:
            continue
        m = re.match(r"(mean_matched|mean|minimax|fisher|autoquant|recon|sqnr)_B([\d.]+)$", t)
        r = re.match(r"random_B([\d.]+)_s(\d+)$", t)
        if m:
            out[m.group(1)].append((100 * c["mac_actual"], c["splits"][split]["AP"], c, m.group(2)))
        elif r:
            out["random"].append((100 * c["mac_actual"], c["splits"][split]["AP"], c, r.group(1)))
        elif t in ("headout", "stem_headout", "head", "base", "fp16_all"):
            out[t].append((100 * c["mac_actual"], c["splits"][split]["AP"], c, None))
    return out


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def plan_blocks(d, m, obj, B, kind="size"):
    """Blocks chosen by objective obj at budget B in the latest COCO (size) or ExDark plan of model m."""
    hs = [h for h in d["heterogeneity"] if h["model"] == m and ("exdark" in h["plan"]) == (kind == "light")]
    hs.sort(key=lambda h: "_v2" in h["plan"], reverse=True)
    for h in hs:
        pb = h["per_budget"].get(B, {})
        if f"{obj}_blocks" in pb:
            return pb[f"{obj}_blocks"]
    return None


def table_coco(d, rows):
    """AP (MAC%) per method and budget on COCO val, one block per model."""
    lines = [r"\begin{tabular}{@{}ll" + "c" * 5 + "@{}}", r"\toprule",
             r"Model & Method & \multicolumn{5}{c}{AP on val2017 (restored MAC share, \%)}\\",
             r"\cmidrule(l){3-7}", r" & & $B=3\%$ & $5\%$ & $10\%$ & $15\%$ & $20\%$\\", r"\midrule"]
    for m in MODELS:
        if m not in rows:
            continue
        pts = method_points(rows[m])
        ref = d["references"].get(m, {})
        fp32 = ref.get("fp32_val", ref.get("fp32", {})).get("AP")
        ho = pts.get("headout", [(None, None)])[0]
        n_rows = 5 + sum(1 for k in ("recon", "sqnr") if pts.get(k))
        lines.append(rf"\multirow{{{n_rows}}}{{*}}{{\shortstack[l]{{{NAME[m]}\\FP32 {fmt(fp32)}\\INT8+$\mathcal{{F}}$ {fmt(ho[1])}}}}}")
        for meth, label, keys in (("autoquant", r"\texttt{auto\_quantize}$^\dagger$", BITS),
                                  ("random", "Random (mean of 5)", BUDGETS),
                                  ("fisher", "Fisher knapsack", BUDGETS),
                                  ("recon", r"Recon.\ error knapsack$^\ddagger$", BUDGETS),
                                  ("sqnr", r"Output SQNR knapsack$^\ddagger$", BUDGETS),
                                  ("mean", r"\textbf{Engine, mean}", BUDGETS),
                                  ("minimax", r"\textbf{Engine, worst-group}", BUDGETS)):
            if meth in ("recon", "sqnr") and not pts.get(meth):
                continue
            cells = []
            for k in keys:
                p = [x for x in pts.get(meth, []) if x[3] == k]
                if not p and meth in ("mean", "minimax"):  # identical selection evaluated under another tag
                    blocks = plan_blocks(d, m, meth, k)
                    t = blocks and resolve(rows[m], "", blocks, "val")
                    if t:
                        c = rows[m][t]
                        cells.append(f"{c['splits']['val']['AP']:.2f} ({100 * c['mac_actual']:.1f})$^{{=}}$")
                        continue
                if not p:
                    cells.append("--")
                elif meth == "random":
                    ap = np.mean([x[1] for x in p]); mac = np.mean([x[0] for x in p])
                    cells.append(f"{ap:.2f}$\\pm${np.std([x[1] for x in p]):.2f} ({mac:.1f})")
                else:
                    cells.append(f"{p[0][1]:.2f} ({p[0][0]:.1f})")
            lines.append(f" & {label} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(T, "coco_methods.tex"), "w").write("\n".join(lines) + "\n")


def table_refs(d, rows):
    """Uniform and heuristic configurations: AP and size-group AP on COCO val."""
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Model & Configuration & MAC (\%) & AP & AP$_S$ & AP$_M$ & AP$_L$\\", r"\midrule"]
    for m in MODELS:
        if m not in rows:
            continue
        ref = d["references"].get(m, {})
        pts = method_points(rows[m])
        entries = [("FP32", None, ref.get("fp32_val") or ref.get("fp32"))]
        entries.append(("INT8 (entropy)", 0.0, ref.get("int8ent")))
        for t, lab in (("headout", r"INT8 + $\mathcal{F}$ (head-out)"), ("stem_headout", r"+ stem"),
                       ("head", "whole head FP16")):
            if t in pts:
                entries.append((lab, pts[t][0][0], pts[t][0][2]["splits"]["val"]))
            elif t == "head" and ref.get("int8ent_head"):
                mac = ref.get("head_mac")
                entries.append((lab, None if mac is None else 100 * mac, ref["int8ent_head"]))
        first = True
        for lab, mac, s in entries:
            if not s:
                continue
            lines.append(f"{NAME[m] if first else ''} & {lab} & {fmt(mac, 1)} & {s['AP']:.2f} & "
                         f"{s['AP_small']:.2f} & {s['AP_medium']:.2f} & {s['AP_large']:.2f}\\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(T, "coco_refs.tex"), "w").write("\n".join(lines) + "\n")


def table_exdark(d, rows, m="yolo11s"):
    if m not in rows:
        return
    ref = d["references"][m]["fp32_exdark_test"]
    lines = [r"\begin{tabular}{@{}lr r" + "r" * len(LIGHT) + r" r@{}}", r"\toprule",
             r"Configuration & MAC & AP & " + " & ".join(LIGHT) + r" & worst rel.\ loss\\",
             r" & (\%) & & " + " & ".join("" for _ in LIGHT) + r" & (\%, group)\\", r"\midrule",
             "FP32 & -- & " + f"{ref['AP']:.2f} & " + " & ".join(f"{ref['light'][g]:.1f}" for g in LIGHT) + r" & --\\"]
    order = ["exdark_fp16_all", "exdark_base", "exdark_headout", "exdark_stem_headout"]
    for B in BUDGETS:
        order += [f"exdark_mean_B{B}", f"exdark_minimax_B{B}", f"exdark_sqnr_B{B}"]
    labels = {"exdark_fp16_all": "FP16 (all restored)", "exdark_base": "INT8 (entropy)",
              "exdark_headout": r"INT8 + $\mathcal{F}$", "exdark_stem_headout": r"+ stem"}
    for t in order:
        c = rows[m].get(t)
        if not c or "exdark_test" not in c["splits"]:
            continue
        s = c["splits"]["exdark_test"]
        kind = "Engine, mean" if "_mean_" in t else "Engine, worst-group" if "_minimax_" in t else r"Output SQNR$^\ddagger$"
        lab = labels.get(t) or (kind + f" $B={float(t.split('_B')[1]) * 100:.0f}\\%$")
        wr = {g: 100 * (ref["light"][g] - s["light"][g]) / ref["light"][g] for g in LIGHT}
        gw = max(wr, key=wr.get)
        lines.append(f"{lab} & {100 * c['mac_actual']:.1f} & {s['AP']:.2f} & " +
                     " & ".join(f"{s['light'][g]:.1f}" for g in LIGHT) + f" & {wr[gw]:.1f} ({gw})\\\\")
        if t == "exdark_stem_headout":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "exdark.tex"), "w").write("\n".join(lines) + "\n")


def resolve(rows_m, prefix, blocks, split):
    """Tag of an evaluated config of this model with exactly this block set (any objective)."""
    want = sorted(blocks)
    for tag, c in sorted(rows_m.items()):
        if (tag.startswith(prefix) and re.match(r"(mean_matched|mean|minimax)_B", tag[len(prefix):])
                and sorted(c.get("blocks") or []) == want and split in c["splits"]):
            return tag
    return None


def table_objective(d, rows, boots):
    """Heterogeneity, predicted and realised effect of the worst-group objective at equal spend.

    The comparison is minimax vs mean_matched (README 10.8a); each selection is resolved to the
    evaluated configuration with the identical block set, whatever objective produced it.
    """
    lines = [r"\begin{tabular}{@{}llr rr rrr@{}}", r"\toprule",
             r"Model / groups & $B$ & $\delta$ & pred.\ benefit & bound $h$ & MAC (cm/mm) & $\Delta$AP & $\Delta$worst (full set) [95\% CI]\\",
             r"\midrule"]
    plans = {h["plan"]: h for h in d["heterogeneity"]}
    for plan, h in sorted(plans.items(), key=lambda x: ("exdark" in x[0], MODELS.index(x[1]["model"]), x[0])):
        if "_v2" not in plan and plan.replace(".json", "_v2.json") in plans:
            continue
        m = h["model"]
        exd = "exdark" in plan
        groups = LIGHT if exd else SIZE
        split = "exdark_test" if exd else "val"
        pre = "exdark_" if exd else ""
        ref = d["references"][m]["fp32_exdark_test"]["light"] if exd else (d["references"][m].get("fp32_val") or d["references"][m]["fp32"])
        for B, pb in sorted(h["per_budget"].items(), key=lambda x: float(x[0])):
            if "pred_benefit_matched" not in pb:
                continue
            head = (f"{NAME[m]} / {'light' if exd else 'size'} & {100 * float(B):.0f}\\% & {h['delta'] * 100:.2f} & "
                    f"{pb['pred_benefit_matched'] * 100:.2f} & {pb['bound_h_matched'] * 100:.2f} & ")
            if pb["same_selection_matched"]:
                lines.append(head + r"\multicolumn{3}{c}{identical selection}\\")
                continue
            ta = resolve(rows[m], pre, pb["mean_matched_blocks"], split)
            tb = resolve(rows[m], pre, pb["minimax_blocks"], split)
            mac = dap = dwr = "--"
            if ta and tb:
                a, b = rows[m][ta], rows[m][tb]
                sa, sb = a["splits"][split], b["splits"][split]
                ga = sa["light"] if exd else sa
                gb = sb["light"] if exd else sb
                mac = f"{100 * a['mac_actual']:.1f}/{100 * b['mac_actual']:.1f}"
                dap = f"{sb['AP'] - sa['AP']:+.2f}"
                dwr = f"{worst_rel(ga, ref, groups) - worst_rel(gb, ref, groups):+.2f}"
                kind = "exdark" if exd else "coco"
                bk = boots.get((m, kind, f"{ta[len(pre):]}:minimax_B{B}"))
                if bk:
                    w = bk["worst_rel"]
                    dwr += f" [{w['ci95'][0]:+.1f}, {w['ci95'][1]:+.1f}]"
            lines.append(head + f"{mac} & {dap} & {dwr}\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(T, "objective.tex"), "w").write("\n".join(lines) + "\n")


def table_surrogate(d, rows):
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Model & allocations & mean $|$error$|$ (AP) & max $|$error$|$ (AP) & corr.\ pred./meas.\\", r"\midrule"]
    for m in MODELS:
        if m not in rows:
            continue
        I = [(c["pred_gap_AP_tune"], c["meas_gap_AP_tune"]) for c in rows[m].values()
             if c.get("meas_gap_AP_tune") is not None]
        if not I:
            continue
        p, q = np.array(I).T
        lines.append(f"{NAME[m]} & {len(I)} & {np.mean(np.abs(p - q)):.2f} & {np.max(np.abs(p - q)):.2f} & "
                     f"{np.corrcoef(p, q)[0, 1]:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "surrogate.tex"), "w").write("\n".join(lines) + "\n")


def fig_pareto(d, rows):
    """AP against restored MAC share, house figure style (190 mm, 7-8 pt, Okabe-Ito colours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "pdf.fonttype": 42, "lines.linewidth": 1.0})
    mm = 1 / 25.4
    ms = [m for m in MODELS if m in rows]
    fig, axes = plt.subplots(2, 2, figsize=(190 * mm, 118 * mm), squeeze=False)
    style = {"mean": ("#0072B2", "o", "Measured, mean"),
             "minimax": ("#009E73", "s", "Measured, worst-group"),
             "sqnr": ("#E69F00", "P", "Output SQNR knapsack"),
             "recon": ("#56B4E9", "D", "Reconstruction-error knapsack"),
             "fisher": ("#D55E00", "^", "Fisher knapsack"),
             "autoquant": ("#CC79A7", "v", "auto_quantize"),
             "random": ("#999999", ".", "Random"),
             "headout": ("black", "+", "Head-out"),
             "stem_headout": ("black", "x", "Stem + head-out"),
             "head": ("black", "*", "Whole head")}
    for ax, m in zip(axes.flat, ms):
        pts = method_points(rows[m])
        ref = d["references"][m]
        if "head" not in pts and ref.get("int8ent_head") and ref.get("head_mac") is not None:
            pts["head"] = [(100 * ref["head_mac"], ref["int8ent_head"]["AP"], None, None)]
        for meth, (col, mk, lab) in style.items():
            p = sorted(pts.get(meth, []), key=lambda x: x[0])
            if not p:
                continue
            x, y = [q[0] for q in p], [q[1] for q in p]
            ls = "-" if meth in ("mean", "minimax", "fisher", "autoquant", "recon", "sqnr") else "none"
            z = 3 if meth == "mean" else 2
            ax.plot(x, y, color=col, marker=mk, ls=ls, lw=1.0 if meth != "mean" else 1.4,
                    ms=3.5 if mk not in (".", "*") else 5, mew=0.75, label=lab, zorder=z)
        fp32 = (ref.get("fp32_val") or ref.get("fp32"))["AP"]
        ax.axhline(fp32, color="0.35", ls=(0, (1, 1.5)), lw=0.75, label="FP32")
        ax.set_title(NAME[m], fontsize=8, pad=3)
        ax.grid(color="#D9D9D9", lw=0.4)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("FP16 share of convolution MACs (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("AP, COCO val2017")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=0.8, w_pad=1.2)
    fig.savefig(os.path.join(F, "pareto_coco.pdf"))


def table_fisher_n(rows):
    """Fisher knapsack scored on 512 vs 2000 tuning images (development models)."""
    lines = [r"\begin{tabular}{@{}ll" + "c" * 4 + "@{}}", r"\toprule",
             r"Model & Fisher score on & $B=5\%$ & $10\%$ & $15\%$ & $20\%$\\", r"\midrule"]
    for m in ("yolo11s", "yolo26s"):
        if m not in rows:
            continue
        for pre, lab in (("", "512 images"), ("n2000_", "2000 images")):
            cells = []
            for B in BUDGETS[1:]:
                c = rows[m].get(f"{pre}fisher_B{B}")
                cells.append(f"{c['splits']['val']['AP']:.2f} ({100 * c['mac_actual']:.1f})" if c and "val" in c["splits"] else "--")
            lines.append(f"{NAME[m] if not pre else ''} & {lab} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(T, "fisher_n.tex"), "w").write("\n".join(lines) + "\n")


def table_blocks():
    """Per-block MAC share and single-block recovery on the tuning sets (appendix)."""
    P = os.path.join(HERE, "data", "plans")
    a11 = json.load(open(os.path.join(P, "alloc_yolo11s_int8ent_ho_v2.json")))
    a26 = json.load(open(os.path.join(P, "alloc_yolo26s_int8ent_ho_v2.json")))
    ex = json.load(open(os.path.join(P, "alloc_yolo11s_int8ent_exdark_ho.json")))
    assert a11["blocks"] == ex["blocks"] and len(a11["blocks"]) == len(a26["blocks"])
    low = 1 + ex["group_names"].index("Low")
    lines = [r"\begin{tabular}{@{}l rrrrr rr l rrrrr@{}}", r"\toprule",
             r" & \multicolumn{7}{c}{YOLO11-s} & & \multicolumn{5}{c}{YOLO26-s}\\",
             r"\cmidrule(lr){2-8}\cmidrule(l){10-14}",
             r"Block & MAC & $\hat r_0$ & $\hat r_S$ & $\hat r_M$ & $\hat r_L$ & $\hat r_0^{\mathrm{ExD}}$ & $\hat r_{\mathrm{Low}}^{\mathrm{ExD}}$ & "
             r"Block & MAC & $\hat r_0$ & $\hat r_S$ & $\hat r_M$ & $\hat r_L$\\", r"\midrule"]
    for i, b in enumerate(a11["blocks"]):
        r, q, e = a11["r"][i], a26["r"][i], ex["r"][i]
        b26 = a26["blocks"][i].replace("one2one_", "o2o.")
        lines.append(f"\\texttt{{{b}}} & {100 * a11['cost'][i]:.1f} & " + " & ".join(f"{x:+.2f}" for x in r) +
                     f" & {e[0]:+.2f} & {e[low]:+.2f} & \\texttt{{{b26}}} & {100 * a26['cost'][i]:.1f} & " +
                     " & ".join(f"{x:+.2f}" for x in q) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "blocks.tex"), "w").write("\n".join(lines) + "\n")


def load_boots():
    out = {}
    for p in glob.glob(os.path.join(HERE, "data", "bootstrap", "g_*.json")):
        b = json.load(open(p))
        name = os.path.basename(p)
        m = name.split("_")[1]
        kind = "exdark" if "exdark" in name else "coco"
        for pk, v in b["diff_b_minus_a"].items():
            out[(m, kind, pk)] = v
    for p in glob.glob(os.path.join(HERE, "data", "bootstrap", "yolo*.json")):
        b = json.load(open(p))
        name = os.path.basename(p)[:-5]
        m = name.split("_")[0]
        mm = re.search(r"_(mean_matched|mean|minimax|fisher)_vs_(mean_matched|mean|minimax|fisher)_B([\d.]+)$", name)
        if mm:  # g_* files were loaded first and take precedence for the same pair
            bb, aa, B = mm.groups()  # file "x_vs_y": a=y (reference side), b=x
            out.setdefault((m, "coco", f"{aa}_B{B}:{bb}_B{B}"), b["diff_b_minus_a"])
    return out


def pair_key(pk):
    """Sort pairs by the method of the A side, then numerically by budget."""
    a, b = pk.split(":")
    bud = re.findall(r"_B([\d.]+)", pk)
    return (re.sub(r"_B[\d.]+", "", a), re.sub(r"_B[\d.]+", "", b), float(bud[0]) if bud else -1)


def table_boot(boots):
    hdr = r"Model (data) & $B$ (\%) & B vs A & $\Delta$AP [95\% CI] & $\Delta$worst [95\% CI]\\"
    lines = [r"\begin{scriptsize}", r"\setlength{\tabcolsep}{5pt}", r"\begin{longtable}{@{}llllr@{}}",
             r"\caption{Paired image-level bootstrap ($1000$ resamples) of every comparison quoted in the text. $\Delta$AP is B minus A in AP points; $\Delta$worst is the worst-group relative loss of A minus that of B in percentage points, so positive values favour B in both columns. Groups are the COCO size groups or the ExDark lighting types; FP32 is the reference. $B$ is the nominal budget of the budgeted selection(s); stem, head and stem + head-out are fixed heuristics. Entries are means over the resamples with percentile intervals; because the worst-group loss is a maximum over groups, its bootstrap mean can differ from the full-set estimate reported in Table~\ref{M-tab:objective} of the article, while the intervals are the same.}\label{tab:bootstrap}\\\\",
             r"\toprule", hdr, r"\midrule", r"\endfirsthead", r"\toprule", hdr, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endlastfoot"]
    nice = {"mean_matched": "matched", "stem_headout": "stem+ho", "sim_mean": "sim-AP", "sim256_mean": "sim-AP (256)",
            "best33": "best of 33"}
    for (m, kind, pk), v in sorted(boots.items(), key=lambda x: (MODELS.index(x[0][0]) if x[0][0] in MODELS else 9, x[0][1], pair_key(x[0][2]))):
        a, b = pk.split(":")
        bud = sorted(set(re.findall(r"_B([\d.]+)", pk)), key=float)
        meth = [nice.get(re.sub(r"_B[\d.]+$", "", t), re.sub(r"_B[\d.]+$", "", t)) for t in (b, a)]
        ap = v["AP"]
        w = v.get("worst_rel")
        lines.append(f"{NAME.get(m, m)} ({'ExDark' if kind == 'exdark' else 'COCO'}) & "
                     f"{'/'.join(f'{100 * float(x):g}' for x in bud)} & {meth[0]} vs {meth[1]} & "
                     f"{ap['mean']:+.2f} [{ap['ci95'][0]:+.2f},{ap['ci95'][1]:+.2f}] & " +
                     (f"{w['mean']:+.2f} [{w['ci95'][0]:+.2f},{w['ci95'][1]:+.2f}]" if w else "--") + r"\\")
    lines += [r"\end{longtable}", r"\end{scriptsize}"]
    open(os.path.join(T, "bootstrap.tex"), "w").write("\n".join(lines) + "\n")


def table_ablation():
    """Post-freeze ablations (README 11.6): greedy order, simulated AP on 2000 and 256 images, best of 33 random,
    each against the engine mean allocation. Points and paired intervals from data/bootstrap/g_*.json."""
    pts, diff = {}, {}
    for p in glob.glob(os.path.join(HERE, "data", "bootstrap", "g_*_coco_*.json")):
        b = json.load(open(p))
        m = os.path.basename(p).split("_")[1]
        for k, v in b["point"].items():
            pts[(m, k)] = v["AP"]
        for pk, v in b["diff_b_minus_a"].items():
            diff[(m, pk.split(":")[0])] = v["AP"]
    cols = [("greedy", "Greedy order"), ("sim_mean", "Sim-AP"), ("sim256_mean", "Sim-AP (256)"), ("best33", "Best of 33")]
    budgets = ["0.03", "0.05", "0.1", "0.15", "0.2"]
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Model & $B$ (\%) & Engine mean & " + " & ".join(c[1] for c in cols) + r"\\", r"\midrule"]
    for m in MODELS:
        for i, B in enumerate(budgets):
            ref = pts.get((m, f"mean_B{B}"))
            cells = []
            for c, _ in cols:
                k = f"{c}_B{B}"
                if (m, k) in diff:
                    d = diff[(m, k)]  # engine mean minus this method
                    mark = r"$^{-}$" if d["ci95"][0] > 0 else (r"$^{+}$" if d["ci95"][1] < 0 else "")
                    cells.append(f"{pts[(m, k)]:.2f}{mark}")
                elif c == "best33" and B in ("0.03", "0.15"):
                    cells.append("--")
                else:
                    cells.append("$=$")
            lines.append((NAME[m] if i == 0 else "") + f" & {100 * float(B):g} & " +
                         (f"{ref:.2f}" if ref is not None else "--") + " & " + " & ".join(cells) + r"\\")
        if m != MODELS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "ablation.tex"), "w").write("\n".join(lines) + "\n")


def table_latency(d, rows):
    """Jetson Orin latency/power per engine: median over repetitions of the trtexec GPU median.

    MAC share and AP are those of the evaluated configuration (COCO val, 5090 engine); FP16 carries the AP of
    the FP32 engine. Energy per frame is latency times mean board power (VDD_IN)."""
    files = sorted(glob.glob(os.path.join(HERE, "data", "orin", "*_bench.jsonl")),
                   key=lambda f: MODELS.index(os.path.basename(f).replace("_bench.jsonl", "")))
    if not files:
        return
    order = ["fp16", "base", "headout", "stem_headout", "head"] + \
        [f"{k}_B{b}" for k in ("mean", "minimax", "fisher", "sqnr") for b in BUDGETS] + [f"autoquant_B{b}" for b in BITS]
    label = {"fp16": r"FP16 (no Q/DQ)$^\dagger$", "base": "INT8 (entropy)", "headout": r"INT8 + $\mathcal{F}$",
             "stem_headout": "+ stem", "head": "whole head FP16"}
    kind = {"mean": "Engine, mean", "minimax": "Engine, worst-group", "fisher": "Fisher knapsack",
            "sqnr": r"Output SQNR", "autoquant": r"\texttt{auto\_quantize}"}
    lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Model & Configuration & MAC (\%) & AP & latency (ms) & vs.\ INT8 (\%) & power (W) & energy (mJ) & engine (MB)\\",
             r"\midrule"]
    for f in files:
        m = os.path.basename(f).replace("_bench.jsonl", "")
        man = json.load(open(f.replace("_bench.jsonl", "_manifest.json")))
        by = defaultdict(list)
        for ln in open(f):
            r = json.loads(ln)
            if r.get("median_ms") is not None:
                by[r["engine"]].append(r)
        if "base" not in by:
            continue
        lat = {e: float(np.median([r["median_ms"] for r in v])) for e, v in by.items()}
        pw = {e: float(np.median([r["vdd_in_mw"] for r in v])) / 1000 for e, v in by.items()}
        size = {e: v[0]["engine_bytes"] / 2 ** 20 for e, v in by.items()}
        first = True
        for e in order:
            if e not in lat:
                continue
            c = rows[m].get(e)
            if e == "fp16":
                mac, ap = 100.0, d["references"][m]["fp32_val"]["AP"]
            elif e == "base":
                mac, ap = 0.0, d["references"][m]["int8ent"]["AP"]
            else:
                mac = 100 * (c["mac_actual"] if c else (man.get(e, {}).get("cost_macs_frac") or 0.0))
                ap = c["splits"]["val"]["AP"] if c and "val" in c["splits"] else None
            mm = re.match(r"(\w+?)_B([\d.]+)$", e)
            lab = label.get(e) or (f"{kind[mm.group(1)]} $B={'%g' % (100 * float(mm.group(2)))}\\%$" if mm.group(1) != "autoquant"
                                   else f"{kind['autoquant']} {mm.group(2)} bits")
            lines.append(f"{NAME[m] if first else ''} & {lab} & {mac:.1f} & {fmt(ap)} & {lat[e]:.2f} & "
                         f"{100 * (lat[e] / lat['base'] - 1):+.1f} & {pw[e]:.1f} & {lat[e] * pw[e]:.0f} & {size[e]:.1f}\\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(os.path.join(T, "latency.tex"), "w").write("\n".join(lines) + "\n")


def main():
    d, rows = load(sys.argv[1])
    os.makedirs(T, exist_ok=True)
    os.makedirs(F, exist_ok=True)
    boots = load_boots()
    table_refs(d, rows)
    table_coco(d, rows)
    table_exdark(d, rows)
    table_surrogate(d, rows)
    table_objective(d, rows, boots)
    table_boot(boots)
    table_ablation()
    table_fisher_n(rows)
    table_blocks()
    table_latency(d, rows)
    fig_pareto(d, rows)
    print("tables:", sorted(os.listdir(T)))


if __name__ == "__main__":
    main()

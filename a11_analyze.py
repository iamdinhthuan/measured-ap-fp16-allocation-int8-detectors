"""rules_11_11.md A analysis: pandey:mean and sqnrfx:pandey (a11), with sqnrfx:mean from the main-text files (f1/f1b).
p = exact two-sided bootstrap p, floor 2/(n+1); identical prediction files -> p = 1, not significant (as y2_analyze.py).
Holm within all 20 pairs and within the 10 large-detector pairs."""
import json, os
os.chdir("/home/thuan/isci_method_20261025")
MAIN = {"yolo26l": "f1b", "yolo11l": "f1", "yolo26s": "f1", "yolo11s": "f1"}

def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); rej = [False] * len(ps)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - k): rej[i] = True
        else: break
    return rej

def row(d, pair):
    a, b = pair.split(":"); v = d["diff_b_minus_a"][pair]["AP"]; n = d["n"]
    same = d["sets"][a] == d["sets"][b]
    q = v["p_le0"]; p = 1.0 if same else max(2 * min(q, 1 - q), 2 / (n + 1))
    sig = (not same) and (v["ci95"][0] > 0 or v["ci95"][1] < 0)
    return dict(pt=d["point"][b]["AP"] - d["point"][a]["AP"], ci=v["ci95"], p=p, sig=sig, same=same)

R = []
for m, tag in [(k, t) for k, t in MAIN.items() if os.path.exists(f"metrics/bootstrap/fx_{k}_a11.json")]:
    d = json.load(open(f"metrics/bootstrap/fx_{m}_a11.json")); o = json.load(open(f"metrics/bootstrap/fx_{m}_{tag}.json"))
    for B in ("0.03", "0.05", "0.1", "0.15", "0.2"):
        r = dict(m=m, B=B, large=m.endswith("l"))
        r["pm"] = row(d, f"pandey_B{B}:mean_B{B}")      # >0: measured better than per-image SQNR
        r["sp"] = row(d, f"sqnrfx_B{B}:pandey_B{B}")    # >0: per-image better than pooled
        r["sm"] = row(o, f"sqnrfx_B{B}:mean_B{B}")      # main-text pooled vs measured
        assert d["sets"][f"mean_B{B}"] == o["sets"][f"mean_B{B}"] and d["sets"][f"sqnrfx_B{B}"] == o["sets"][f"sqnrfx_B{B}"], (m, B)
        R.append(r)
for fam, sel in (("all20", lambda r: True), ("large10", lambda r: r["large"])):
    S = [r for r in R if sel(r)]
    for k in ("pm", "sp", "sm"):
        for r, h in zip(S, holm([r[k]["p"] for r in S])): r[k]["holm_" + fam] = h
for r in R:
    f = lambda x: f"{x['pt']:+.2f} [{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}]{'*' if x['sig'] else ' '}{'H' if x.get('holm_all20') else ' '}{'h' if x.get('holm_large10') else ' '}{'=' if x['same'] else ' '}"
    print(f"{r['m']:8s} {r['B']:5s} meas-pandey {f(r['pm'])} | pandey-pooled {f(r['sp'])} | meas-pooled {f(r['sm'])}")
for fam, sel in (("all20", lambda r: True), ("large10", lambda r: r["large"])):
    S = [r for r in R if sel(r)]
    c = lambda k, cond: sum(1 for r in S if cond(r[k]))
    both = sum(1 for r in S if r["pm"]["sig"] and r["pm"]["pt"] > 0 and r["sm"]["sig"] and r["sm"]["pt"] > 0)
    bothH = sum(1 for r in S if r["pm"].get("holm_" + fam) and r["pm"]["pt"] > 0 and r["sm"].get("holm_" + fam) and r["sm"]["pt"] > 0)
    print(f"{fam}: meas>pandey sig {c('pm', lambda x: x['sig'] and x['pt']>0)} (holm {c('pm', lambda x: x.get('holm_'+fam) and x['pt']>0)}), "
          f"pandey>meas sig {c('pm', lambda x: x['sig'] and x['pt']<0)}; meas>pooled sig {c('sm', lambda x: x['sig'] and x['pt']>0)} (holm {c('sm', lambda x: x.get('holm_'+fam) and x['pt']>0)}); "
          f"both-forms sig {both} (holm {bothH}); pandey>pooled sig {c('sp', lambda x: x['sig'] and x['pt']>0)}, pooled>pandey sig {c('sp', lambda x: x['sig'] and x['pt']<0)}; identical pandey/pooled {c('sp', lambda x: x['same'])}")
    pos = [r["pm"]["pt"] for r in S if r["pm"]["sig"] and r["pm"]["pt"] > 0]
    if pos: print(f"  meas>pandey margins {min(pos):.2f}..{max(pos):.2f}; max meas-pandey {max(r['pm']['pt'] for r in S):.2f}")

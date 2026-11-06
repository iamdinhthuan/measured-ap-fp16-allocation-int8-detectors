"""rules_11_11c.md C analysis: fit:mean (>0 measured better than FIT) and fisherblk:fit (>0 FIT better than block Fisher).
Conventions as a11_analyze.py (exact two-sided p floor 2/(n+1); identical files -> p=1, not significant; Holm within all20 and large10)."""
import json, os
os.chdir("/home/thuan/isci_method_20261025")
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
    return dict(pt=d["point"][b]["AP"] - d["point"][a]["AP"], ci=v["ci95"], p=p, sig=sig, same=same, A=d["point"][a]["AP"])
R = []
for m in ("yolo11s", "yolo26s", "yolo11l", "yolo26l"):
    f = f"metrics/bootstrap/fx_{m}_c11.json"
    if not os.path.exists(f): print("missing", m); continue
    d = json.load(open(f))
    for B in ("0.03", "0.05", "0.1", "0.15", "0.2"):
        R.append(dict(m=m, B=B, large=m.endswith("l"), fm=row(d, f"fit_B{B}:mean_B{B}"), bf=row(d, f"fisherblk_B{B}:fit_B{B}")))
for fam, sel in (("all20", lambda r: True), ("large10", lambda r: r["large"])):
    S = [r for r in R if sel(r)]
    for k in ("fm", "bf"):
        for r, h in zip(S, holm([r[k]["p"] for r in S])): r[k]["holm_" + fam] = h
for r in R:
    f = lambda x: f"{x['pt']:+.2f} [{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}]{'*' if x['sig'] else ' '}{'H' if x.get('holm_all20') else ' '}{'h' if x.get('holm_large10') else ' '}{'=' if x['same'] else ' '}"
    print(f"{r['m']:8s} {r['B']:5s} FIT {r['fm']['A']:.2f} meas-fit {f(r['fm'])} | fit-fisherblk {f(r['bf'])}")
for fam, sel in (("all20", lambda r: True), ("large10", lambda r: r["large"])):
    S = [r for r in R if sel(r)]
    c = lambda k, cond: sum(1 for r in S if cond(r[k]))
    print(f"{fam} (n={len(S)}): meas>fit sig {c('fm', lambda x: x['sig'] and x['pt']>0)} (holm {c('fm', lambda x: x.get('holm_'+fam) and x['pt']>0)}), fit>meas sig {c('fm', lambda x: x['sig'] and x['pt']<0)}; "
          f"fit>fisherblk sig {c('bf', lambda x: x['sig'] and x['pt']>0)}, fisherblk>fit sig {c('bf', lambda x: x['sig'] and x['pt']<0)}, identical {c('bf', lambda x: x['same'])}")
    pos = [r["fm"]["pt"] for r in S if r["fm"]["sig"] and r["fm"]["pt"] > 0]
    if pos: print(f"  meas>fit margins {min(pos):.2f}..{max(pos):.2f}")

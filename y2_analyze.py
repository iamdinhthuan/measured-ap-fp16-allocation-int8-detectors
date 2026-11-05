"""Y2 analysis (rules_11_10.md): 10k vs 1k SQNR pairs, exact two-sided p (floor 2/(n+1)), Holm in 20 and 10-large families."""
import json, os
os.chdir("/home/thuan/isci_method_20261025")
MODELS = [("yolo26l", "f1b"), ("yolo11l", "f1"), ("yolo26s", "f1"), ("yolo11s", "f1")]
rows = []
for m, tag in MODELS:
    new_p = f"metrics/bootstrap/fx_{m}_" + (tag if os.environ.get("Y2TEST") else "y2_10k") + ".json"
    if not os.path.exists(new_p):
        print("MISSING", new_p); continue
    new, old = json.load(open(new_p)), json.load(open(f"metrics/bootstrap/fx_{m}_{tag}.json"))
    n = new["n"]
    for pair, v in new["diff_b_minus_a"].items():
        if not pair.startswith("sqnrfx_"):
            continue
        a, b = pair.split(":")
        same = new["sets"][a] == new["sets"][b]
        pt_new = new["point"][b]["AP"] - new["point"][a]["AP"]
        pt_old = old["point"][b]["AP"] - old["point"][a]["AP"]
        assert abs(pt_new - pt_old) < 1e-9, (m, pair, pt_new, pt_old)
        assert new["sets"][a] == old["sets"][a] and new["sets"][b] == old["sets"][b], (m, pair)
        q = v["AP"]["p_le0"]
        p = 1.0 if same else max(2 * min(q, 1 - q), 2 / (n + 1))
        sig = (not same) and (v["AP"]["ci95"][0] > 0 or v["AP"]["ci95"][1] < 0)
        o = old["diff_b_minus_a"][pair]["AP"]
        q1 = o["p_le0"]; p1 = 1.0 if same else max(2 * min(q1, 1 - q1), 2 / 1001)
        sig1 = (not same) and (o["ci95"][0] > 0 or o["ci95"][1] < 0)
        rows.append(dict(m=m, pair=pair, large=m.endswith("l"), pt=pt_new, ci=v["AP"]["ci95"], p=p, sig=sig,
                         ci1=o["ci95"], p1=p1, sig1=sig1, same=same))

def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); rej = [False] * len(ps)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - k): rej[i] = True
        else: break
    return rej

for fam, sel in (("all20", lambda r: True), ("large10", lambda r: r["large"])):
    R = [r for r in rows if sel(r)]
    h10 = holm([r["p"] for r in R]); h1 = holm([r["p1"] for r in R])
    for r, x, y in zip(R, h10, h1):
        r[f"holm_{fam}"], r[f"holm1_{fam}"] = x, y
    print(f"{fam}: n_pairs={len(R)} sig10k={sum(r['sig'] for r in R)} holm10k={sum(h10)} | sig1k={sum(r['sig1'] for r in R)} holm1k={sum(h1)}")
for r in rows:
    print(f"{r['m']:8s} {r['pair']:28s} pt={r['pt']:+.3f} ci10k=[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p10k={r['p']:.5f} sig={int(r['sig'])} "
          f"H20={int(r.get('holm_all20',0))} H10={int(r.get('holm_large10',0))} | ci1k=[{r['ci1'][0]:+.3f},{r['ci1'][1]:+.3f}] p1k={r['p1']:.4f} "
          f"sig={int(r['sig1'])} H20={int(r.get('holm1_all20',0))} H10={int(r.get('holm1_large10',0))} same={int(r['same'])}")
if len(rows) == 20 and not os.environ.get("Y2TEST"):
    json.dump(rows, open("metrics/bootstrap/y2_summary.json", "w"), indent=1)

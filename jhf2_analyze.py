"""Apply the JHF2 reporting rule (rules_20261109_jhf2.md section 2).
Usage: python3 jhf2_analyze.py <repo> <bench.jsonl> <out.json>"""
import json, sys, statistics as st
repo, bp, op = sys.argv[1:4]
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if "error" in r: print("ERROR", r); continue
        d.setdefault(r["engine"], []).append(r)
    return d
S = load(bp)
T5 = load(f"{repo}/metrics/orin_latency/yolo26l_bench.jsonl")
def summ(rs):
    ms = st.median(r["median_ms"] for r in rs); w = st.mean(r["vdd_in_mw"] for r in rs) / 1000
    return dict(n=len(rs), ms=ms, W=w, mJ=w * ms, MiB=rs[0]["engine_bytes"] / 2**20, modes=sorted({r.get("power_mode", "") for r in rs}))
out = {k: summ(v) for k, v in S.items()}
b = out["base"]["ms"]
ae = f"{repo}/metrics/alloc_eval/yolo26l_int8ent/"
APF = {"hf_B0.1": "fxh_hyb_B0.1", "hf_B0.15": "fxh_hyb_B0.15", "hf_B0.2": "fxh_hyb_B0.2",
       "mean_B0.05": "mean_B0.05", "mean_B0.1": "mean_B0.1", "mean_B0.2": "mean_B0.2",
       "rb_mean_B0.1": "mean_B0.1", "rb_mean_B0.2": "mean_B0.2"}
for k, v in out.items():
    v["vs_int8_pct"] = 100 * (v["ms"] / b - 1)
    if k in APF:
        try: v["AP_val"] = 100 * json.load(open(ae + APF[k] + "_val.json"))["stats"]["AP"]
        except Exception as e: v["AP_val"] = None
t5b = st.median(r["median_ms"] for r in T5["base"])
cons = {}
for k in ["headout", "head", "mean_B0.05", "mean_B0.1", "mean_B0.2"]:
    t = st.median(r["median_ms"] for r in T5[k])
    cons[k] = dict(table5_ms=t, table5_vs_int8_pct=100 * (t / t5b - 1), jhf2_ms=out[k]["ms"], jhf2_vs_int8_pct=out[k]["vs_int8_pct"])
spread = max(abs(out[f"rb_mean_B{x}"]["ms"] - out[f"mean_B{x}"]["ms"]) / out[f"mean_B{x}"]["ms"] for x in ("0.1", "0.2"))
cmp = {}
for x in ("0.1", "0.2"):
    h, m = out[f"hf_B{x}"]["ms"], out[f"mean_B{x}"]["ms"]
    cmp[x] = dict(diff_ms=h - m, diff_pct=100 * (h / m - 1), within_spread=abs(h / m - 1) <= spread + 1e-12)
res = dict(engines=out, table5_int8_ms=t5b, consistency=cons, build_spread_pct=100 * spread, hf_vs_mean=cmp)
json.dump(res, open(op, "w"), indent=1)
for k in sorted(out): v = out[k]; print(f"{k:14s} {v['ms']:7.3f} ms {v['vs_int8_pct']:+6.2f}% {v['W']:5.2f} W {v['mJ']:6.1f} mJ {v['MiB']:5.1f} MiB n={v['n']} AP={v.get('AP_val')} {v['modes']}")
print("spread %", 100 * spread); print(json.dumps(cmp)); print(json.dumps(cons, indent=0))

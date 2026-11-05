"""E3: pick, per budget, the random allocation with the best tuning AP (README 11.6)."""
import json
import sys

R, m = sys.argv[1:3]
p = json.load(open(R))
out = dict(p, allocations={}, criterion="best of 33 random (tuning AP)")
for B in ("0.05", "0.1", "0.2"):
    ks = [k for k in p["allocations"] if k.startswith(f"random@{B}_")]
    ap = {k: json.load(open(f"metrics/alloc_eval/{m}_int8ent/r33_{k.replace('@', '_B')}.json"))["splits"]["tune"]["stats"]["AP"]
          for k in ks}
    assert len(ap) == 33, (m, B, len(ap))
    best = max(ap, key=ap.get)
    out["allocations"][f"best33@{B}"] = dict(p["allocations"][best], source_key=best, tune_AP=ap[best], tune_AP_all=ap)
    print(m, B, best, round(100 * ap[best], 2), "median", round(100 * sorted(ap.values())[len(ap) // 2], 2), flush=True)
fn = R.replace("random33_", "best33_")
assert fn != R
json.dump(out, open(fn, "w"), indent=1)

"""JHF2 graphs: Table 5 YOLO26-l graphs re-derived from the hp32 Jetson base (base.onnx, sha 65c7e25f...) and the head-first
graphs. Gate: each re-derived Table 5 graph equals its manifest sha256. Usage: python3 jhf2_export.py <repo> <dir>"""
import hashlib, json, os, sys
import onnx
repo, wd = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from qdq_restore import count_qdq, restore
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
base = os.path.join(wd, "base.onnx"); assert sha(base).startswith("65c7e25f")
man = json.load(open(os.path.join(repo, "metrics/orin_latency/yolo26l_manifest.json")))
ev = os.path.join(repo, "metrics/alloc_eval/yolo26l_int8ent")
out = {"base": {"sha256": sha(base)}}
def mk(tag, rx):
    m = onnx.load(base); n = count_qdq(m, rx); restore(m, rx); p = os.path.join(wd, f"{tag}.onnx"); onnx.save(m, p); return n, sha(p)
for tag in ["headout", "head", "mean_B0.05", "mean_B0.1", "mean_B0.2"]:
    n, s = mk(tag, man[tag]["regex"])
    ok = s == man[tag]["sha256"] and n == man[tag]["qdq_inputs_restored"]
    print("GATE", tag, ok, n); assert ok
    out[tag] = {"regex": man[tag]["regex"], "cost_macs_frac": man[tag]["cost_macs_frac"], "qdq_inputs_restored": n, "sha256": s, "table5_graph": True}
for b in ["0.1", "0.15", "0.2"]:
    r = json.load(open(os.path.join(ev, f"fxh_hyb_B{b}.json")))
    n, s = mk(f"hf_B{b}", r["regex"]); assert n == r["qdq_inputs_restored"]
    out[f"hf_B{b}"] = {"source": f"metrics/alloc_eval/yolo26l_int8ent/fxh_hyb_B{b}.json", "regex": r["regex"],
                       "cost_macs_frac": r["cost_macs_frac"], "qdq_inputs_restored": n, "sha256": s}
    print("HF", b, n, s[:16])
json.dump(out, open(os.path.join(wd, "manifest_jhf2.json"), "w"), indent=1)

"""Build and measure allocations chosen by allocate.py.

For each selected allocation the union of its block regexes is restored to
FP16 in the base Q/DQ graph, compiled once, and evaluated on the tuning split
(measured vs additive prediction) and on val2017 (held out; never used to pick
blocks or budgets). Predictions are saved for paired bootstrap.

Usage: python eval_alloc.py --alloc metrics/alloc_yolo11s_int8max.json --keys mean@0.1,minimax@0.1
"""
import argparse
import json
import os
import re
import time

import onnx

from allocate import conv_macs
from block_sensitivity import ROOT, latency_ms, run
from qdq_restore import count_qdq, restore

PY = os.sys.executable
SPLITS = {
    "tune": (os.path.join(ROOT, "data", "tune2000", "instances_tune.json"),
             os.path.join(ROOT, "data", "tune2000", "images")),
    "val": (os.path.join(ROOT, "data", "coco_src", "annotations", "instances_val2017.json"),
            os.path.join(ROOT, "data", "coco", "images", "val2017")),
    **{f"exdark_{sp}": (os.path.join(ROOT, "data", "exdark", f"exdark_{sp}.json"),
                        os.path.join(ROOT, "data", "exdark_raw", "ExDark")) for sp in ("train", "val", "test")},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--splits", default="tune,val")
    ap.add_argument("--prefix", default="", help="prepended to output tags, keeps plans from colliding")
    a = ap.parse_args()
    os.chdir(ROOT)
    plan = json.load(open(a.alloc))
    sfx = "_exdark" if plan.get("split") == "exdark" else ""
    sens_path = os.path.join(ROOT, "metrics", f"sens_{plan['model']}_{plan['base']}{sfx}.json")
    base_onnx = os.path.join(ROOT, "exports", f"{plan['model']}_{plan['base']}.onnx")
    outdir = os.path.join(ROOT, "metrics", "alloc_eval", f"{plan['model']}_{plan['base']}")
    os.makedirs(outdir, exist_ok=True)
    log = os.path.join(ROOT, "logs", f"alloc_eval_{plan['model']}_{plan['base']}.log")
    for key in a.keys.split(","):
        sel = plan["allocations"][key]
        tag = a.prefix + key.replace("@", "_B")
        if os.path.exists(os.path.join(outdir, f"{tag}.json")):
            raise SystemExit(f"{tag}.json exists in {outdir}; use a new --prefix instead of overwriting")
        # manual configurations (folklore baselines) carry their own regex
        rx = sel.get("regex") or "|".join(f"(?:{json.load(open(sens_path))['blocks'][b]})" for b in sel["blocks"])
        if plan.get("force_regex"):
            rx = f"(?:{plan['force_regex']})" + (f"|{rx}" if rx else "")
        if sel.get("cost") is None:  # manual configs: MAC fraction of the Conv nodes the regex restores
            macs = conv_macs(os.path.join(ROOT, "exports", f"{plan['model']}.onnx"))
            sel["cost"] = sum(v for k, v in macs.items() if re.search(rx, k)) / sum(macs.values())
        onnx_path = os.path.join(ROOT, "exports", "tmp", f"{plan['model']}_{plan['base']}_{tag}.onnx")
        eng = onnx_path.replace(".onnx", ".plan")
        m = onnx.load(base_onnx)
        n_q = count_qdq(m, rx)
        restore(m, rx)
        onnx.save(m, onnx_path)
        t0 = time.time()
        run(f"{PY} tools/build_engine.py --onnx {onnx_path} --engine {eng}", log)
        res = {"key": key, "blocks": sel["blocks"], "regex": rx, "qdq_inputs_restored": n_q,
               "cost_macs_frac": sel["cost"], "pred_gap_AP_tune": sel.get("pred_gap_AP"),
               "pred_gap_SML_tune": sel.get("pred_gap_SML"), "build_s": round(time.time() - t0, 1),
               "latency_ms": latency_ms(eng), "engine_bytes": os.path.getsize(eng), "splits": {}}
        for sp in a.splits.split(","):
            gt, img = SPLITS[sp]
            out = os.path.join(outdir, f"{tag}_{sp}.json")
            run(f"{PY} infer_eval.py --engine {eng} --gt {gt} --images {img} --out {out} --save-pred", log)
            res["splits"][sp] = json.load(open(out))
        json.dump(res, open(os.path.join(outdir, f"{tag}.json"), "w"), indent=1)
        os.remove(eng)
        os.remove(onnx_path)
        s = {sp: {k: round(100 * v["stats"][k], 2) for k in ("AP", "AP_small", "AP_medium", "AP_large")}
             for sp, v in res["splits"].items()}
        print(f"{key:12s} cost={sel['cost']:.3f} lat={res['latency_ms']:.3f} {s}", flush=True)
    print("ALLOC_DONE", flush=True)


if __name__ == "__main__":
    main()

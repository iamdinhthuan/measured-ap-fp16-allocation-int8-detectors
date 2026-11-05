"""M8 (rules_11_10.md, appended before any result): YOLO26-l block sweep with the forced set F already restored.

Copy of block_sensitivity.py main(); the published script is untouched. Reference engine = INT8 + F; each block b is
restored on top of F (r_b = AP(F u b) - AP(F)). FP32 and all-FP16 rows reuse the published tune evaluations (same
tags, read from metrics/tune/<model>/). New tags: int8entF, int8entF_restore_<block>.
Output (never overwritten): metrics/sens_<model>_int8ent_fbase.json; its row "int8ent" is the INT8 + F engine so that
allocate.py (stats[s['base']]) uses it as the reference.
Usage: python fx_sens_f.py --model yolo26l
"""
import argparse
import json
import os
import sys

import onnx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from block_sensitivity import blocks, evaluate_variant  # noqa: E402
from qdq_restore import count_qdq, restore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    a.split, a.base = "tune", "int8ent"
    os.chdir(ROOT)
    out = os.path.join(ROOT, "metrics", f"sens_{a.model}_int8ent_fbase.json")
    if os.path.exists(out) and json.load(open(out)).get("done"):
        raise SystemExit(f"{out} exists and is complete")
    log = os.path.join(ROOT, "logs", f"sens_{a.model}_int8ent_fbase.log")
    pub = json.load(open(os.path.join(ROOT, "metrics", f"sens_{a.model}_int8ent.json")))
    frx = json.load(open(os.path.join(ROOT, "metrics", f"alloc_{a.model}_int8ent_ho.json")))["force_regex"]
    assert frx
    base_onnx = os.path.join(ROOT, "exports", f"{a.model}_int8ent.onnx")
    tmp = os.path.join(ROOT, "exports", "tmp")
    m = onnx.load(base_onnx)
    bl = blocks(m)
    assert bl == pub["blocks"], "block partition differs from the published sweep"
    summary = {"model": a.model, "base": "int8ent", "base_row": "INT8 + F (tag int8entF)", "force_regex": frx,
               "split": "tune", "gt": pub["gt"], "blocks": bl, "rows": {}}
    summary["rows"]["fp32"] = evaluate_variant("fp32", os.path.join(ROOT, "exports", f"{a.model}.onnx"), a, log)
    fpath = os.path.join(tmp, f"{a.model}_int8entF.onnx")
    mm = onnx.load(base_onnx)
    nf = restore(mm, frx)
    onnx.save(mm, fpath)
    summary["rows"]["int8ent"] = evaluate_variant("int8entF", fpath, a, log)
    summary["rows"]["int8ent"]["qdq_inputs_F"] = nf
    os.remove(fpath)
    allf = os.path.join(tmp, f"{a.model}_int8ent_restore_all.onnx")
    if not os.path.exists(allf):
        mm = onnx.load(base_onnx)
        restore(mm, ".")
        onnx.save(mm, allf)
    summary["rows"]["fp16_all"] = evaluate_variant("int8ent_restore_all", allf, a, log)
    b = summary["rows"]["int8ent"]["stats"]
    print(f"F row AP={100*b['AP']:.2f} fp16_all AP={100*summary['rows']['fp16_all']['stats']['AP']:.2f}", flush=True)
    for name, rx in bl.items():
        tag = f"int8entF_restore_{name}"
        path = os.path.join(tmp, f"{a.model}_{tag}.onnx")
        mm = onnx.load(base_onnx)
        restore(mm, frx)
        restore(mm, rx)
        onnx.save(mm, path)
        r = evaluate_variant(tag, path, a, log)
        r["qdq_inputs"] = count_qdq(m, rx)
        summary["rows"][name] = r
        os.remove(path)
        print(f"{name:20s} dAP={100*(r['stats']['AP']-b['AP']):+.2f} dS={100*(r['stats']['AP_small']-b['AP_small']):+.2f} "
              f"dL={100*(r['stats']['AP_large']-b['AP_large']):+.2f}", flush=True)
        json.dump(summary, open(out, "w"), indent=1)
    summary["done"] = True
    json.dump(summary, open(out, "w"), indent=1)
    print("SENS_F_DONE", flush=True)


if __name__ == "__main__":
    main()

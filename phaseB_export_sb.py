"""Phase B pilot: single-block restoration graphs on top of the forced head-out set, for latency on the Orin.

For every block b of the sensitivity sweep, restores (forced set | b) in the same hp32 base as orin_export.py.
A block whose union restores no more Q/DQ inputs than the forced set alone gives the head-out graph again
and is skipped (nothing to time). Writes exports/orin/{model}_sb/sb_{block}.onnx and a manifest; never
overwrites an existing graph.
Usage: python phaseB_export_sb.py --model yolo26l
"""
import argparse
import hashlib
import json
import os

import onnx

from allocate import conv_macs
from qdq_restore import count_qdq, restore

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="int8ent")
    a = ap.parse_args()
    os.chdir(ROOT)
    sens = json.load(open(f"metrics/sens_sim_{a.model}_{a.base}.json"))
    plan = json.load(open(f"metrics/alloc_{a.model}_{a.base}_ho.json"))
    ho = plan["force_regex"]
    hp32 = os.path.join("exports", f"{a.model}_{a.base}_hp32.onnx")
    macs = conv_macs(os.path.join("exports", f"{a.model}.onnx"))
    total = sum(macs.values())
    outdir = os.path.join("exports", "orin", f"{a.model}_sb")
    os.makedirs(outdir, exist_ok=True)
    man_path = os.path.join(outdir, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    n_ho = count_qdq(onnx.load(hp32), ho)
    man["_forced"] = {"regex": ho, "qdq_inputs_restored": n_ho, "hp32": hp32}
    for b, brx in sens["blocks"].items():
        rx = f"(?:{ho})|(?:{brx})"
        m = onnx.load(hp32)
        n_q = count_qdq(m, rx)
        if n_q == n_ho:
            man[f"sb_{b}"] = {"regex": rx, "skipped": "same graph as forced set"}
            print(f"sb_{b}: no extra Q/DQ input, skipped", flush=True)
            continue
        out = os.path.join(outdir, f"sb_{b}.onnx")
        assert not os.path.exists(out), out
        restore(m, rx)
        onnx.save(m, out)
        cost = sum(v for k, v in macs.items() if __import__("re").search(rx, k)) / total
        man[f"sb_{b}"] = {"block": b, "regex": rx, "cost_macs_frac": cost, "qdq_inputs_restored": n_q,
                          "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
        print(f"sb_{b}: restored {n_q} DQ inputs ({n_q - n_ho} beyond forced), MAC {100 * cost:.2f}%", flush=True)
    json.dump(man, open(man_path, "w"), indent=1)


if __name__ == "__main__":
    main()

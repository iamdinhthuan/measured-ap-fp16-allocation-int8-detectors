"""Evaluation queue for RT-DETR-l (rules_11_12d.md section 6). Each task: (plan, key, prefix). A task whose restored
regex was already evaluated (same string, as fx_boot.py matches) is skipped; finished tags are skipped (resume after
reboot). Every evaluated split must carry arch == "rtdetr". Also evaluates the FP32 and whole-INT8 engines on val2017.
Usage: python rd_queue.py STAGE   (STAGE: engine | proxies)"""
import glob
import json
import os
import subprocess
import sys

import rd_common
from block_sensitivity import latency_ms  # noqa: F401  (import check only)

ROOT = rd_common.ROOT
os.chdir(ROOT)
P = sys.executable
M, BASE = "rtdetr-l", "int8ent"
D = os.path.join("metrics", "alloc_eval", f"{M}_{BASE}")
BUD = ("0.03", "0.05", "0.1", "0.15", "0.2")
VAL = ("data/coco_src/annotations/instances_val2017.json", "data/coco/images/val2017")


def sh(cmd):
    print("$", cmd, flush=True)
    if subprocess.call(cmd, shell=True):
        raise SystemExit(f"FAILED: {cmd}")


def plain(tag, onnx_path):
    out = os.path.join(D, f"{tag}_val.json")
    if os.path.exists(out):
        return
    os.makedirs(D, exist_ok=True)
    eng = os.path.join("exports", "tmp", f"{M}_{tag}_val.plan")
    sh(f"{P} tools/build_engine.py --onnx {onnx_path} --engine {eng}")
    sh(f"{P} infer_eval_rd.py --engine {eng} --gt {VAL[0]} --images {VAL[1]} --out {out} --save-pred")
    os.remove(eng)
    assert json.load(open(out))["arch"] == rd_common.ARCH


def evaluated():
    known = {}
    for f in glob.glob(os.path.join(D, "*.json")):
        if f.endswith(("_val.json", "_tune.json")):
            continue
        r = json.load(open(f))
        for sp, v in r.get("splits", {}).items():
            assert v.get("arch") == rd_common.ARCH, (f, sp)
        known.setdefault(r["regex"], os.path.basename(f))
    return known


def regex(plan, key, blocks):
    sel = plan["allocations"][key]
    rx = sel.get("regex") or "|".join(f"(?:{blocks[b]})" for b in sel["blocks"])
    if plan.get("force_regex"):
        rx = f"(?:{plan['force_regex']})" + (f"|{rx}" if rx else "")
    return rx


def task(cfg, key, prefix=""):
    if not os.path.exists(cfg):
        print(f"skip {cfg} (missing)", flush=True)
        return
    plan = json.load(open(cfg))
    if key not in plan["allocations"]:
        print(f"skip {key} (not in {cfg})", flush=True)
        return
    blocks = json.load(open(os.path.join("metrics", f"sens_{M}_{BASE}.json")))["blocks"]
    tag = prefix + key.replace("@", "_B")
    if os.path.exists(os.path.join(D, f"{tag}.json")):
        return
    rx, known = regex(plan, key, blocks), evaluated()
    if rx in known:
        print(f"{cfg}:{key} = {known[rx]} (shared)", flush=True)
        return
    sh(f"{P} rd_wrap.py eval_alloc.py --alloc {cfg} --keys {key} --prefix '{prefix}'")
    r = json.load(open(os.path.join(D, f"{tag}.json")))
    assert all(v["arch"] == rd_common.ARCH for v in r["splits"].values())


def main():
    stage = sys.argv[1]
    if stage == "engine":
        fcfg = os.path.join("configs", f"manual_{M}_{BASE}.json")
        if not os.path.exists(fcfg):
            json.dump({"model": M, "base": BASE, "split": "tune", "force_regex": rd_common.FORCE_RX,
                       "allocations": {"F@0": {"blocks": [], "cost": None}}}, open(fcfg, "w"), indent=1)
        plain("fp32", os.path.join("exports", f"{M}.onnx"))
        plain("int8", os.path.join("exports", f"{M}_{BASE}.onnx"))
        task(fcfg, "F@0")
        alloc = os.path.join("metrics", f"alloc_{M}_{BASE}_ho.json")
        for B in BUD:
            task(alloc, f"mean@{B}")
        for B in BUD:
            task(os.path.join("configs", f"greedy_{M}_{BASE}_ho.json"), f"greedy@{B}")
        for B in ("0.05", "0.1", "0.2"):
            for s in range(1, 6):
                task(os.path.join("metrics", f"random_{M}_{BASE}_ho.json"), f"random@{B}_s{s}")
    elif stage == "proxies":
        for B in BUD:
            task(os.path.join("configs", f"fx_{M}_{BASE}_ho_noise_n2000.json"), f"sqnrq@{B}", "fx2000_")
        for B in BUD:
            task(os.path.join("configs", f"fx_{M}_{BASE}_ho_pandey_n256.json"), f"pandey@{B}", "fxp256_")
        for B in BUD:
            task(os.path.join("configs", f"fx_{M}_{BASE}_ho_hyb.json"), f"hyb@{B}", "hyb_")
    print("QUEUE_DONE", stage, flush=True)


if __name__ == "__main__":
    main()

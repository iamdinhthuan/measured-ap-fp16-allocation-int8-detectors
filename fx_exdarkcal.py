"""Y4 (README 11.10, rules_11_10.md): YOLO11-s INT8 calibrated on ExDark train (exploratory, post-freeze).

Same recipe as the published int8ent graph (gate_headroom.py: ModelOpt ONNX entropy calibration, 512 images,
Sigmoid excluded, CPU calibration) except that the 512 calibration images are drawn (seed 20261103) from the
official ExDark TRAIN split (disjoint from the test split by construction). Two engines: whole-graph INT8 and
INT8 + forced set F (same regex as the published 'headout'). Evaluated on ExDark test and COCO val2017, predictions
saved for the paired bootstrap against the COCO-calibrated engines (exdark_base, exdark_headout).
Outputs (never overwritten): exports/yolo11s_int8ent_exdarkcal.onnx, metrics/exdarkcal/<tag>_<split>.json(.pred.json.gz)
Usage: python fx_exdarkcal.py
"""
import json
import os
import random
import subprocess
import sys

import numpy as np
import onnx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from coco_common import preprocess  # noqa: E402
from qdq_restore import restore  # noqa: E402

PY = sys.executable
F_RX = r"/model\.\d+/(cv2\.\d/cv2\.\d\.2|cv3\.\d/cv3\.\d\.2|dfl)/"
EX_IMG = os.path.join(ROOT, "data", "exdark_raw", "ExDark")
SPLITS = {"exdark_test": (os.path.join(ROOT, "data", "exdark", "exdark_test.json"), EX_IMG),
          "val": (os.path.join(ROOT, "data", "coco_src", "annotations", "instances_val2017.json"),
                  os.path.join(ROOT, "data", "coco", "images", "val2017"))}


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        subprocess.run(cmd, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT)


def main():
    os.chdir(ROOT)
    src = os.path.join("exports", "yolo11s.onnx")
    q_path = os.path.join("exports", "yolo11s_int8ent_exdarkcal.onnx")
    outdir = os.path.join("metrics", "exdarkcal")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join("exports", "tmp"), exist_ok=True)
    log = os.path.join("logs", "exdarkcal.log")
    train = json.load(open(os.path.join("data", "exdark", "exdark_train.json")))["images"]
    test_ids = {im["file_name"] for im in json.load(open(SPLITS["exdark_test"][0]))["images"]}
    names = sorted(im["file_name"] for im in train)
    assert not (set(names) & test_ids)
    random.Random(20261103).shuffle(names)
    calib_files = names[:512]
    json.dump({"seed": 20261103, "files": calib_files}, open(os.path.join(outdir, "calib_files.json"), "w"), indent=0)
    if not os.path.exists(q_path):
        from modelopt.onnx.quantization import quantize as mq
        inp = onnx.load(src, load_external_data=False).graph.input[0].name
        calib = np.stack([preprocess(os.path.join(EX_IMG, f), 640)[0][0] for f in calib_files]).astype(np.float32)
        mq(src, quantize_mode="int8", calibration_data={inp: calib}, calibration_method="entropy",
           calibration_eps=["cpu"], output_path=q_path, op_types_to_exclude=["Sigmoid"])
        print("QUANTIZED", q_path, flush=True)
    for tag, rx in (("exdarkcal_base", None), ("exdarkcal_headout", F_RX)):
        if os.path.exists(os.path.join(outdir, f"{tag}_val.json")):
            continue
        m = onnx.load(q_path)
        if rx:
            restore(m, rx)
        tmp = os.path.join("exports", "tmp", f"yolo11s_{tag}.onnx")
        onnx.save(m, tmp)
        eng = tmp.replace(".onnx", ".plan")
        run(f"{PY} tools/build_engine.py --onnx {tmp} --engine {eng}", log)
        for sp, (gt, img) in SPLITS.items():
            out = os.path.join(outdir, f"{tag}_{sp}.json")
            run(f"{PY} infer_eval.py --engine {eng} --gt {gt} --images {img} --out {out} --save-pred", log)
            st = json.load(open(out))["stats"]
            print(tag, sp, {k: round(100 * st[k], 2) for k in ("AP", "AP_small", "AP_large")}, flush=True)
        os.remove(eng)
        os.remove(tmp)
    print("EXDARKCAL_DONE", flush=True)


if __name__ == "__main__":
    main()

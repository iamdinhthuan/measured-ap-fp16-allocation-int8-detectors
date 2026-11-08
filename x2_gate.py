"""X2 (rules_20261108_x2.md, written before any result): regenerate exports/<m>.onnx and exports/<m>_int8ent.onnx with the
published settings, then gate G0 (block partition; yolo11l INT8 sha256 compared descriptively). Output metrics/x2/gate_<m>.json (never
overwritten). Usage: python x2_gate.py <model>"""
import glob
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
M = sys.argv[1]
os.makedirs(os.path.join(ROOT, "metrics", "x2"), exist_ok=True)
OUT = os.path.join(ROOT, "metrics", "x2", f"gate_{M}.json")
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = {"rules": "rules_20261108_x2.md", "model": M, "pt_sha256": sha(os.path.join(ROOT, "exports", f"{M}.pt"))}
src = os.path.join(ROOT, "exports", f"{M}.onnx")
if not os.path.exists(src):
    import torch
    import ultralytics
    from ultralytics import YOLO
    tmpd = os.path.join(ROOT, "exports", f"x2_export_{M}")
    os.makedirs(tmpd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "exports", f"{M}.pt"), tmpd)
    f = YOLO(os.path.join(tmpd, f"{M}.pt")).export(format="onnx", opset=17, simplify=True, batch=1, dynamic=False,
                                                   nms=False, imgsz=640)
    shutil.move(f, src)
    shutil.rmtree(tmpd)
    rec.update(torch=torch.__version__, ultralytics=ultralytics.__version__)
rec["onnx_sha256"] = sha(src)
q = os.path.join(ROOT, "exports", f"{M}_int8ent.onnx")
if not os.path.exists(q):
    import numpy as np
    import onnx
    from modelopt.onnx.quantization import quantize as mq
    from coco_common import COCO_CALIB_DIR, preprocess
    files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, "*.jpg")))
    assert len(files) == 512, len(files)
    calib = np.stack([preprocess(f, 640)[0][0] for f in files]).astype(np.float32)
    inp = onnx.load(src, load_external_data=False).graph.input[0].name
    mq(src, quantize_mode="int8", calibration_data={inp: calib}, calibration_method="entropy",
       calibration_eps=["cpu"], output_path=q, op_types_to_exclude=["Sigmoid"], nodes_to_exclude=None)
rec["int8ent_sha256"] = sha(q)
import onnx  # noqa: E402
from block_sensitivity import blocks  # noqa: E402
pub = json.load(open(os.path.join(ROOT, "metrics", f"sens_{M}_int8ent.json")))
rec["G0_blocks"] = blocks(onnx.load(q)) == pub["blocks"]
ok = rec["G0_blocks"]
if M == "yolo11l":
    rec["sha_matches_m8l_regeneration"] = rec["int8ent_sha256"] == "630e5cb823459ff122a32aee983c1f3411608311f52c3aea4e6ffc02059a7525"
rec["G0_pass"] = bool(ok)
json.dump(rec, open(OUT, "w"), indent=1)
print(M, "int8ent", rec["int8ent_sha256"], "G0_PASS" if ok else "G0_FAIL", flush=True)

"""M8L (rules_20261108_m8l.md, written before any result): reconstruct yolo11l.onnx and yolo11l_int8ent.onnx, then the
reproduction gate G1/G2 on tune2000 under NEW tags g8l_* (published metrics/tune files are only read).
Output (never overwritten): metrics/m8l_gate.json. Exit 3 on gate failure."""
import glob
import hashlib
import json
import os
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
OUT = os.path.join(ROOT, "metrics", "m8l_gate.json")
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists")
M = "yolo11l"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = {"rules": "rules_20261108_m8l.md"}
src = os.path.join(ROOT, "exports", f"{M}.onnx")
assert sha(os.path.join(ROOT, "exports", f"{M}.pt")) == "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"
if not os.path.exists(src):
    import torch
    import ultralytics
    from ultralytics import YOLO
    tmpd = os.path.join(ROOT, "exports", "m8l_export")
    os.makedirs(tmpd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "exports", f"{M}.pt"), tmpd)
    f = YOLO(os.path.join(tmpd, f"{M}.pt")).export(format="onnx", opset=17, simplify=True, batch=1, dynamic=False,
                                                   nms=False, imgsz=640)
    shutil.move(f, src)
    shutil.rmtree(tmpd)
    rec.update(torch=torch.__version__, ultralytics=ultralytics.__version__)
rec["onnx_sha256"] = sha(src)
rec["onnx_sha_matches_published"] = rec["onnx_sha256"] == "64fc754792a837d7ea4314134321079727d4bb1b6c9de4b5524f8f4bff42cf4a"
print("onnx", rec["onnx_sha256"], "match" if rec["onnx_sha_matches_published"] else "MISMATCH", flush=True)
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
print("int8ent", rec["int8ent_sha256"], flush=True)
import onnx  # noqa: E402
from block_sensitivity import blocks, evaluate_variant  # noqa: E402
from qdq_restore import restore  # noqa: E402
pub_s = json.load(open(os.path.join(ROOT, "metrics", f"sens_{M}_int8ent.json")))
bl = blocks(onnx.load(q))
assert bl == pub_s["blocks"], "block partition differs"
a = types.SimpleNamespace(model=M, split="tune", base="int8ent")
log = os.path.join(ROOT, "logs", "m8l_gate.log")
tmp = os.path.join(ROOT, "exports", "tmp")
os.makedirs(tmp, exist_ok=True)
pubdir = os.path.join(ROOT, "metrics", "tune", M)
checks = [("g8l_int8ent", None, "int8ent.json"), ("g8l_restore_head.cv3.0", "head.cv3.0", "int8ent_restore_head.cv3.0.json"),
          ("g8l_restore_m9", "m9", "int8ent_restore_m9.json")]
rec["checks"] = {}
ok = True
for tag, b, pubf in checks:
    assert not os.path.exists(os.path.join(pubdir, f"{tag}.json")), f"{tag} already evaluated"
    path = q
    if b:
        mm = onnx.load(q)
        restore(mm, bl[b])
        path = os.path.join(tmp, f"{M}_{tag}.onnx")
        onnx.save(mm, path)
    r = evaluate_variant(tag, path, a, log)
    assert "build_s" in r and r["build_s"] > 0, "no engine was built"
    if b:
        os.remove(path)
    p = json.load(open(os.path.join(pubdir, pubf)))["stats"]["AP"]
    d = 100 * (r["stats"]["AP"] - p)
    rec["checks"][tag] = {"AP": 100 * r["stats"]["AP"], "published": 100 * p, "diff": d, "build_s": r["build_s"]}
    ok &= abs(d) <= 0.07
    print(f"{tag:26s} AP={100*r['stats']['AP']:.3f} published={100*p:.3f} diff={d:+.3f}", flush=True)
rec["gate_pass"] = bool(ok)
json.dump(rec, open(OUT, "w"), indent=1)
print("GATE_PASS" if ok else "GATE_FAIL", flush=True)
sys.exit(0 if ok else 3)

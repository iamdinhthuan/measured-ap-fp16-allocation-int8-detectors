"""Gate of rules_11_12d.md section 4 (tuning split only): G0 FP32 reproduction (Ultralytics predictor, TensorRT
engine, FP32-lifted onnxruntime reference), G1 FP16, G2 headroom over INT8 + F. Rows share tags with the sweep."""
import argparse
import copy
import json
import os
import sys

import rd_common  # noqa: F401  (must come first)
import block_sensitivity as bs
import onnx
from coco_common import COCO80_TO_91
from infer_eval import evaluate
from qdq_restore import restore
from sim_sensitivity import predict as ort_predict, to_fp32

ROOT = rd_common.ROOT
assert bs.blocks is rd_common.blocks_rd and bs.run is rd_common.run_rd
M, BASE = "rtdetr-l", "int8ent"
out = os.path.join(ROOT, "metrics", f"gate_{M}.json")
res = json.load(open(out)) if os.path.exists(out) else {}
a = argparse.Namespace(model=M, base=BASE, split="tune")
log = os.path.join(ROOT, "logs", f"sens_{M}_{BASE}.log")
gt, img_dir = bs.SPLITS["tune"][0], bs.SPLITS["tune"][1]
items = sorted((im["id"], os.path.join(img_dir, im["file_name"])) for im in json.load(open(gt))["images"])
tmp = os.path.join(ROOT, "exports", "tmp")
os.makedirs(tmp, exist_ok=True)
base_onnx = os.path.join(ROOT, "exports", f"{M}_{BASE}.onnx")


def save():
    json.dump(res, open(out, "w"), indent=1)


def ap(r):
    assert r.get("arch") == rd_common.ARCH, r.get("engine")
    return 100 * r["stats"]["AP"]


if "ultralytics" not in res:
    from ultralytics import RTDETR
    model = RTDETR(os.path.join(ROOT, "exports", f"{M}.pt"))
    preds = []
    for img_id, p in items:
        r = model.predict(p, conf=0.001, max_det=300, imgsz=640, half=False, device=0, verbose=False)[0]
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            preds.append({"image_id": int(img_id), "category_id": COCO80_TO_91[int(c)],
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    st, n = evaluate(gt, preds)
    res["ultralytics"] = {"stats": st, "n_images": n, "n_det": len(preds)}
    save()
rows = {"fp32": bs.evaluate_variant("fp32", os.path.join(ROOT, "exports", f"{M}.onnx"), a, log),
        "int8ent": bs.evaluate_variant(BASE, base_onnx, a, log)}
allf = os.path.join(tmp, f"{M}_{BASE}_restore_all.onnx")
if not os.path.exists(allf):
    mm = onnx.load(base_onnx)
    restore(mm, ".")
    onnx.save(mm, allf)
rows["fp16_all"] = bs.evaluate_variant(f"{BASE}_restore_all", allf, a, log)
ff = os.path.join(tmp, f"{M}_{BASE}_restore_F.onnx")
if not os.path.exists(ff):
    mm = onnx.load(base_onnx)
    restore(mm, rd_common.FORCE_RX)
    onnx.save(mm, ff)
rows["int8_F"] = bs.evaluate_variant(f"{BASE}_restore_F", ff, a, log)
if "ort_ref" not in res:
    m = copy.deepcopy(onnx.load(base_onnx))
    restore(m, ".*")
    st, n = evaluate(gt, ort_predict(to_fp32(m), items))
    res["ort_ref"] = {"stats": st, "n_images": n}
    save()
v = {"ultra": 100 * res["ultralytics"]["stats"]["AP"], "ort_ref": 100 * res["ort_ref"]["stats"]["AP"],
     **{k: ap(r) for k, r in rows.items()}}
g0 = abs(v["fp32"] - v["ultra"]) <= 0.3 and abs(v["ort_ref"] - v["ultra"]) <= 0.3
g1 = v["fp32"] - v["fp16_all"] <= 1.0
g2 = v["fp16_all"] - v["int8_F"] >= 1.0
res.update({"ap": v, "G0": g0, "G1": g1, "G2": g2, "rules": "rules_11_12d.md"})
save()
print(json.dumps({k: round(x, 2) for k, x in v.items()}), f"G0={g0} G1={g1} G2={g2}",
      "GATE_PASS" if (g0 and g1 and g2) else "GATE_STOP", flush=True)

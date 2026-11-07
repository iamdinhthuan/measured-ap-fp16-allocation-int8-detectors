"""rules_11_12d_add1.md A1: where does the INT8 collapse of RT-DETR-l happen? onnxruntime CPU EP, first 200 tuning
images: unrestored INT8 graph vs all-restored graph (both lifted to FP32); census of op types with DQ-fed inputs."""
import copy
import json
import os
from collections import Counter

import rd_common
import onnx
import onnxruntime as ort
from coco_common import COCO80_TO_91, CONF_FLOOR
from infer_eval import evaluate
from qdq_restore import restore
from sim_sensitivity import OPT_LEVEL, to_fp32

ROOT = rd_common.ROOT
os.chdir(ROOT)
out = os.path.join("metrics", "diag_rtdetr-l_int8ent_ortcpu.json")
assert not os.path.exists(out), out
q = onnx.load(os.path.join("exports", "rtdetr-l_int8ent.onnx"))
prod = {o: n for n in q.graph.node for o in n.output}
census, names = Counter(), {}
for n in q.graph.node:
    if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
        continue
    k = sum(1 for i in n.input if i in prod and prod[i].op_type == "DequantizeLinear")
    if k:
        census[n.op_type] += 1
        names.setdefault(n.op_type, []).append(n.name)
full = json.load(open("data/tune2000/instances_tune.json"))
ims = sorted(full["images"], key=lambda im: im["file_name"])[:200]
keep = {im["id"] for im in ims}
gt = os.path.join("data", "tune2000", "instances_tune_first200.json")
if not os.path.exists(gt):
    json.dump(dict(full, images=ims, annotations=[a for a in full["annotations"] if a["image_id"] in keep]), open(gt, "w"))
items = sorted((im["id"], os.path.join("data", "tune2000", "images", im["file_name"])) for im in ims)


def run(m):
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVEL
    so.intra_op_num_threads = 6
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    preds = []
    for img_id, p in items:
        x, gain, padx, pady, w0, h0 = rd_common.preprocess_stretch(p, 640)
        buf = s.run(None, {s.get_inputs()[0].name: x})[0]
        for x1, y1, x2, y2, sc, c in rd_common.decode_rd(buf, CONF_FLOOR, gain, padx, pady, w0, h0):
            preds.append({"image_id": int(img_id), "category_id": COCO80_TO_91[c],
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)})
    return evaluate(gt, preds)[0]["AP"] * 100


res = {"n_img": len(items), "executor": f"onnxruntime {ort.__version__} CPU EP, {OPT_LEVEL.name}, lifted to FP32",
       "dq_fed_op_census": dict(census), "dq_fed_nodes": names}
res["int8_unrestored_AP"] = run(to_fp32(copy.deepcopy(q)))
r = copy.deepcopy(q)
restore(r, ".*")
res["all_restored_AP"] = run(to_fp32(r))
f = copy.deepcopy(q)
restore(f, rd_common.FORCE_RX)
res["int8_F_AP"] = run(to_fp32(f))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "dq_fed_nodes"}), flush=True)

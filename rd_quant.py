"""INT8 Q/DQ graph of RT-DETR-l with the recipe of gate_headroom.py variant int8ent, unchanged (ModelOpt entropy,
the 512 COCO calibration images, CPU calibration, Sigmoid excluded); only the preprocessing is RT-DETR's stretch."""
import glob
import os
import time

import numpy as np
import onnx

import rd_common
from coco_common import COCO_CALIB_DIR

ROOT = rd_common.ROOT
src = os.path.join(ROOT, "exports", "rtdetr-l.onnx")
dst = os.path.join(ROOT, "exports", "rtdetr-l_int8ent.onnx")
assert not os.path.exists(dst), dst
from modelopt.onnx.quantization import quantize as mq  # noqa: E402
inp = onnx.load(src, load_external_data=False).graph.input[0].name
files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, "*.jpg")))
assert len(files) == 512, len(files)
calib = np.stack([rd_common.preprocess_stretch(f, 640)[0][0] for f in files]).astype(np.float32)
t0 = time.time()
mq(src, quantize_mode="int8", calibration_data={inp: calib}, calibration_method="entropy",
   calibration_eps=["cpu"], output_path=dst, op_types_to_exclude=["Sigmoid"])
print("QUANT_DONE", dst, f"{time.time() - t0:.0f}s", flush=True)

"""rules_11_12d_add1.md A2: sweep with base INT8 + F (exports/rtdetr-l_int8entF.onnx = INT8 graph, F restored)."""
import os
import sys

import onnx

import rd_common  # noqa: F401  (must come first)
import block_sensitivity as bs
from qdq_restore import restore

assert bs.blocks is rd_common.blocks_rd and bs.run is rd_common.run_rd
src = os.path.join(rd_common.ROOT, "exports", "rtdetr-l_int8ent.onnx")
dst = os.path.join(rd_common.ROOT, "exports", "rtdetr-l_int8entF.onnx")
if not os.path.exists(dst):
    m = onnx.load(src)
    restore(m, rd_common.FORCE_RX)
    onnx.save(m, dst)
sys.argv = [sys.argv[0], "--model", "rtdetr-l", "--base", "int8entF"]
bs.main()

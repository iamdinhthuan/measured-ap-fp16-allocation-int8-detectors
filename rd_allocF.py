"""rules_11_12d_add1.md A2: allocate.py on the INT8 + F sweep (same MILP, cost, budgets; F cost taken from B)."""
import os
import sys

import rd_common
import allocate

assert allocate.conv_macs is rd_common.lin_macs
out = os.path.join(rd_common.ROOT, "metrics", "alloc_rtdetr-l_int8entF_ho.json")
assert not os.path.exists(out), out
os.chdir(rd_common.ROOT)
sys.argv = [sys.argv[0], "--sens", "metrics/sens_rtdetr-l_int8entF.json", "--onnx", "exports/rtdetr-l.onnx",
            "--budgets", "0.03,0.05,0.1,0.15,0.2", "--force-regex", rd_common.FORCE_RX, "--out", out]
allocate.main()

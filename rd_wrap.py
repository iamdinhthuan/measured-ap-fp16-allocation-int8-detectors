"""Run a published pipeline script on RT-DETR-l through the adapters (rules_11_12d.md).
Usage: python rd_wrap.py fx_factorial.py --model rtdetr-l --mode noise ...
The script runs as __main__ after rd_common has rebound coco_common/allocate/block_sensitivity, so its
'from X import name' statements bind the adapted functions. Scripts that DEFINE one of the adapted functions
(allocate.py, block_sensitivity.py) would shadow the adapters and are refused; use rd_alloc.py / rd_sens.py."""
import re
import runpy
import sys

import rd_common  # noqa: F401  (must come first)

script = sys.argv[1]
src = open(script).read()
bad = re.findall(r"^def (conv_macs|preprocess|decode_output|blocks|run)\(", src, re.M)
if bad:
    raise SystemExit(f"{script} defines {bad}: it would bypass the RT-DETR adapters")
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")

"""infer_eval.py for RT-DETR-l (rules_11_12d.md): stretch preprocessing and decode_rtdetr via rd_common; adds arch."""
import json
import sys

import rd_common  # noqa: F401  (patches coco_common before infer_eval binds its names)
import infer_eval

assert infer_eval.decode_output is rd_common.decode_rd and infer_eval.preprocess is rd_common.preprocess_stretch
infer_eval.main()
out = sys.argv[sys.argv.index("--out") + 1]
res = json.load(open(out))
res["arch"] = rd_common.ARCH
json.dump(res, open(out, "w"), indent=1)

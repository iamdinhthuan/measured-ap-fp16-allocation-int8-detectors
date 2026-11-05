"""Phase B pilot (README 11.7): boundary features of the yolo26l single-block, held-out and random graphs."""
import glob, json, os
from boundary_features import features

dst = "metrics/phaseB_pilot_features_yolo26l.json"
if os.path.exists(dst):
    raise SystemExit(f"{dst} exists")
out = {}
for d in ("yolo26l", "yolo26l_sb"):
    for p in sorted(glob.glob(f"exports/orin/{d}/*.onnx")):
        out[os.path.basename(p)[:-5]] = features(p)
        print(d, os.path.basename(p), out[os.path.basename(p)[:-5]], flush=True)
json.dump(out, open(dst, "w"), indent=1)
print("FEAT_DONE")

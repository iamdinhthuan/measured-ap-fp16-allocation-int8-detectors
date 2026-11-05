"""Post-freeze E4 (README 11.6): score Orin val2017 predictions with the same evaluate() as the desktop engines."""
import glob, gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_eval import evaluate

GT = "data/coco_src/annotations/instances_val2017.json"
out = "metrics/orin_e4/orin_e4_scores.json"
if os.path.exists(out):
    raise SystemExit(f"{out} exists; refusing to overwrite")
res = {}
for f in sorted(glob.glob("metrics/orin_e4/yolo26l_*.pred.json.gz")):
    name = os.path.basename(f)[len("yolo26l_"):-len(".pred.json.gz")]
    preds = json.load(gzip.open(f))
    stats, n = evaluate(GT, preds)
    res[name] = {"n_img": n, "n_det": len(preds), **{k: round(100 * v, 2) for k, v in stats.items()}}
    print(name, res[name], flush=True)
json.dump(res, open(out, "w"), indent=1)
print("ORIN_SCORE_DONE", out)

"""X3 (README 11.9, rules_11_9.md): two more 2000-image train2017 tuning draws.

Pool = train images with >= 1 non-crowd annotation, minus the 512 calibration images and the tune2000 images.
One shuffle with seed 20261102: first 2000 -> draw 1, next 2000 -> draw 2 (disjoint).
Usage: python make_tuning_draws.py --train-ann instances_train2017.json --calib-list calib.txt --tune-list tune.txt --out DIR
"""
import argparse, hashlib, json, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--calib-list", required=True)
    ap.add_argument("--tune-list", required=True)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = open(a.train_ann, "rb").read()
    gt = json.loads(raw)
    ids = lambda p: {int(os.path.splitext(l.strip())[0]) for l in open(p) if l.strip()}
    calib, tune = ids(a.calib_list), ids(a.tune_list)
    assert len(calib) == 512 and len(tune) == 2000
    has_ann = {x["image_id"] for x in gt["annotations"] if not x.get("iscrowd", 0)}
    pool = sorted(i["id"] for i in gt["images"] if i["id"] in has_ann and i["id"] not in calib and i["id"] not in tune)
    order = pool[:]
    random.Random(a.seed).shuffle(order)
    for k in (1, 2):
        keep = set(order[(k - 1) * a.n: k * a.n])
        assert not keep & (calib | tune)
        sub = {"info": gt.get("info", {}), "licenses": gt.get("licenses", []), "categories": gt["categories"],
               "images": [i for i in gt["images"] if i["id"] in keep],
               "annotations": [x for x in gt["annotations"] if x["image_id"] in keep]}
        d = os.path.join(a.out, f"tune_d{k}")
        os.makedirs(d, exist_ok=True)
        json.dump(sub, open(os.path.join(d, "instances_tune.json"), "w"))
        json.dump({"seed": a.seed, "draw": k, "n": a.n, "source_ann_sha256": hashlib.sha256(raw).hexdigest(),
                   "excluded_calib": len(calib), "excluded_tune2000": len(tune), "pool": len(pool),
                   "files": [i["file_name"] for i in sub["images"]]}, open(os.path.join(d, "tune_manifest.json"), "w"), indent=0)
        open(os.path.join(d, "urls.txt"), "w").write("".join(i["coco_url"] + "\n" for i in sub["images"]))
        print(d, len(sub["images"]), "images", len(sub["annotations"]), "annotations")


if __name__ == "__main__":
    main()

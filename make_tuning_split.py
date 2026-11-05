"""Draw a labelled train2017 tuning split, disjoint from the 512 calibration images.

The tuning split is where block sensitivities are measured and the allocation is
chosen; val2017 is held out for the final comparison only. Uniform random draw
over train images with at least one non-crowd annotation, fixed seed.

Usage: python make_tuning_split.py --train-ann .../instances_train2017.json \
           --calib-dir .../calib/images --n 2000 --seed 20261025 --out tune2000
"""
import argparse
import hashlib
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = open(a.train_ann, "rb").read()
    gt = json.loads(raw)
    calib = {int(os.path.splitext(f)[0]) for f in os.listdir(a.calib_dir) if f.endswith(".jpg")}
    has_ann = {x["image_id"] for x in gt["annotations"] if not x.get("iscrowd", 0)}
    pool = sorted(i["id"] for i in gt["images"] if i["id"] in has_ann and i["id"] not in calib)
    ids = sorted(random.Random(a.seed).sample(pool, a.n))
    keep = set(ids)
    assert not keep & calib
    sub = {"info": gt.get("info", {}), "licenses": gt.get("licenses", []), "categories": gt["categories"],
           "images": [i for i in gt["images"] if i["id"] in keep],
           "annotations": [x for x in gt["annotations"] if x["image_id"] in keep]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(sub, open(os.path.join(a.out, "instances_tune.json"), "w"))
    json.dump({"seed": a.seed, "n": a.n, "source_ann_sha256": hashlib.sha256(raw).hexdigest(),
               "calib_ids_excluded": len(calib), "pool": len(pool),
               "files": [i["file_name"] for i in sub["images"]],
               "coco_urls": [i["coco_url"] for i in sub["images"]]},
              open(os.path.join(a.out, "tune_manifest.json"), "w"), indent=0)
    area = [x["area"] for x in sub["annotations"] if not x.get("iscrowd", 0)]
    s = sum(a_ < 32 ** 2 for a_ in area); l_ = sum(a_ > 96 ** 2 for a_ in area)
    print(f"{len(ids)} images, {len(area)} objects: small={s} medium={len(area)-s-l_} large={l_}")


if __name__ == "__main__":
    main()

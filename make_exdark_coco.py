"""ExDark (low-light, 12 classes) -> COCO-format GT, one json per official split.

All 12 ExDark classes exist in COCO, so COCO-pretrained detectors are evaluated
zero-shot: the GT category list holds only the 12 mapped COCO ids, hence
COCOeval ignores predictions of the other 68 classes. Box area (w*h) stands in
for mask area, so S/M/L thresholds are applied to boxes (stated in the paper).
Each image keeps its ExDark lighting type (1..10), indoor/outdoor flag and split
(1 train, 2 val, 3 test) as extra fields.

Usage: python make_exdark_coco.py --raw data/exdark_raw --out data/exdark
"""
import argparse
import hashlib
import json
import os

from PIL import Image

# ExDark class -> COCO 91-index category id
EXDARK_TO_COCO = {"Bicycle": 2, "Boat": 9, "Bottle": 44, "Bus": 6, "Car": 3, "Cat": 17, "Chair": 62,
                  "Cup": 47, "Dog": 18, "Motorbike": 4, "People": 1, "Table": 67}
COCO_NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 9: "boat", 17: "cat",
              18: "dog", 44: "bottle", 47: "cup", 62: "chair", 67: "dining table"}
CLASS_BY_LABEL = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike",
                  "People", "Table"]
LIGHT = ["Low", "Ambient", "Object", "Single", "Weak", "Strong", "Screen", "Window", "Shadow", "Twilight"]
SPLITS = {1: "train", 2: "val", 3: "test"}


def find(root, folder, name):
    """Case-insensitive lookup of root/folder/name (ExDark mixes extension case)."""
    d = os.path.join(root, folder)
    low = {f.lower(): f for f in os.listdir(d)}
    f = low.get(name.lower())
    return os.path.join(d, f) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir with imageclasslist.txt, ExDark/ (images) and ExDark_Annno/")
    ap.add_argument("--images", default="ExDark")
    ap.add_argument("--annos", default="ExDark_Annno")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [l.split() for l in open(os.path.join(a.raw, "imageclasslist.txt")).read().splitlines()[1:] if l.strip()]
    cats = [{"id": c, "name": COCO_NAMES[c]} for c in sorted(COCO_NAMES)]
    out = {s: {"images": [], "annotations": [], "categories": cats} for s in SPLITS.values()}
    skipped, ann_id = [], 1
    for img_id, (name, label, light, inout, split) in enumerate(rows, 1):
        folder = CLASS_BY_LABEL[int(label) - 1]
        img = find(os.path.join(a.raw, a.images), folder, name)
        txt = find(os.path.join(a.raw, a.annos), folder, name + ".txt")
        if img is None or txt is None:
            skipped.append(name)
            continue
        try:
            with Image.open(img) as im:
                w, h = im.size
        except OSError:
            skipped.append(name)
            continue
        s = SPLITS[int(split)]
        out[s]["images"].append({"id": img_id, "file_name": os.path.relpath(img, os.path.join(a.raw, a.images)),
                                 "width": w, "height": h, "light": LIGHT[int(light) - 1],
                                 "indoor": int(inout) == 1, "exdark_class": folder})
        for line in open(txt).read().splitlines()[1:]:
            p = line.split()
            if len(p) < 5 or p[0] not in EXDARK_TO_COCO:
                continue
            l, t, bw, bh = map(float, p[1:5])
            # clip to the image; drop degenerate boxes
            l, t = max(0.0, l), max(0.0, t)
            bw, bh = min(bw, w - l), min(bh, h - t)
            if bw <= 1 or bh <= 1:
                continue
            out[s]["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": EXDARK_TO_COCO[p[0]],
                                          "bbox": [l, t, bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1
    manifest = {"source_split_file_sha256": hashlib.sha256(open(os.path.join(a.raw, "imageclasslist.txt"), "rb").read()).hexdigest(),
                "skipped": skipped, "splits": {}}
    for s, d in out.items():
        path = os.path.join(a.out, f"exdark_{s}.json")
        json.dump(d, open(path, "w"))
        areas = [x["area"] for x in d["annotations"]]
        manifest["splits"][s] = {"images": len(d["images"]), "objects": len(areas),
                                 "small": sum(x < 32 ** 2 for x in areas),
                                 "medium": sum(32 ** 2 <= x < 96 ** 2 for x in areas),
                                 "large": sum(x >= 96 ** 2 for x in areas),
                                 "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest["splits"], indent=1), f"skipped={len(skipped)}")


if __name__ == "__main__":
    main()

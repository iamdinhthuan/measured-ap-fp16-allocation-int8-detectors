#!/bin/bash
# M8L data reconstruction (rules_20261108_m8l.md section 0). Resumable. CPU/network only.
set -e
cd /home/thuan/isci_method_20261025
P=/home/thuan/miniconda3/envs/qtsd/bin/python
TC=/home/thuan/topic_c_ivc/data/datasets/coco_pilot_v1_20260923
echo "== $(date +%F_%T) prep start"
mkdir -p data/coco/images data/coco_src/annotations data/coco/calib/images data/tune2000/images
[ -e data/coco/images/val2017 ] || ln -s $TC/images/val2017 data/coco/images/val2017
[ -e data/coco_src/annotations/instances_val2017.json ] || ln -s $TC/annotations/instances_val2017.json data/coco_src/annotations/instances_val2017.json
# calibration images
cut -c1-16 m8l_calib_files.txt | sed 's|^|http://images.cocodataset.org/train2017/|' > /tmp/m8l_calib_urls.txt
cd data/coco/calib/images
xargs -P 8 -n 1 -a /tmp/m8l_calib_urls.txt sh -c 'f=$(basename "$0"); [ -s "$f" ] || curl -sf --retry 3 -o "$f" "$0"'
cd - >/dev/null
n=$(ls data/coco/calib/images/*.jpg | wc -l); echo "calib images: $n"; [ "$n" -eq 512 ]
# tuning split
if [ ! -f data/tune2000/instances_tune.json ]; then
  unzip -p $TC/archives/annotations_trainval2017.zip annotations/instances_train2017.json > data/m8l_train_ann.json
  s=$(sha256sum data/m8l_train_ann.json | cut -d" " -f1); echo "train ann sha256 $s"
  [ "$s" = 610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d ]
  $P make_tuning_split.py --train-ann data/m8l_train_ann.json --calib-dir data/coco/calib/images --n 2000 --seed 20261025 --out data/tune2000
  rm -f data/m8l_train_ann.json
fi
$P - <<'PY'
import json
a=json.load(open("data/tune2000/tune_manifest.json")); b=json.load(open("m8l_tune_manifest_published.json"))
assert a["files"]==b["files"] and a["source_ann_sha256"]==b["source_ann_sha256"], "tune split differs from published"
open("/tmp/m8l_tune_urls.txt","w").write("\n".join(a["coco_urls"])+"\n"); print("tune manifest identical to published")
PY
cd data/tune2000/images
xargs -P 8 -n 1 -a /tmp/m8l_tune_urls.txt sh -c 'f=$(basename "$0"); [ -s "$f" ] || curl -sf --retry 3 -o "$f" "$0"'
cd - >/dev/null
n=$(ls data/tune2000/images/*.jpg | wc -l); echo "tune images: $n"; [ "$n" -eq 2000 ]
echo M8L_PREP_DONE

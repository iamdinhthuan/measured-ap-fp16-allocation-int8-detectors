#!/bin/bash
# X2 data reconstruction (rules_20261108_x2.md section 0): val2017 symlinks + 512 calibration images. CPU/network only.
set -e
cd /home/thuan/isci_method_20261025
TC=/home/thuan/topic_c_ivc/data/datasets/coco_pilot_v1_20260923
echo "== $(date +%F_%T) prep start"
mkdir -p data/coco/images data/coco_src/annotations data/coco/calib/images
[ -e data/coco/images/val2017 ] || ln -s $TC/images/val2017 data/coco/images/val2017
[ -e data/coco_src/annotations/instances_val2017.json ] || ln -s $TC/annotations/instances_val2017.json data/coco_src/annotations/instances_val2017.json
n=$(ls data/coco/images/val2017/ | wc -l); echo "val images: $n"; [ "$n" -eq 5000 ]
cut -c1-16 m8l_calib_files.txt | sed 's|^|http://images.cocodataset.org/train2017/|' > /tmp/x2_calib_urls.txt
cd data/coco/calib/images
xargs -P 8 -n 1 -a /tmp/x2_calib_urls.txt sh -c 'f=$(basename "$0"); [ -s "$f" ] || curl -sf --retry 3 -o "$f" "$0"'
cd - >/dev/null
n=$(ls data/coco/calib/images/*.jpg | wc -l); echo "calib images: $n"; [ "$n" -eq 512 ]
echo X2_PREP_DONE

# waits for coco download, extracts all shards, builds final demo sets
D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
PY=.venv-eval/bin/python
until grep -q ALL_DL_DONE logs/coco_dl.log; do $PY scripts/data/coco_extract.py; sleep 60; done
$PY scripts/data/coco_extract.py
echo "images: $(ls data/coco/train2017 | wc -l)"
$PY scripts/data/build_sets.py 50000 10000
echo PIPELINE_DONE

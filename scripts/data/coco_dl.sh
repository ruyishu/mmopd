D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=$D/.cache/hf TMPDIR=$D/.cache/tmp
mkdir -p data/coco/raw data/coco/train2017
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, HfApi
import time
hf_hub_download("phiyodr/coco2017","data/train-00000-of-00001-0084e041f1902997.parquet",repo_type="dataset",local_dir="data/coco/raw/captions")
print("captions done",flush=True)
fs=sorted(f for f in HfApi().list_repo_files("detection-datasets/coco",repo_type="dataset") if f.startswith("data/train-"))
print("shards",len(fs),flush=True)
for f in fs:
    for t in range(5):
        try:
            hf_hub_download("detection-datasets/coco",f,repo_type="dataset",local_dir="data/coco/raw/det"); print("DL",f,time.strftime("%H:%M:%S"),flush=True); break
        except Exception as e: print("retry",f,e,flush=True); time.sleep(10)
PY
echo ALL_DL_DONE

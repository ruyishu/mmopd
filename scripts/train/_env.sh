# shared env; source from repo root. STU_GPU / TEA_GPU = physical GPU ids.
D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
export HOME=$D/.cache/home TRITON_CACHE_DIR=$D/.cache/triton TMPDIR=$D/.cache/tmp XDG_CACHE_HOME=$D/.cache/xdg \
  HF_HOME=$D/.cache/hf HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=$D/src \
  PATH=$D/.venv-eval/bin:/usr/local/bin:/usr/bin:/bin TOKENIZERS_PARALLELISM=false
STU_GPU=${STU_GPU:-5}; TEA_GPU=${TEA_GPU:-4}
CFG=${CFG:-configs/demo_1p7b.yaml}
mkdir -p logs ckpt
if [ "$STU_GPU" = "$TEA_GPU" ]; then export CUDA_VISIBLE_DEVICES=$STU_GPU; DEVS="cuda:0 cuda:0"; else export CUDA_VISIBLE_DEVICES=$STU_GPU,$TEA_GPU; DEVS="cuda:0 cuda:1"; fi
# write a device-resolved copy of the config
read SDEV TDEV <<< "$DEVS"
RCFG=logs/.cfg_$$.yaml
python - "$CFG" "$SDEV" "$TDEV" "$RCFG" <<'PY'
import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); c["student"]["device"]=sys.argv[2]; c["teacher"]["device"]=sys.argv[3]; yaml.safe_dump(c,open(sys.argv[4],"w"))
PY
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader -i $STU_GPU,$TEA_GPU

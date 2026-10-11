#!/bin/bash
# Base eval: student GSM8K on GPU1, teacher MMBench on GPU4. Arg1 = limit (e.g. 50) or "all"
P=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd
cd $P || exit 1
L=${1:-50}; LIM=$([ "$L" = all ] && echo "" || echo $L)
export HOME=$P/.cache/home TRITON_CACHE_DIR=$P/.cache/triton VLLM_USE_FLASHINFER_SAMPLER=0 TMPDIR=$P/.cache/tmp XDG_CACHE_HOME=$P/.cache/xdg PATH=$P/.venv-eval/bin:/usr/local/bin:/usr/bin:/bin HF_HOME=$P/.cache/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$P/.cache/vllm
PY=$P/.venv-eval/bin/python
mkdir -p outputs/base_eval_$L logs
(CUDA_VISIBLE_DEVICES=${STU_GPU:-1} GPU_UTIL=${GPU_UTIL:-0.85} setsid nohup $PY scripts/eval/eval_gsm8k.py models/Qwen3-1.7B outputs/base_eval_$L/student $LIM > logs/eval_gsm8k_$L.log 2>&1 < /dev/null &)
sleep ${STAGGER:-0}
(CUDA_VISIBLE_DEVICES=${TEA_GPU:-4} GPU_UTIL=${GPU_UTIL:-0.85} setsid nohup $PY scripts/eval/eval_mmbench.py models/Qwen3-VL-4B-Instruct outputs/base_eval_$L/teacher $LIM > logs/eval_mmbench_$L.log 2>&1 < /dev/null &)
sleep 3; echo started $L

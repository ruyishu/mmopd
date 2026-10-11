#!/bin/bash
# usage: STU_GPU=5 bash scripts/train/eval_ckpt.sh ckpt/demo_opd/step100 [N=200]
# GSM8K: merge LoRA -> vLLM, same settings as base eval (scripts/eval/eval_gsm8k.py), N samples.
# MMBench dev: HF generate with student vision path, N questions, max 64 tokens.
CK=$1; N=${2:-200}; TEA_GPU=${STU_GPU:-5}
source "$(dirname "$0")/_env.sh"
OUT=outputs/ckpt_eval/$(echo $CK | tr / _); mkdir -p $OUT
if [ -d $CK/lora ]; then
  [ -d $CK/merged ] || python scripts/eval/merge_lora.py models/Qwen3-1.7B $CK/lora $CK/merged
  M=$CK/merged
else M=models/Qwen3-1.7B; fi
GPU_UTIL=${GPU_UTIL:-0.3} python scripts/eval/eval_gsm8k.py $M $OUT/gsm8k $N 2>&1 | tail -3
python scripts/eval/eval_ckpt_mmbench.py $RCFG $CK $OUT/mmbench $N 2>&1 | tail -3

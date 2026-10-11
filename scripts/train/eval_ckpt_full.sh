#!/bin/bash
# full eval: GSM8K 1319 (merged LoRA, vLLM) + MMBench-EN dev full circular (HF). usage: STU_GPU=1 bash scripts/train/eval_ckpt_full.sh CKPT
CK=$1; TEA_GPU=${STU_GPU:-1}
source "$(dirname "$0")/_env.sh"
OUT=outputs/full_eval/$(echo $CK | tr / _); mkdir -p $OUT
if [ -d $CK/lora ]; then [ -d $CK/merged ] || python scripts/eval/merge_lora.py models/Qwen3-1.7B $CK/lora $CK/merged; M=$CK/merged; else M=models/Qwen3-1.7B; fi
GPU_UTIL=${GPU_UTIL:-0.18} python scripts/eval/eval_gsm8k.py $M $OUT/gsm8k 2>&1 | grep RESULT
python scripts/eval/eval_ckpt_mmbench.py $RCFG $CK $OUT/mmbench 0 2>&1 | grep -E "RESULT|Error|progress"
echo EVAL_FULL_DONE $CK

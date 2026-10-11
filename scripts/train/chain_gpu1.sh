#!/bin/bash
# chain on GPU1: rollout check, teacher circular, full evals warmup/OPD, then SFT + its eval
export STU_GPU=${G:-1} TEA_GPU=${G:-1}
D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
source scripts/train/_env.sh
echo "== rollout_check $(date)"; python scripts/eval/rollout_check.py $RCFG ckpt/demo_opd/final outputs/rollout_check 64 2>&1 | grep -A3 -E "SUMMARY|CAPPED|Error"
echo "== teacher mmbench circular $(date)"; GPU_UTIL=0.2 python scripts/eval/eval_mmbench.py models/Qwen3-VL-4B-Instruct outputs/full_eval/teacher_mmbench 2>&1 | grep -E "RESULT|Error"
echo "== eval warmup $(date)"; bash scripts/train/eval_ckpt_full.sh ckpt/warmup_full/final
echo "== eval opd $(date)"; bash scripts/train/eval_ckpt_full.sh ckpt/demo_opd/final
until grep -q TEACHER_GEN_DONE logs/teacher_gen.log 2>/dev/null; do sleep 30; done
echo "== SFT $(date)"; MODE=sft TAG=sft_full bash scripts/train/run_offline.sh --out ckpt/demo_sft > /dev/null 2>&1; tail -1 ckpt/demo_sft/train_log.jsonl
echo "== eval sft $(date)"; bash scripts/train/eval_ckpt_full.sh ckpt/demo_sft/final
echo CHAIN1_DONE $(date)

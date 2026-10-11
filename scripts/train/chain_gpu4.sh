#!/bin/bash
# chain on GPU4: teacher responses, KD (student+teacher colocated), eval
export STU_GPU=${G:-4} TEA_GPU=${G:-4}
D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
source scripts/train/_env.sh
echo "== teacher_gen $(date)"; GPU_UTIL=0.25 python -u scripts/data/teacher_gen.py > logs/teacher_gen.log 2>&1; tail -2 logs/teacher_gen.log
echo "== KD $(date)"; MODE=kd TAG=kd_full bash scripts/train/run_offline.sh --out ckpt/demo_kd > /dev/null 2>&1; tail -1 ckpt/demo_kd/train_log.jsonl
echo "== eval kd $(date)"; bash scripts/train/eval_ckpt_full.sh ckpt/demo_kd/final
echo CHAIN4_DONE $(date)

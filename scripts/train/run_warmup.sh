#!/bin/bash
# usage: STU_GPU=5 bash scripts/train/run_warmup.sh [extra args e.g. --max_steps 5 --micro 2 --accum 1 --limit 64 --out ckpt/smoke_warmup]
TEA_GPU=${TEA_GPU:-${STU_GPU:-5}}
source "$(dirname "$0")/_env.sh"
TAG=${TAG:-warmup_$(date +%m%d_%H%M)}
python -u -m mm_opd.train.warmup --config $RCFG "$@" 2>&1 | tee logs/$TAG.log

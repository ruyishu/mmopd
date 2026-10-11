#!/bin/bash
# usage: STU_GPU=5 TEA_GPU=4 bash scripts/train/run_opd.sh [--max_steps 5 --micro 2 --accum 1 --no_init --out ckpt/smoke_opd]
source "$(dirname "$0")/_env.sh"
TAG=${TAG:-opd_$(date +%m%d_%H%M)}
python -u -m mm_opd.train.opd --config $RCFG "$@" 2>&1 | tee logs/$TAG.log

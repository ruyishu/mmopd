#!/bin/bash
# usage: MODE=sft|kd STU_GPU=1 TEA_GPU=4 bash scripts/train/run_offline.sh [args]
[ "$MODE" = sft ] && TEA_GPU=${TEA_GPU:-${STU_GPU:-1}}
source "$(dirname "$0")/_env.sh"
TAG=${TAG:-${MODE}_$(date +%m%d_%H%M)}
python -u -m mm_opd.train.offline --config $RCFG --mode $MODE "$@" 2>&1 | tee logs/$TAG.log

#!/bin/bash
# background launcher: bash scripts/train/launch_bg.sh warmup|opd [args]; env STU_GPU/TEA_GPU/TAG
S=$1; shift; D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; TAG=${TAG:-${S}_$(date +%m%d_%H%M)}
(TAG=$TAG setsid nohup bash $D/scripts/train/run_$S.sh "$@" > $D/logs/$TAG.nohup 2>&1 < /dev/null &)
echo "launched $S -> logs/$TAG.log"

#!/bin/bash
# smoke tests (few steps, tiny batches). env STU_GPU / TEA_GPU.
D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
TAG=smoke_warmup bash scripts/train/run_warmup.sh --max_steps 5 --micro 16 --accum 1 --limit 200 --out ckpt/smoke_warmup
TAG=smoke_opd bash scripts/train/run_opd.sh --max_steps 3 --micro 4 --accum 1 --limit 64 --no_init --out ckpt/smoke_opd
echo SMOKE_DONE

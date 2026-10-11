D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
rm -rf ckpt/smoke_opd2
STU_GPU=${STU_GPU:-1} TEA_GPU=${TEA_GPU:-4} TAG=smoke_opd2 bash scripts/train/run_opd.sh --max_steps 2 --micro 4 --accum 8 --limit 128 --no_init --out ckpt/smoke_opd2
echo OPD2_EXIT $?
STU_GPU=${STU_GPU:-1} GPU_UTIL=0.18 bash scripts/train/eval_ckpt.sh ckpt/smoke_warmup/final 20
echo EVAL_EXIT $?
echo SMOKE2_DONE

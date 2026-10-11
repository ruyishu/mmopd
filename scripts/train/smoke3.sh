D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
until grep -q SMOKE2_DONE logs/smoke2.nohup; do sleep 10; done
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
rm -rf ckpt/smoke_opd3
STU_GPU=${STU_GPU:-1} TEA_GPU=${TEA_GPU:-4} TAG=smoke_opd3 bash scripts/train/run_opd.sh --max_steps 3 --micro 4 --accum 8 --limit 128 --no_init --out ckpt/smoke_opd3
echo OPD3_EXIT $?

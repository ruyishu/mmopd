#!/bin/bash
# usage: G=<gpu> JOBS="opd:1k kd:3k ho:ckpt/demo_sft/final ..." bash scripts/train/chain_low.sh
export STU_GPU=$G TEA_GPU=$G
D=/testessfs10/users/zeyu.zhang/zk2/tmp/tmpwdn99-media/mmopd; cd $D
heldout() { CFG=configs/demo_1p7b.yaml source scripts/train/_env.sh >/dev/null; python scripts/eval/heldout_kl.py $RCFG $1 outputs/full_eval/$(echo $1 | tr / _)/heldout 2>&1 | grep -E "RESULT|Error"; }
for J in $JOBS; do
  T=${J%%:*}; A=${J#*:}; echo "== $J $(date)"
  case $T in
    opd) N=$([ $A = 1k ] && echo 1000 || echo 3000); OUTD=ckpt/low${A}_opd
         CFG=configs/lowdata_$A.yaml TAG=low${A}_opd bash scripts/train/run_opd.sh --limit $N --out $OUTD > /dev/null 2>&1
         tail -1 $OUTD/train_log.jsonl; bash scripts/train/eval_ckpt_full.sh $OUTD/final; heldout $OUTD/final ;;
    kd)  N=$([ $A = 1k ] && echo 1000 || echo 3000); OUTD=ckpt/low${A}_kd
         CFG=configs/lowdata_$A.yaml MODE=kd TAG=low${A}_kd bash scripts/train/run_offline.sh --limit $N --out $OUTD > /dev/null 2>&1
         tail -1 $OUTD/train_log.jsonl; bash scripts/train/eval_ckpt_full.sh $OUTD/final; heldout $OUTD/final ;;
    ho)  heldout $A ;;
  esac
done
echo CHAIN_DONE $(date)

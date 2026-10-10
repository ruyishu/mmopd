#!/usr/bin/env bash
# 多模态评测入口。encoder-free 学生还没注册进 VLMEvalKit 或 lmms-eval，
# 所以这个脚本不启动打分，也不写结果文件。
set -euo pipefail

cat <<'EOF'
多模态任务：MMBench、MMMU、OCRBench、TextVQA、BLINK 空间子集、VSR、POPE。
工具二选一：VLMEvalKit，或 lmms-eval。
VSR 如果套件里没有现成任务，单独加载，不从清单里删掉。
POPE 用来看幻觉。OCRBench 和 TextVQA 即使低也要报。

学生还没有对应的模型适配器。下面是以后要接上的命令形状，现在不能跑：
  VLMEvalKit: python -m vlmeval.run --model mm_opd --data <kit 里的 MMBench、MMMU、OCRBench、TextVQA、BLINK、POPE id>
  lmms-eval:  python -m lmms_eval --model mm_opd --tasks <同上>

任务 id 以安装的套件为准，不在这里写死。本脚本不产生分数。
EOF
exit 2

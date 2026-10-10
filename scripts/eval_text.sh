#!/usr/bin/env bash
# 用 lm-eval-harness 评纯文本骨干：MMLU、GSM8K、ARC-Challenge。
# 传入的是文本 LLM（底座或已经合并的 LoRA），不是带投影的学生，也不喂图像。
# 依赖：uv sync --extra eval
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "用法: scripts/eval_text.sh MODEL_PATH [额外的 lm-eval 参数...]" >&2
  echo "任务: mmlu,gsm8k,arc_challenge。不写死分数。" >&2
  exit 1
fi

MODEL="$1"
shift

if ! uv run --extra eval python -c "import lm_eval" >/dev/null 2>&1; then
  echo "没有 lm-eval。先执行: uv sync --extra eval" >&2
  exit 1
fi

mkdir -p "${ROOT}/outputs/text"
exec uv run --extra eval lm-eval \
  --model hf \
  --model_args "pretrained=${MODEL},dtype=bfloat16" \
  --tasks mmlu,gsm8k,arc_challenge \
  --batch_size 1 \
  --output_path "${ROOT}/outputs/text" \
  "$@"

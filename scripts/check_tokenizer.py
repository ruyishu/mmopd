"""用学生和教师的 tokenizer 编码同一批文本，比较 token id。

只下载 tokenizer，不下载模型权重。id 不一致时退出码为 1。
打印的 vocab_size 是 tokenizer 自己的大小，小于模型 config 里垫过的 151936。
这里要的是两边 encode 结果一致。
默认模型对对应 configs/small_1p7b.yaml。主配置：

    uv run python scripts/check_tokenizer.py \\
        --student Qwen/Qwen3-4B \\
        --teacher Qwen/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import sys

from transformers import AutoTokenizer

TEXTS = (
    "Describe the spatial relation between the cup and the book.",
    "图中红色方块在蓝色方块的哪一侧？",
    "What is 17 + 28?",
    "<|vision_start|><|image_pad|><|vision_end|>",
)

CHAT = [{"role": "user", "content": "图里有几个人？只回答数字。"}]


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _chat_ids(tokenizer) -> list[int] | None:
    if not getattr(tokenizer, "chat_template", None):
        return None
    rendered = tokenizer.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
    return _encode(tokenizer, rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    student = AutoTokenizer.from_pretrained(args.student)
    teacher = AutoTokenizer.from_pretrained(args.teacher)

    ok = True
    print(f"student {args.student} vocab_size={student.vocab_size} len={len(student)}")
    print(f"teacher {args.teacher} vocab_size={teacher.vocab_size} len={len(teacher)}")
    if student.vocab_size != teacher.vocab_size or len(student) != len(teacher):
        ok = False
        print("MISMATCH vocab size")

    for text in TEXTS:
        left = _encode(student, text)
        right = _encode(teacher, text)
        match = left == right
        ok = ok and match
        status = "ok" if match else "MISMATCH"
        preview = text if len(text) < 48 else text[:45] + "..."
        print(f"{status} encode {preview!r} student={left[:12]} teacher={right[:12]} len={len(left)}/{len(right)}")

    left_chat = _chat_ids(student)
    right_chat = _chat_ids(teacher)
    if left_chat is None or right_chat is None:
        ok = False
        print("MISMATCH chat template missing on one tokenizer")
    else:
        match = left_chat == right_chat
        ok = ok and match
        status = "ok" if match else "MISMATCH"
        print(f"{status} chat_template ids len={len(left_chat)}/{len(right_chat)}")

    if ok:
        print("tokenizers match on the sample texts")
        return 0
    print("tokenizers differ", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

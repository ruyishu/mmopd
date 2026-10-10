"""回答 token 上的逐 token reverse KL。

KL(student ‖ teacher) 是 mode-seeking 的方向，和 plan.md 里的主损失一致。
不乘 temperature 的平方。全词表计算；按 top-k 分块是以后的显存优化，这里没有。
"""

import torch
from torch import Tensor


def reverse_kl(
    student_logprobs: Tensor,
    teacher_logprobs: Tensor,
    answer_mask: Tensor,
) -> Tensor:
    """student_logprobs、teacher_logprobs 形状 (B, T, V)，已经是 log-softmax。

    answer_mask 形状 (B, T)，回答位置为 1。没有回答 token 时抛错，不返回 0。
    """
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            f"logprob shapes differ: {tuple(student_logprobs.shape)} vs {tuple(teacher_logprobs.shape)}"
        )
    if student_logprobs.ndim != 3:
        raise ValueError(f"expected logprobs of shape (B, T, V), got {tuple(student_logprobs.shape)}")
    if answer_mask.shape != student_logprobs.shape[:2]:
        raise ValueError(
            f"answer_mask shape {tuple(answer_mask.shape)} != {(student_logprobs.shape[0], student_logprobs.shape[1])}"
        )
    mask = answer_mask.to(dtype=torch.float32)
    denom = mask.sum()
    if int(denom.item()) == 0:
        raise ValueError("answer_mask has no answer tokens")
    student = student_logprobs.float()
    teacher = teacher_logprobs.float()
    per_token = torch.sum(student.exp() * (student - teacher), dim=-1)
    return torch.sum(per_token * mask) / denom


def reverse_kl_from_logits(
    student_logits: Tensor,
    teacher_logits: Tensor,
    answer_mask: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """先按 temperature 做 log-softmax，再算 reverse KL。temperature 必须为正。"""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    student_logprobs = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logprobs = torch.log_softmax(teacher_logits.float() / temperature, dim=-1)
    return reverse_kl(student_logprobs, teacher_logprobs, answer_mask)

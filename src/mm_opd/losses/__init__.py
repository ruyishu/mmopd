"""蒸馏损失。主损失是回答 token 上的 reverse KL。"""

from mm_opd.losses.reverse_kl import reverse_kl, reverse_kl_from_logits

__all__ = ["reverse_kl", "reverse_kl_from_logits"]

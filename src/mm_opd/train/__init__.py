"""两阶段训练的入口。两个入口都还没实现。"""

from mm_opd.train.opd import run_opd
from mm_opd.train.warmup import run_warmup

__all__ = ["run_opd", "run_warmup"]

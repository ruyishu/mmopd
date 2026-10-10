"""无编码器学生：patch 投影加文本 LLM。

这里只放结构和哪一阶段训练哪些参数。不下载权重，不生成回答。
"""

from dataclasses import dataclass

from torch import Tensor, nn

from mm_opd.model.patch_projector import PatchProjector


@dataclass(frozen=True)
class StudentSpec:
    """学生配置。hidden_size 是文本 LLM 的隐层，不是教师的。"""

    model_name: str
    hidden_size: int
    patch_size: int = 14
    projector: str = "linear"


def modules_for_stage(stage: str) -> tuple[str, ...]:
    """预热只训投影。OPD 训投影和 LLM 的 LoRA。LLM 主体保持冻结。"""
    if stage == "warmup":
        return ("projector",)
    if stage == "opd":
        return ("projector", "llm.lora")
    raise ValueError(f"unknown stage: {stage!r}")


class EncoderFreeStudent(nn.Module):
    """文本 LLM 外面套一个可训练投影。LLM 本体由训练入口以后再挂上。"""

    def __init__(self, spec: StudentSpec) -> None:
        super().__init__()
        self.spec = spec
        self.projector = PatchProjector(
            patch_size=spec.patch_size,
            hidden_size=spec.hidden_size,
            kind=spec.projector,
        )
        self.llm = None

    def patch_tokens(self, images: Tensor) -> Tensor:
        """只做投影，不进 LLM。"""
        return self.projector(images)

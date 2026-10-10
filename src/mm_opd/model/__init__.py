"""学生侧模块：patch 投影和包住文本 LLM 的学生。"""

from mm_opd.model.patch_projector import PatchProjector
from mm_opd.model.student import EncoderFreeStudent, StudentSpec, modules_for_stage

__all__ = [
    "EncoderFreeStudent",
    "PatchProjector",
    "StudentSpec",
    "modules_for_stage",
]

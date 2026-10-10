"""训练数据路径和子集下标。不内置样本，也不下载数据。"""

from mm_opd.data.datasets import (
    OPD_FRACTIONS,
    opd_annotation_paths,
    sample_indices,
    warmup_annotation_path,
)

__all__ = [
    "OPD_FRACTIONS",
    "opd_annotation_paths",
    "sample_indices",
    "warmup_annotation_path",
]

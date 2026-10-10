"""预热与 OPD 的数据合同。

预热用 LLaVA-Pretrain 558K 的 5万–10万对，监督是图文对本身。
OPD 只用图像和问题，回答由学生采样，标注答案不进损失。
文件不在时抛 FileNotFoundError，不返回空数据集冒充「零样本」。
下载步骤见 data/README.md。
"""

import random
from pathlib import Path

OPD_FRACTIONS = (0.01, 0.05, 0.10, 1.0)
WARMUP_SUBSET_RANGE = (50_000, 100_000)

def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError("repo root not found; run from a source checkout")


def data_root() -> Path:
    return _repo_root() / "data"


def sample_indices(n: int, fraction: float, seed: int) -> list[int]:
    """用固定种子打乱后取前 k 个下标，再排序。

    同一个 n 和 seed 下，较小的 fraction 是较大 fraction 的子集。
    fraction 取 1 时返回全部下标。n 为 0 时返回空列表。
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if n == 0:
        return []
    k = n if fraction == 1 else int(round(n * fraction))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return sorted(order[:k])


def warmup_annotation_path() -> Path:
    """LLaVA-Pretrain 558K 的对话标注。图像在同目录的 images/。"""
    path = data_root() / "llava-pretrain" / "blip_laion_cc_sbu_558k.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; see data/README.md")
    return path


def opd_annotation_paths() -> dict[str, Path]:
    """OPD 提示的标注路径。读取时只保留图像和问题。"""
    paths = {
        "llava-instruct-665k": data_root() / "llava-instruct" / "llava_v1_5_mix665k.json",
        "sharegpt4v": data_root()
        / "sharegpt4v"
        / "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing " + ", ".join(missing) + "; see data/README.md")
    return paths

"""把原始图像切成 patch，再投到文本 LLM 的隐层维度。

没有 ViT，也没有 SigLIP。输入是像素，输出是视觉 token。
"""

from torch import Tensor, nn


class PatchProjector(nn.Module):
    """14×14 一类的 raw patch，经线性层或两层 MLP 变成视觉 token。

    图像须能被 patch 整除，否则直接报错，不裁剪。像素按调用方给定的 [0, 1]
    范围先映射到 [-1, 1]。本模块不加载检查点。
    """

    def __init__(self, patch_size: int, hidden_size: int, kind: str = "linear") -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.kind = kind
        in_features = patch_size * patch_size * 3
        if kind == "linear":
            self.proj: nn.Module = nn.Linear(in_features, hidden_size)
        elif kind == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"projector kind must be 'linear' or 'mlp', got {kind!r}")

    def forward(self, images: Tensor) -> Tensor:
        """images: float tensor，形状 (B, 3, H, W)，数值约在 [0, 1]。

        返回 (B, num_patches, hidden_size)，patch 按行优先。
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images of shape (B, 3, H, W), got {tuple(images.shape)}")
        batch, _, height, width = images.shape
        patch = self.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"image size {(height, width)} is not divisible by patch_size {patch}"
            )
        scaled = (images - 0.5) / 0.5
        # (B, 3, n_h, n_w, patch, patch) -> (B, n_h * n_w, 3 * patch * patch)
        tiles = scaled.unfold(2, patch, patch).unfold(3, patch, patch)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(batch, -1, 3 * patch * patch)
        return self.proj(tiles)

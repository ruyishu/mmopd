"""阶段 1：冻住文本 LLM，只在 LLaVA-Pretrain 子集上训练投影。

损失是图文对的 next-token 交叉熵。这一阶段不调用教师。
"""


def run_warmup(config_path: str) -> None:
    """读取 configs/*.yaml 并训练投影。

    未实现。不会下载模型，不会写指标。
    """
    raise NotImplementedError(
        f"warmup is not implemented (config {config_path}). Data contract: plan.md section 4."
    )

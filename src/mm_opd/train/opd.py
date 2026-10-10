"""阶段 2：学生在图像+问题上自采样，教师给逐 token reverse KL。

更新投影和 LLM 的 LoRA。纯文本正则用原来的文本 LLM，不用只读文本的 VLM。
"""


def run_opd(config_path: str) -> None:
    """读取 configs/*.yaml 并做 on-policy distillation。

    未实现。不会采样，不会写指标。
    """
    raise NotImplementedError(
        f"OPD is not implemented (config {config_path}). Data contract: plan.md section 4."
    )

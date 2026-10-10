# MM-OPD 算力与模型选择（2×L20 48GB，无 NVLink）

## 词表核对（HF config.json，已实测）
Qwen3-1.7B / 4B / 8B 与 Qwen3-VL-2B / 4B / 8B-Instruct 的 vocab_size 均为 151936，hidden 一一对应（1.7B↔VL-2B: 2048；4B↔VL-4B: 2560；8B↔VL-8B: 4096）。逐 token KL 可直接算，无需词表映射。
注意：VL 模型的视觉特殊 token（<|vision_start|> 等）在 Qwen3 词表里已存在，学生可复用作图像占位符；建议正式实验前再用两个 tokenizer 对同一批文本做 encode 一致性检查。

## 推荐配置
- 主配置：教师 Qwen3-VL-8B-Instruct（bf16，vLLM/HF 推理，GPU1）；学生 Qwen3-4B + 线性投影 + LoRA（GPU0）。
- 快速迭代：教师 Qwen3-VL-4B；学生 Qwen3-1.7B。也可单卡跑通全流程。
- 冲上限（可选）：学生 Qwen3-8B + LoRA，需梯度检查点和更小 batch，较紧。

## 显存粗估（估算，非实测）
- 教师 VL-8B bf16 权重 ≈ 17GB，加图像 token 的 KV/激活，GPU1 留 ~25–30GB 余量，够用。
- 学生 4B bf16 权重 ≈ 8GB；LoRA + 投影的梯度/优化器 < 2GB；激活（梯度检查点，seq≈2k 含图像 token）≈ 10–20GB；合计 ≈ 25–35GB。
- 学生采样：在 GPU0 上同进程 vLLM（gpu_memory_utilization≈0.3）或用 HF generate；vLLM 需每 N 步同步 LoRA/投影权重（投影层是自定义的，vLLM 原生不支持，初期用 HF generate 更省事）。
- 只取答案位置的教师 logits：vocab 151936 × 答案长度，fp32 下每 1k token ≈ 0.6GB，可 top-k 或分块算 KL。

## OPD 每步流程
学生采样回答 → 教师对（图像+问题+学生回答）做一次前向拿 logits → 答案 token 上算反向 KL → 更新投影层 + LoRA。
瓶颈是学生生成；建议先做阶段1热身（冻 LLM 只训投影），否则采样质量太差。

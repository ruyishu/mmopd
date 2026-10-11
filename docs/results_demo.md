# mm-opd demo 结果（2026-10-11）

## 设置
- **教师**：Qwen3-VL-4B-Instruct，全程冻结。
- **学生**：Qwen3-1.7B + 冻结的 Qwen3-VL-4B 视觉塔（`model.visual`，含 merger，不用 deepstack）+ 可训练 2 层 MLP 投影（2560→2048）。图像固定 448×448 → 196 个 image token，嵌入替换 `<|image_pad|>`。原始 patch 投影路径保留在 `vision.kind: patch`。
- **预热**：50k COCO caption 对，782 步，global 64，lr 1e-3，只训投影（LLM、ViT 冻结）。
- **阶段 2**（OPD / 离线 KD / SFT）：同一批 10k LLaVA-Instruct-665K COCO 问题（只取第一轮问题），从预热投影初始化，投影 + LoRA（r16, α32, q/k/v/o），lr LoRA 1e-4 / 投影 5e-5，20 步 warmup + cosine，300 步，global 32（micro 4 × accum 8）。
  - OPD：学生用 vLLM 自采样（temp 1.0, top_p 1.0, max 512），教师用自己的模板和处理器看原图，对学生回答 token 打分，逐 token reverse KL（全词表，分块计算）。
  - 离线 KD：在教师回答上 teacher-forcing，reverse KL（同时记录 forward KL）。教师回答由 vLLM 生成（temp 0.7 / top_p 0.8 / top_k 20，max 512）。
  - SFT：教师回答上的交叉熵。
- 全程非思考模式（`enable_thinking=False`）。
- **评测**
  - GSM8K 全量 1319 题：合并 LoRA 后用 vLLM，temp 0.7 / top_p 0.8 / top_k 20 / presence 1.5，max 1024，seed 0，提示要求 `\boxed{}`。
  - MMBench-EN dev 全量 1164 题：数据集自带的旋转按 `index % 1e6` 分组，全部旋转都对才算 circular 对；单次准确率只看原序行；解析失败算错；max 64 token。学生走 HF generate（HF 没有 presence_penalty）。

## 基线
- 学生 Qwen3-1.7B GSM8K（全量）：83.09（warmup 那次复跑 82.49，同一 LLM，差异来自采样）。
- 教师 Qwen3-VL-4B MMBench dev：circular **84.62** / 单次 **89.18**（官方 V1.1 为 83.9）。

## 四路对比

| ckpt | 数据 / 步数 | MMBench circular | 单次 | GSM8K | 训练时间 |
|---|---|---|---|---|---|
| warmup | 50k caption / 782 | 34.71 | 56.27 | 82.49 | 约 45 min |
| SFT | 10k 教师回答 / 300 | 44.76 | 61.43 | 81.43 | 50.8 min |
| 离线 KD | 10k 教师回答 / 300 | **51.55** | **64.18** | 81.50 | 110.0 min |
| OPD | 10k 问题 / 300 | 51.29 | 63.83 | 81.43 | 约 135 min |

- warmup 有 405/4329 行解析失败（算错），其余模型全部以字母开头作答。
- 时间：SFT 和 KD 来自日志里的 `train_minutes`，OPD 按日志时间戳估算（含 vLLM 启动），warmup 按文件时间估算。GPU 和别人共享，时间只供参考。

## OPD 训练曲线（`results/train_logs/demo_opd/train_log.jsonl`）
- reverse KL：前 20 步均值 2.62（回答很短，约 14 token，类似 caption），约 50 步时降到 0.63，此后平台在 0.54–0.60。
- 平均回答长度：从约 14 token 涨到 130–160；约 40 步以后，每步都至少有一条 rollout 撞到 512 上限。
- grad norm：后期 0.4–0.9，步 10 和步 150 各有一次尖峰（13.8 / 5.1）。

## Rollout 检查（OPD final，64 个训练问题，temp 1.0）
- 12.5% 撞到 512 上限，平均 176 token。
- 4-gram 重复率：整体 0.034，撞上限的那些平均 0.148。
- 只有 1 条明显退化（重复 “The person in the front/back is wearing …”）；其余长回答连贯。

## 已知问题
- MMBench 学生评测走 HF 采样，没有 presence_penalty，和教师（vLLM）的设置不完全一致。
- GPU 与其他用户共享，速度波动大。
- 共享盘 `/testessfs10` 已满（2026-10-11 12:48），后续实验被阻塞。
- 每个设置只跑了一个 seed，样本内差异（例如 KD 和 OPD 差 0.3 分）在噪声范围内。

## 结论与下一步
- 在 10k 问题、300 步的规模下：**OPD ≈ 离线 KD > SFT > warmup**（MMBench circular 51.3 / 51.6 / 44.8 / 34.7）。GSM8K 只比 warmup 降约 1 分（82.5 → 81.4–81.5）。
- OPD 的样本效率优势在这个规模下没有显现。
- 下一步（代码和配置已就绪：`configs/lowdata_{1k,3k}.yaml`、`scripts/train/chain_low.sh`、`scripts/eval/heldout_kl.py`），等磁盘空间：
  - 低数据 1k / 3k 下的 OPD vs KD（各 3 个 epoch）。
  - 200 个 held-out 长回答，用教师打分的 reverse KL 和 NLL 衡量离教师有多近。

## 结果文件
`results/`：base_eval_*、ckpt_eval（200 样本）、full_eval（全量）的 `*_result.json`；`rollout_check/summary.json`；`train_logs/*/train_log.jsonl`。

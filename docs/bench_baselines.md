# mm-opd 公开基准结果汇总（只收录官方或原论文公布的数字）

> 规则：数字都取自下面列出的原始来源，没有编造；"未公布"表示原始来源里没有这个数。不同来源的评测设置不同，不能直接横向比较。

## 表 A：老师模型（Qwen3-VL，多模态）
来源：Qwen3-VL 技术报告 Table 4（https://arxiv.org/abs/2511.21631）。HF 模型卡（https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct）里的成绩是图片形式，没有逐项核对。

| 模型 | MMBench-EN（V1.1） | MMBench-CN | MMMU（val） | OCRBench | TextVQA | POPE | BLINK | VSR |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | 78.4 | 75.9 | 53.4 | 858 | 未公布 | 未公布 | 53.8 | 未公布 |
| Qwen3-VL-4B-Instruct | 83.9 | 83.5 | 67.4 | 881 | 未公布 | 未公布 | 65.8 | 未公布 |
| Qwen3-VL-8B-Instruct | 84.5 | 84.7 | 69.6 | 896 | 未公布 | 未公布 | 69.1 | 未公布 |
| Qwen3-VL-2B-Thinking | 79.9 | 78.8 | 61.4 | 792 | 未公布 | 未公布 | 57.2 | 未公布 |
| Qwen3-VL-4B-Thinking | 84.6 | 83.8 | 70.8 | 808 | 未公布 | 未公布 | 63.4 | 未公布 |
| Qwen3-VL-8B-Thinking | 85.3 | 85.5 | 74.1 | 819 | 未公布 | 未公布 | 64.7 | 未公布 |

注：OCRBench 满分 1000。报告正文写的是 MMBench-V1.1，表里标的是 MMBench-EN/CN。MMMU 的表头没写是哪个划分，按惯例应该是 val，但没核实。报告里有些相关替代基准：HallusionBench（8B-Instruct 61.1，可替代 POPE 看幻觉）、VSI-Bench、EmbSpatialBench、RefSpatialBench（空间理解，可部分替代 VSR）、DocVQA、InfoVQA、ChartQA。

## 表 B：学生模型 / 纯文本
### B1 Base 模型（预训练，未做后训练）
来源：Qwen3 技术报告 Table 6/7/8（https://arxiv.org/abs/2505.09388）。设置：MMLU 5-shot，MMLU-Redux 5-shot，GSM8K 4-shot CoT。

| 模型 | MMLU | MMLU-Redux | GSM8K | ARC-C |
|---|---|---|---|---|
| Qwen3-1.7B-Base | 62.63 | 61.66 | 75.44 | 未公布 |
| Qwen3-4B-Base | 72.99 | 72.79 | 87.79 | 未公布 |
| Qwen3-8B-Base | 76.89 | 76.17 | 89.84 | 未公布 |

### B2 后训练模型（非思考 / Instruct 模式）以及 Qwen3-VL 的纯文本成绩
来源：Qwen3-VL 技术报告 Table 6（Instruct）和 Table 7（Thinking）。报告没有给出 MMLU 原版、GSM8K、ARC-C 的成绩。

| 模型 | MMLU-Redux（Instruct） | MMLU-Pro（Instruct） | MMLU-Redux（Thinking） | MMLU-Pro（Thinking） |
|---|---|---|---|---|
| Qwen3-1.7B | 63.6 | 42.3 | 73.9 | 58.1 |
| Qwen3-4B | 77.3 | 58.0 | 83.7 | 70.4 |
| Qwen3-8B | 79.5 | 63.4 | 87.5 | 74.6 |
| Qwen3-VL-2B（只喂文本） | 66.5 | 49.0 | 76.9 | 62.3 |
| Qwen3-VL-4B（只喂文本） | 81.5 | 67.1 | 86.0 | 73.6 |
| Qwen3-VL-8B（只喂文本） | 84.9 | 71.6 | 88.8 | 77.3 |

注：同等规模下，Qwen3-VL 只喂文本时的成绩高于对应的 Qwen3。这说明在纯文本任务上，VLM 老师确实可能比纯文本学生更强，对 OPD 的可行性是个利好。

## 表 C：已发表的 SFT / 无编码器参考基线
来源：EVEv2 论文 Table 2（https://arxiv.org/abs/2502.06788）。这张表把 LLaVA-1.5、Fuyu、EVE、Mono-InternVL、EVEv2 放在同一张表里比较，所以优先用它。补充来源：LLaVA-1.5 原论文（https://arxiv.org/abs/2310.03744）、EVE 原论文（https://arxiv.org/abs/2406.11832）、Mono-InternVL 原论文（https://arxiv.org/abs/2410.08202）。

| 模型 | 是否有视觉编码器 | MMBench-EN | MMMU | OCRBench | TextVQA | POPE | BLINK | VSR |
|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B（Vicuna，558K 预训练 + 665K SFT） | 有（CLIP ViT） | 64.3 | 35.3 | 318 | 46.1（EVEv2 表）；58.2（原论文，带 OCR token） | 85.9 | 未公布 | 未公布 |
| Fuyu-8B | 无 | 10.7 | 27.9 | 366 | 未公布 | 84.0 | 未公布 | 未公布 |
| EVE-7B (HD) | 无 | 52.3 | 32.6 | 398 | 56.8 | 85.0 | 未公布 | 未公布 |
| Mono-InternVL-2B（1.8B 激活参数） | 无 | 65.5 | 33.7 | 767 | 72.6 | 未公布 | 未公布 | 未公布 |
| EVEv2.0-7B | 无 | 66.3 | 39.3 | 702 | 71.1 | 87.6 | 未公布 | 未公布 |

注：Fuyu 的 MMBench 只有 10.7，大概率是输出格式没对齐造成的，不代表真实能力。EVE-7B（非 HD）在 EVE 论文里的成绩是 MMB 49.5、TextVQA 51.9、POPE 83.6。

### 基于 Qwen3 小模型的 LLaVA 式 SFT 公开结果
- LLaVA-OneVision-1.5（https://www.lmms-lab.com/posts/llava_onevision_1_5）：8B 版本 MMBench-EN 84.1、MMMU val 55.4；4B 版本 MMBench-EN 84.2、MMMU val 52.7。这是带 ViT 的 LLaVA 式模型，用了约 85M 预训练数据加 22M 指令数据；LLM 底座据称是 Qwen3，没有逐一核实。
- LLaVA-OneVision-2（https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2）：有 Qwen3-4B-Instruct-2507 的训练配方，但暂时没找到公开的 Qwen3-4B 版成绩。
- 无编码器 + Qwen3 小模型的公开结果：这次没有找到。检索得比较粗，只做了一次网页搜索。

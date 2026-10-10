# mm-opd 研究计划

## 1. 一句话核心 idea 与卖点

- 给纯文本 LLM 原生多模态能力，但不加 ViT/SigLIP：原始图像 patch 经线性层或浅层 MLP 投影后进入文本 LLM（Fuyu / EVE 式 encoder-free）。
- 不用大规模 SFT。学生在图文输入上自己采样回答，同族 VLM 教师在学生轨迹上给逐 token reverse KL。短预热（投影对齐 / 轻量 SFT）在前，因为随机投影看不见图像。
- 卖点针对 encoder-free 的两个老问题：投影学不会看，所以吃数据；多模态 SFT 冲掉语言能力。教师已经会看图，学生把生成分布对齐过去，用少量数据接近教师的多模态水平；on-policy 的 reverse KL 留在学生自己的分布附近，减轻遗忘。WINGS（NeurIPS 2024）里，多模态微调后 16 个纯文本数据集有 12 个出现遗忘，这是要压住的失败模式。
- 空白：没有工作同时做到「多模态教师 → 文本底座 / encoder-free 学生 → OPD」。判断来自 2026-10-10 的检索（arXiv 与网页）；Semantic Scholar 被限流，不能当作完备证明。

最近邻（都不构成同一设定）：

- 架构是 encoder-free，训练是预训练或 SFT：Fuyu、EVE / EVEv2、Mono-InternVL。
- OPD 但学生仍是 VLM，或方向相反：LEGO-OPD（LLM + VLM 教师 → VLM 学生，arXiv:2610.00333）；H-OPD（VL 教师与文本教师按置信度仲裁，监督 VLM，arXiv:2607.02592）；Visual-OPSD（模型内部自蒸馏，学生仍输入原图，arXiv:2606.18974）；VOLD（文本教师 → VLM，arXiv:2510.23497）。
- 跨模态 OPD，但不是「图 → 无编码器文本学生」：X-OPD（文本 → 语音 LLM，arXiv:2603.24596）、OPOD（多模态教师 → omni 学生，arXiv:2607.20918）、JO-OPD（同一语音模型内 S2T → S2TS，arXiv:2609.15313）。同方向还有 X³-OPD（文本 → 音频-语言，arXiv:2607.21550）。
- 直接约束方法的负面结果：VOLD 没有 SFT 冷启动则 OPD 无效；S-OPD（arXiv:2609.39120）指出瓶颈是学生感知，不是教师监督本身；LIM（arXiv:2605.12517）显示 VLM 只读文本通常弱于底座 LLM；H-OPD 里「文本教师 + 图像描述」在 LogicVista 上为 26.6%，低于学生基线 35.8%。ARMADA（arXiv:2603.10877）是表示级 KD，教师不是 VLM，也不是 on-policy。

## 2. 方法

架构

- 学生：文本 LLM + 一个可训练投影。图像切成固定大小的 raw patch，像素归一化后经线性层或两层 MLP 映到词嵌入维度，按光栅顺序作为视觉 token 插入文本序列。无视觉编码器。
- 教师：同族 VLM，全程冻结。主族用 Qwen3-VL 教 Qwen3。教师用自己的视觉编码器看图，学生用 patch 投影看图，两侧视觉 token 不对齐，也不做隐表示蒸馏。
- 监督只打在生成的文本 token 上。词表必须相同，reverse KL 才有定义。主实验不做跨词表对齐。

两阶段

- 预热：只训投影，LLM 冻结。少量图文对上做轻量 SFT（短回答或 caption）。目标是让图像 token 改变下一词分布，不是刷榜。
- OPD：学生在图像 + 文本指令上采样完整回答。教师看到同一张图、同一指令和学生已写出的前缀，给出下一词分布。损失是回答 token 上的 reverse KL，即 KL(π_student ∥ π_teacher)，可加温度。主损失不用 forward KL。
- 参数：预热只更新投影；OPD 更新投影加 LLM 的 LoRA。不从头训练 LLM。
- 文本侧：纯文本 prompt 不交给只读文本的 VLM。用冻结的原文本 LLM 做 token 级 KL 正则，或混入一小部分以原 LLM 为教师的文本 OPD。

## 3. 关键实验与基线

同一图文数据池、同一子采样比例、后三条共用同一预热，学生侧可训练 FLOPs 对齐。另报端到端 wall-clock，教师前向单独列，不算进学生优势。

- B1 LLaVA 式：冻结 SigLIP 或同类 ViT + MLP + SFT。有编码器的常规路线。
- B2 encoder-free + SFT：与本文相同的 patch 投影，监督是硬标签。
- B3 encoder-free + 离线 KD：教师在固定标注上 teacher-forcing，学生最小化 forward KL 或 soft CE。学生不采样。
- B4 encoder-free + OPD：本文。

B4 对 B2、B3 的差别只在目标是否 on-policy、是否 reverse KL。B1 用来回答去掉编码器之后还剩多少。全量数据下不要求超过 B1。

## 4. 数据与评测

环境用 uv，Python 3.11，依赖在 `pyproject.toml`。下面是数据合同，不是已跑完的实验。

数据

- 预热：LLaVA-Pretrain 558K 中抽 5万–10万对（小配置 5万，主配置 10万）。只训投影，LLM 冻结。损失是图文对的 next-token 交叉熵。
- OPD：提示只留图像和问题，回答由学生自己采样。来源是 LLaVA-Instruct-665K 与 ShareGPT4V。标注答案不进损失。
- 缩放：同一 OPD 提示池的 1%、5%、10%、全量。种子固定，小比例是大比例的子集。四条基线共用这些子集。
- OCR 可选：投影对文字明显不够时，才把 TextVQA、DocVQA 的训练集加进预热。评测集不进训练。

评测

- 多模态：MMBench、MMMU、OCRBench、TextVQA、BLINK 空间子集、VSR、POPE。工具用 VLMEvalKit 或 lmms-eval。POPE 看幻觉。OCRBench 与 TextVQA 预期弱，必须报。VSR 若套件里没有现成任务，单独加载，不删这项。
- 文本：MMLU、GSM8K、ARC-Challenge。工具用 lm-eval-harness。报相对底座的差值。文本评测只走纯文本骨干，不喂图像。
- 多模态主分是上面七项的宏平均，同时附分项。解码设置统一。

## 5. 核心消融

- 目标函数：OPD vs SFT vs 离线 KD。预热、数据、可训练参数相同。
- 数据缩放：第 4 节的 1%、5%、10%、全量。四条基线都画，看 B4 是否在低数据区抬起曲线。
- 教师输入：原图 vs 只看 caption。学生始终吃 patch。二者打平则视觉特权不成立。
- 预热长度：0、短（输出不再崩溃为止）、与完整 SFT 等步数。预期 0 步 OPD 无效。
- 模型：两族，每族至少两个规模。主族 Qwen3-VL → Qwen3。第二族只要求与其 VLM 教师词表一致，配对在可行性阶段确定。小规模跑完全部消融；大规模只重复主对比和缩放曲线上的两个点。

## 6. 能让审稿人给 accept 的目标结果

以下是投稿门槛，不是已有数字。

- 数据效率：数个百分点的数据下，encoder-free + OPD 达到同族教师多模态宏平均的 80–90%。
- 同数据、同预热下，OPD 高于 encoder-free SFT 和离线 KD；低数据区相对有编码器的 SFT 缩小大部分差距。
- 文本：MMLU 与 GSM8K 相对底座不降；相对吃过同样多模态数据的 SFT，OPD 更高。
- 两族、至少两个规模上，数据效率与文本保持的方向一致。
- 正文里的分析要能单独支撑主张：缩放曲线；原图教师相对 caption 教师的差距；预热为 0 时 OPD 失败；空间题与 OCR 题的误差拆分。读完应能看出增益来自 on-policy，以及教师看见了图像。

低数据达不到教师的 80%，或文本相对底座明显下降，就不用这个主张投稿。

## 7. 风险与对策

- 教师纯文本不强，甚至弱于底座。LIM：LLaVA-1.5-7B 在 MMLU+ARC 上 39.77%，Vicuna-1.5-7B 为 53.87%；Qwen2.5-VL-7B 为 81.00%，Qwen2.5-7B 为 83.38%。对策：图文 OPD 的教师看原图；纯文本正则只用原 LLM。开跑前用小规模 pass@k 确认教师在目标图文任务上强于预热后的学生。图文也不强就换教师。
- OCR 与细粒度弱。raw patch 的线性投影对小字、密集文字不利；S-OPD 表明感知不够时更强的教师帮不上。对策：主张放在综合理解、空间和数据效率上；OCRBench 与 TextVQA 全报。预热可加少量 TextVQA / DocVQA 训练集（第 4 节），但不加回 ViT。与 B1 的差距若集中在 OCR、综合题接近，就按这个边界写。
- 训练不稳定。分布差太远时 reverse KL 没有有用梯度或直接发散。对策：预热是方法的一部分；KL 使用温度，JSD 只作对照，不替换主损失；先训投影再开 LoRA；监控回答长度、KL 和空回答比例。
- 算力对比不公平。看图的 VLM 教师比 SFT 贵。对策：主表同时给学生侧 FLOPs 和含教师的 wall-clock。主张的是样本效率。
- 词表或模板不一致，KL 打在错误 token 上。对策：同族、同一 chat 模板；损失只覆盖回答 token。

## 8. 时间线

日期全部 TBD。按 CVPR 式截稿倒排，截稿日未知，不写日历。

- M1 可行性（TBD）：用 `scripts/check_tokenizer.py` 核对同族词表与 chat 模板；教师对预热学生的图文 pass@k；确认短预热后图像能改变输出。不通过则停止。
- M2 主族小模型（TBD）：跑通预热和四条基线，得到第一条缩放曲线。
- M3 消融（TBD）：原图 vs caption、预热长度、OPD vs 离线 KD。
- M4 第二规模与第二族（TBD）：只重复主对比和缩放曲线的关键点。
- M5 分析与写作（TBD）：文本变化、误差拆分、成稿。
- M6 内审与投稿（TBD）：对照第 6 节决定投 CVPR，或改投备选。

## 9. 目标会议

- 主投 CVPR。问题是视觉系统：去掉编码器之后，用什么训练目标让文本 LLM 在少量数据下学会看图。
- 备选 NeurIPS：若贡献主要落在 OPD 目标、缩放规律和文本保持。
- 备选 ICLR：若贡献主要落在训练动力学，即预热长度以及 on-policy 与离线 KD 的差别。
- 会议选择等 M3 之后再定。结果低于第 6 节时不投主会。

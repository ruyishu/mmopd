# 数据

训练数据不进 git。下面的命令在仓库根目录执行，只下载标注和预热图像。OPD 不使用标注里的回答，读入时丢掉答案，只留图像和问题。评测集不放进训练。

子集由 `configs/*.yaml` 的 `seed` 决定。`sample_indices` 用同一种子打乱后取前缀，所以 1% 含在 5% 里，5% 含在 10% 里。

## 预热：LLaVA-Pretrain 558K

小配置抽 5 万对，主配置抽 10 万对，范围是 5万–10万。只训投影。

来源：[liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)。要用的是 `blip_laion_cc_sbu_558k.json` 和 `images.zip`。

```bash
uv run huggingface-cli download liuhaotian/LLaVA-Pretrain \
  blip_laion_cc_sbu_558k.json images.zip \
  --repo-type dataset \
  --local-dir data/llava-pretrain
unzip -q data/llava-pretrain/images.zip -d data/llava-pretrain
```

解压后图像应在 `data/llava-pretrain/images/`，与 json 里的相对路径一致。

## OPD 提示：LLaVA-Instruct-665K

标注在 [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) 的 `llava_v1_5_mix665k.json`。这个仓库没有图像。

```bash
uv run huggingface-cli download liuhaotian/LLaVA-Instruct-150K \
  llava_v1_5_mix665k.json \
  --repo-type dataset \
  --local-dir data/llava-instruct
```

图像按 [LLaVA 的 Visual Instruction Tuning 说明](https://github.com/haotian-liu/LLaVA) 下载，并保持 json 里的相对路径：COCO `train2017`、GQA、OCR-VQA、TextVQA、VisualGenome。这些是训练图，不是 MMBench / MMMU 等评测集。

## OPD 提示：ShareGPT4V

用监督微调那份混合标注，不用纯 caption 文件当答案。

来源：[Lin-Chen/ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)，文件 `sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json`。

```bash
uv run huggingface-cli download Lin-Chen/ShareGPT4V \
  sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
  --repo-type dataset \
  --local-dir data/sharegpt4v
```

与 LLaVA-Instruct 重叠的图像共用上面的图像根目录。其余路径按 [ShareGPT4V 仓库](https://github.com/ShareGPT4Omni/ShareGPT4V) 补齐。

## 可选 OCR 训练

只在预热对文字明显不够时加，不进主数据表。

- TextVQA 训练集：[facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa) 的 `train` split，或 [textvqa.org](https://textvqa.org/dataset)。放到 `data/textvqa/`。
- DocVQA 训练集：向 [docvqa.org](https://www.docvqa.org/datasets) 申请 train。放到 `data/docvqa/`。不要用 test。

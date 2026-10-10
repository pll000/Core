# CoRe: A Comprehensive Framework for Cross-Image Comparative Reasoning in Vision-Language Models

[English](README.md) · [论文](https://arxiv.org/abs/2607.12786) · [数据集](https://huggingface.co/datasets/3212367097pl-pllove/CoRe-20K) · [Apache-2.0 许可证](LICENSE)

本仓库提供论文实现，论文已被 **ACM Multimedia 2026（ACM MM 2026）** 接收。

**作者：** Lin Peng、Cong Wan、Zeyu Guo、SongLin Dong、Yihong Gong。

**通讯作者：** Cong Wan。

基于 Qwen3-VL-4B-Thinking 和 GRPO 的跨图像比较推理训练，支持人数、深度、距离和空间关系四类任务。

## 安装环境

使用独立的 Linux Python 环境，先安装与你的 CUDA 版本匹配的 PyTorch（>= 2.6）和 torchvision，再执行：

```bash
git clone https://github.com/pll000/Core.git
cd Core
python -m pip install -r requirements.txt
python -m pip install --no-deps -e third_party/ms-swift
```

## 准备数据

训练前请下载 [Hugging Face 上的 CoRe-20K 数据集](https://huggingface.co/datasets/3212367097pl-pllove/CoRe-20K)。完整压缩包约 299 MB，包含 60,000 个图片文件、训练和测试标注以及元数据。在 `Core` 仓库根目录执行：

```bash
curl -fL --retry 3 "https://huggingface.co/datasets/3212367097pl-pllove/CoRe-20K/resolve/main/CoRe-data.tar.gz?download=true" -o CoRe-data.tar.gz
tar -xzf CoRe-data.tar.gz --strip-components=1
```

解压后的文件位于 `data/` 下，图片目录如下：

```text
data/images/
├── remot_count_meta/
├── remot_depth_easy/
├── remot_distance/
└── remot_relation_easy/
```

## 开始训练

训练参数已在脚本中设置，默认使用 8 张 GPU：

```bash
bash scripts/train.sh
```

使用本地模型时，指定模型目录即可：

```bash
MODEL=/path/to/Qwen3-VL-4B-Thinking bash scripts/train.sh
```

模型检查点和训练日志保存在 `outputs/core-grpo/` 下。

## 评测

将模型路径替换为训练生成的检查点目录：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model /path/to/checkpoint \
  --task counting \
  --output outputs/counting_predictions.jsonl
```

`--task` 可选 `counting`、`depth`、`distance`、`relation`。评测结束后会同时保存预测结果和准确率统计。

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@inproceedings{peng2026core,
  title     = {{CoRe}: A Comprehensive Framework for Cross-Image Comparative Reasoning in Vision-Language Models},
  author    = {Peng, Lin and Wan, Cong and Guo, Zeyu and Dong, SongLin and Gong, Yihong},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.12786}
}
```

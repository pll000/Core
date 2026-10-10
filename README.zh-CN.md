# CoRe

[English](README.md) · [Apache-2.0 许可证](LICENSE)

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

仓库已包含 `data/` 下的训练和测试标注。图片不纳入 Git；训练前请将对应的 CoRe 图片文件夹放入以下目录：

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

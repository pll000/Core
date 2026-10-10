# CoRe

[中文说明](README.zh-CN.md) · [Apache-2.0 License](LICENSE)

Cross-image comparative reasoning with Qwen3-VL-4B-Thinking and GRPO, covering counting, depth, distance, and spatial relations.

## Installation

Use a separate Linux Python environment. Install PyTorch (>= 2.6) and a matching torchvision version for your CUDA setup, then run:

```bash
git clone https://github.com/pll000/Core.git
cd Core
python -m pip install -r requirements.txt
python -m pip install --no-deps -e third_party/ms-swift
```

## Data

Training and test annotations are included under `data/`. Image files are excluded from Git; place the corresponding CoRe image folders under `data/images/` before training:

```text
data/images/
├── remot_count_meta/
├── remot_depth_easy/
├── remot_distance/
└── remot_relation_easy/
```

## Training

The training settings are already defined in the launcher, which uses eight GPUs by default:

```bash
bash scripts/train.sh
```

To use a local model, provide its directory:

```bash
MODEL=/path/to/Qwen3-VL-4B-Thinking bash scripts/train.sh
```

Checkpoints and training logs are saved under `outputs/core-grpo/`.

## Evaluation

Replace the model path with a trained checkpoint directory:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model /path/to/checkpoint \
  --task counting \
  --output outputs/counting_predictions.jsonl
```

Available tasks: `counting`, `depth`, `distance`, and `relation`. Evaluation saves both predictions and accuracy statistics.

# CoRe: A Comprehensive Framework for Cross-Image Comparative Reasoning in Vision-Language Models

[中文说明](README.zh-CN.md) · [Paper](https://arxiv.org/abs/2607.12786) · [Dataset](https://huggingface.co/datasets/3212367097pl-pllove/CoRe-20K) · [Apache-2.0 License](LICENSE)

Code for our paper accepted at **ACM Multimedia 2026 (ACM MM 2026)**.

**Authors:** Lin Peng, Cong Wan, Zeyu Guo, SongLin Dong, and Yihong Gong.

**Corresponding author:** Cong Wan.

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

Download [CoRe-20K on Hugging Face](https://huggingface.co/datasets/3212367097pl-pllove/CoRe-20K) before training. The complete archive (299 MB) includes 60,000 image files, training and test annotations, and metadata. Run these commands from the `Core` repository root:

```bash
curl -fL --retry 3 "https://huggingface.co/datasets/3212367097pl-pllove/CoRe-20K/resolve/main/CoRe-data.tar.gz?download=true" -o CoRe-data.tar.gz
tar -xzf CoRe-data.tar.gz --strip-components=1
```

The archive extracts into `data/`, with image folders arranged as follows:

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

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{peng2026core,
  title     = {{CoRe}: A Comprehensive Framework for Cross-Image Comparative Reasoning in Vision-Language Models},
  author    = {Peng, Lin and Wan, Cong and Guo, Zeyu and Dong, SongLin and Gong, Yihong},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.12786}
}
```

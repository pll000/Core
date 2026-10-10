#!/usr/bin/env bash
# Recovered four-task GRPO configuration. Run DRY_RUN=1 first to inspect it.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
TRAIN_DATA=${TRAIN_DATA:-$DATA_ROOT/train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/core-grpo}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Thinking}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
export NPROC_PER_NODE=${NPROC_PER_NODE:-${#DEVICES[@]}}
export PYTHONPATH="$ROOT/third_party/ms-swift${PYTHONPATH:+:$PYTHONPATH}"
BATCH_SIZE=${BATCH_SIZE:-8}
if [[ ! "$BATCH_SIZE" =~ ^[0-9]+$ ]] || (( BATCH_SIZE < 4 || BATCH_SIZE % 4 != 0 )); then
    echo 'BATCH_SIZE must be a positive multiple of 4: the original reward groups four consecutive rollouts.' >&2
    exit 2
fi
if [[ ! "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > ${#DEVICES[@]} )); then
    echo 'NPROC_PER_NODE must be between 1 and the number of visible devices.' >&2
    exit 2
fi
# These settings preserve the reward batching contract and held-out test split.
for arg in "$@"; do
    case "$arg" in
        --num_generations*|--per_device_train_batch_size*|--per_device_eval_batch_size*|--dataset*|--val_dataset*|--remove_unused_columns*|--split_dataset_ratio*|--external_plugins*|--reward_funcs*|--reward_weights*|--generation_batch_size*|--steps_per_generation*)
            echo "Use the documented environment variables; overriding $arg breaks this recipe's data/reward contract." >&2
            exit 2 ;;
    esac
done
RUNTIME_DATA="$OUTPUT_DIR/runtime/train.jsonl"
CMD=("$PYTHON" -m swift.cli.main rlhf
    --rlhf_type grpo --model "$MODEL" --model_type qwen3_vl --template qwen3_vl
    --external_plugins "$ROOT/rewards/unified_reward_four_new.py"
    --reward_funcs unified_triplet_orm format cosine repetition --reward_weights 1 1 1 1
    --dataset "$RUNTIME_DATA" --remove_unused_columns false --split_dataset_ratio 0
    --load_from_cache_file false --strict true
    --train_type full --freeze_vit true --freeze_aligner true
    --torch_dtype bfloat16 --gradient_checkpointing true --padding_free false
    --max_completion_length 5000 --max_length 262144
    --num_train_epochs 1 --per_device_train_batch_size "$BATCH_SIZE"
    --per_device_eval_batch_size "$BATCH_SIZE" --gradient_accumulation_steps 4
    --learning_rate 1e-6 --lr_scheduler_type cosine --warmup_ratio 0.01
    --num_generations 4 --temperature 1.0 --top_p 0.9 --top_k 50
    --deepspeed zero3 --use_vllm false --num_iterations 1 --async_generate false
    --beta 0.001 --seed 42 --data_seed 42
    --save_strategy steps --save_steps 50 --save_total_limit 4
    --eval_strategy no --logging_steps 1 --log_completions true
    --dataloader_num_workers 4 --report_to tensorboard --output_dir "$OUTPUT_DIR"
    "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s NPROC_PER_NODE=%s\n' "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE"
    printf 'Source dataset: %s\nRuntime dataset: %s\n' "$TRAIN_DATA" "$RUNTIME_DATA"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
"$PYTHON" "$ROOT/scripts/check_environment.py"
"$PYTHON" "$ROOT/scripts/prepare_data.py" --input "$TRAIN_DATA" --data-root "$DATA_ROOT" --output "$RUNTIME_DATA"
exec "${CMD[@]}"

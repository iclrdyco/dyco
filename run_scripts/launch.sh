#!/usr/bin/env bash
# GRPO training launcher.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or a model repository ID.}"
: "${DATA_PATH:?Set DATA_PATH to your training JSONL file.}"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory containing the training images.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the directory for checkpoints.}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
SEED="${SEED:-42}"

for value in "$GPUS_PER_NODE" "$NNODES" "$PER_DEVICE_BS" "$GRAD_ACCUM" "$NUM_GENERATIONS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU, node, batch, accumulation, and generation counts must be positive integers." >&2
        exit 2
    fi
done
# A rollout batch is gathered across devices before gradient accumulation.
ROLLOUT_BATCH=$((NNODES * GPUS_PER_NODE * PER_DEVICE_BS))
if (( ROLLOUT_BATCH % NUM_GENERATIONS != 0 )); then
    echo "NNODES * GPUS_PER_NODE * PER_DEVICE_BS must be divisible by NUM_GENERATIONS." >&2
    exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src/open-r1-multimodal/src${PYTHONPATH:+:$PYTHONPATH}"
export DEBUG_MODE="${DEBUG_MODE:-false}"

command=(torchrun
    --nproc-per-node "$GPUS_PER_NODE" --nnodes "$NNODES" --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
    "${PROJECT_ROOT}/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py"
    --model_name_or_path "$MODEL_PATH"
    --data_file_paths "$DATA_PATH" --image_folders "$IMAGE_ROOT"
    --output_dir "$OUTPUT_DIR"
    --dataset_name unused --task_type general
    --is_reward_customized_from_vlm_module True
    --per_device_train_batch_size "$PER_DEVICE_BS"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --gradient_checkpointing True --bf16 True
    --num_train_epochs 1 --learning_rate 1e-6
    --seed "$SEED" --data_seed "$SEED"
    --num_generations "$NUM_GENERATIONS" --max_completion_length 2048
    --max_pixels 802816 --reward_funcs accuracy format
    --use_vllm False --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
    --dyco_enabled "${DYCO_ENABLED:-True}" --dyco_alpha "${DYCO_ALPHA:-0.2}"
    --dyco_tau "${DYCO_TAU:-0.05}" --dyco_centered "${DYCO_CENTERED:-False}"
    --logging_steps 10 --save_steps 1000 --report_to "${REPORT_TO:-none}"
    --deepspeed "${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/src/open-r1-multimodal/configs/zero3.json}"
    --beta 0.04 "$@")

# Inspect the command without loading dependencies, models, or data.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"

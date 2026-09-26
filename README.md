# DyCo-RL: GRPO Implementation

This repository provides GRPO and GRPO + DyCo-RL training for Qwen2.5-VL.

## Installation

Use Python 3.10 or newer with a compatible CUDA-enabled PyTorch 2.5/2.6 environment. The code targets Transformers 4.49.0 and uses FlashAttention 2 and DeepSpeed ZeRO-3.

```bash
INSTALL_FLASH_ATTN=1 bash setup.sh
```

## Training

```bash
export MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
export DATA_PATH="/path/to/train.jsonl"
export IMAGE_ROOT="/path/to/images"
export OUTPUT_DIR="/path/to/output"
bash run_scripts/run_grpo.sh
```

Set `DYCO_ENABLED=false` for the GRPO baseline. DyCo-RL defaults to `DYCO_ALPHA=0.2` and `DYCO_TAU=0.05`; `DYCO_CENTERED=true` selects the centered variant. Modality attention masses are min–max normalized within each response, and non-neutral tokens receive signed advantage reweighting.

The launcher defaults to 8 GPUs, a per-device batch size of 4, gradient accumulation of 2, and 4 rollouts per prompt. Override `GPUS_PER_NODE`, `PER_DEVICE_BS`, `GRAD_ACCUM`, `NUM_GENERATIONS`, and `SEED` as needed. Use `DRY_RUN=1` to inspect the command. Remote experiment tracking is disabled by default.

## Data format

Each JSONL record contains an image path and a question/reference-answer pair. Relative image paths are resolved against `IMAGE_ROOT`.

```json
{"image": "example.png", "conversations": [{"from": "human", "value": "<image>{question}"}, {"from": "gpt", "value": "{reference_answer}"}]}
```

Accuracy rewards use local answer verification; format rewards check the reasoning and answer tags.

## License

Apache-2.0. Third-party license headers and attribution are retained; see `THIRD_PARTY_NOTICES.md`.

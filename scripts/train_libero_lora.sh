#!/usr/bin/env bash
# Phase 4: LoRA fine-tuning on full LIBERO dataset
# LoRA r=64 with 10x higher LR (1e-3 vs 1e-4) per HF PEFT docs
# batch_size=2 (LoRA saves optimizer memory but not activations)
set -euo pipefail

cd "$(dirname "$0")/.."
set -a && source .env && set +a
export PATH="$(pwd)/.local/bin:$HOME/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Check Ollama isn't stealing VRAM
if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "ERROR: Ollama is running (~5.3GB VRAM). Stop it first: sudo systemctl stop ollama"
    exit 1
fi

uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --peft.method_type=LORA \
  --peft.r=64 \
  --policy.optimizer_lr=1e-3 \
  --policy.scheduler_decay_lr=1e-4 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.image_transforms.enable=true \
  --env.type=libero \
  --env.task=libero_object \
  --output_dir=./experiments/libero_lora \
  --steps=100000 \
  --batch_size=2 \
  --log_freq=100 \
  --save_freq=20000 \
  --eval_freq=20000 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --wandb.enable=true \
  --wandb.project=smolvla-libero \
  --seed=1000

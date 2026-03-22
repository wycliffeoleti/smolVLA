#!/usr/bin/env bash
# Phase 2 Step 1: Validation run (5k steps) — confirm model is learning
set -euo pipefail

cd "$(dirname "$0")/.."
set -a && source .env && set +a
export PATH="$(pwd)/.local/bin:$HOME/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniforge3/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/smol-libero \
  --dataset.image_transforms.enable=true \
  --env.type=libero \
  --env.task=libero_object \
  --output_dir=./experiments/libero_validation \
  --steps=5000 \
  --batch_size=2 \
  --log_freq=50 \
  --save_freq=2500 \
  --eval_freq=2500 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --wandb.enable=true \
  --wandb.project=smolvla-libero \
  --seed=1000

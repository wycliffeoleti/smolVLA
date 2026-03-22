#!/usr/bin/env bash
# Phase 3: Train SmolVLA on full LIBERO dataset (1,693 episodes, 40 tasks)
# Previous run used smol-libero (50 eps, 1 task) → 0% eval due to insufficient data
# This matches the published training setup more closely
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
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.image_transforms.enable=true \
  --env.type=libero \
  --env.task=libero_object \
  --output_dir=./experiments/libero_full_dataset \
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

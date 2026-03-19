# SmolVLA Manufacturing Pick-and-Place

## Stack
- **Framework**: LeRobot v0.5.0 (PyTorch, NOT JAX)
- **Model**: SmolVLA (450M params) — `lerobot/smolvla_base`
- **Simulation**: LIBERO via MuJoCo
- **Fine-tuning**: PEFT/LoRA
- **Dataset**: `HuggingFaceVLA/smol-libero` (50 episodes, 13K frames, v3.0 format)

## Hardware Constraints
- **GPU**: RTX 4060 8GB VRAM — batch_size=4 max for SmolVLA, use bfloat16
- **Storage**: Code on primary NVMe (~25GB free), data/weights/checkpoints on nvmedisk2 (~492GB free)
- **HF cache**: `/media/wolnxpc/nvmedisk2/smolVLA/hf_cache` (set via HF_HOME in .env)

## Critical Gotchas
1. ffmpeg 7.x with libsvtav1 required — installed via conda at `~/miniforge3/bin/ffmpeg`
2. `source .env` before any lerobot CLI commands (sets HF_HOME, MUJOCO_GL, PATH)
3. 50+ episodes minimum for SmolVLA — 25 episodes explicitly fails
4. Obs keys: `observation.state` (8-dim), `observation.images.image`, `observation.images.image2`, `action` (7-dim)
5. LIBERO training uses `--policy.load_vlm_weights=true` and `--batch_size=4`

## CLI Commands
```bash
# Always source .env first
source .env

# Eval (1 episode smoke test)
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_base \
  --env.type=libero --env.task=libero_object --env.task_ids="[0]" \
  --eval.batch_size=1 --eval.n_episodes=1

# Train (smoke test — 10 steps)
uv run lerobot-train \
  --policy.type=smolvla --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/smol-libero \
  --env.type=libero --env.task=libero_object \
  --output_dir=./experiments/smoke_test \
  --steps=10 --batch_size=4 --eval.batch_size=1 --eval.n_episodes=1 --eval_freq=10
```

## Project Layout
- `src/smolvla_manuf/` — custom code
- `scripts/` — training/eval scripts
- `configs/` — training configs
- `data/` → nvmedisk2 symlink
- `experiments/` → nvmedisk2 symlink
- `models/` → nvmedisk2 symlink

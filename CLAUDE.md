# SmolVLA Manufacturing Pick-and-Place

## Stack
- **Framework**: LeRobot v0.5.0 (PyTorch, NOT JAX)
- **Model**: SmolVLA (450M params) — `lerobot/smolvla_base`
- **Simulation**: LIBERO via MuJoCo
- **Fine-tuning**: PEFT/LoRA
- **Dataset**: `HuggingFaceVLA/libero` (1,693 episodes, 273K frames, 40 tasks, v3.0 format)

## Hardware Constraints
- **GPU**: RTX 4060 8GB VRAM — batch_size=2 max for SmolVLA (4 OOMs), use bfloat16
- **Storage**: Code on primary NVMe (~25GB free), data/weights/checkpoints on nvmedisk2 (~492GB free)
- **HF cache**: `/media/wolnxpc/nvmedisk2/smolVLA/hf_cache` (set via HF_HOME in .env)

## Critical Gotchas
1. ffmpeg 7.x with libsvtav1 required — installed via conda at `~/miniforge3/bin/ffmpeg`
2. `set -a && source .env && set +a` before any lerobot CLI commands — plain `source .env` doesn't export vars to child processes (HF_HOME, MUJOCO_GL, PATH to conda bins)
3. cmake wrapper at `.local/bin/cmake` injects `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (needed for egl-probe build with cmake 4.x)
4. EGL dev headers installed via conda (`libegl-devel`) at `~/miniforge3/include/EGL/`
5. 50+ episodes minimum for SmolVLA — 25 episodes explicitly fails
6. Obs keys: `observation.state` (8-dim), `observation.images.image`, `observation.images.image2`, `action` (7-dim)
7. LIBERO training uses `--policy.load_vlm_weights=true` and `--batch_size=2` (4 causes OOM on 8GB)
8. Dataset was v2.1 on HF Hub — converted locally to v3.0 via `lerobot.datasets.v30.convert_dataset_v21_to_v30`
9. Eval requires `--rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'` — LIBERO uses `image`/`image2`, SmolVLA expects `camera1`/`camera2`/`camera3`
10. Training requires `--policy.push_to_hub=false` unless `--policy.repo_id` is set
11. `log_freq` defaults to 200 — set `--log_freq=1` for smoke tests to see loss values
12. LeRobot refuses to train if `output_dir` already exists and `resume=False` — don't pre-create it
13. Stop Ollama (`sudo systemctl stop ollama`) before training — it uses ~5.3GB VRAM
14. `--env.task=libero_10` (all 40 tasks) OOMs on 8GB — use `libero_object` (10 tasks) for eval
15. `smol-libero` (50 eps, 1 task) gives 0% eval — need full `HuggingFaceVLA/libero` (1,693 eps)

## CLI Commands
```bash
# Always export .env vars (plain source doesn't export)
set -a && source .env && set +a
export PATH=.local/bin:~/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=~/miniforge3/lib:${LD_LIBRARY_PATH:-}

# Eval (1 episode smoke test)
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_base \
  --env.type=libero --env.task=libero_object --env.task_ids="[0]" \
  --eval.batch_size=1 --eval.n_episodes=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

# Train (smoke test — 10 steps)
uv run lerobot-train \
  --policy.type=smolvla --policy.load_vlm_weights=true --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/smol-libero \
  --env.type=libero --env.task=libero_object \
  --output_dir=./experiments/smoke_test \
  --steps=10 --batch_size=2 --eval.batch_size=1 --eval.n_episodes=1 --eval_freq=10
```

## Training Results

### Phase 2: smol-libero (50 episodes, 1 task)
- **50k steps**: Loss 1.27 → 0.09 (93% reduction) but **0% eval success**
- **Root cause**: 50 episodes of 1 task insufficient — model learns action patterns but can't complete pick-and-place
- **Checkpoint**: `experiments/libero_full/checkpoints/050000/pretrained_model`

### Phase 3: Full LIBERO (1,693 episodes, 40 tasks) — BEST
- **Dataset**: `HuggingFaceVLA/libero` (33GB, 273K frames)
- **100k steps**: Loss 1.04 → 0.10, **20% eval success rate** (best at 60k checkpoint)
- **Speed**: ~9 steps/s (0.110s/step), 3h 26min total
- **VRAM**: 7.6 GB stable (batch_size=2, bfloat16)
- **Best checkpoint**: `experiments/libero_full_dataset/checkpoints/060000/pretrained_model`
- **Eval progression**: 8% (20k) → 8% (40k) → **20% (60k)** → 18% (80k) → 18% (100k)
- **Wandb**: `smolvla-libero` project
- **Published baseline**: 96% with batch_size=64 on 8xH100 — our 20% with batch_size=2 on RTX 4060 is expected
- **Learnable params**: 100M of 450M (train_expert_only=True, no LoRA)

## Project Layout
- `src/smolvla_manuf/` — custom code
- `scripts/` — training/eval scripts
- `configs/` — training configs
- `data/` → nvmedisk2 symlink
- `experiments/` → nvmedisk2 symlink
- `models/` → nvmedisk2 symlink

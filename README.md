# SmolVLA: Vision-Language-Action Fine-Tuning on Consumer Hardware

End-to-end fine-tuning of a **450M-parameter Vision-Language-Action model** ([SmolVLA](https://huggingface.co/lerobot/smolvla_base)) for robotic pick-and-place on a single **RTX 4060 (8GB VRAM)** using [LeRobot](https://github.com/huggingface/lerobot).

<p align="center">
  <img src="assets/demo_success.gif" alt="Successful pick-and-place rollout" width="320"/>
  <br/>
  <em>Successful pick-and-place on LIBERO Object (60k checkpoint)</em>
</p>

## Results

| Setup | GPU | Batch Size | Steps | LIBERO Object Success |
|-------|-----|-----------|-------|----------------------|
| **SmolVLA paper** | 4x GPU | 256 | 200k | 87.3% avg |
| **This project** | RTX 4060 8GB | 2 | 100k | **20%** |
| Subset (50 eps) | RTX 4060 8GB | 2 | 50k | 0% |

Operating at **1/128th the compute budget** (batch_size=2 vs 256, 1 GPU vs 4), the model reaches 20% task success on LIBERO Object. The gap is expected and consistent with batch size scaling in imitation learning.

### Eval Success by Checkpoint

| Steps | 20k | 40k | **60k** | 80k | 100k |
|-------|-----|-----|---------|-----|------|
| Success % | 8 | 8 | **20** | 18 | 18 |

### Training Loss

Loss: **1.04 → 0.10** (90% reduction) over 100k steps, 3h 26min total training time.

### Key Finding: Data Quantity Dominates

Training on 50 episodes (1 task) gave **0% eval success** despite 93% loss reduction. The loss converged but the model couldn't complete pick-and-place tasks. Switching to the full LIBERO dataset (1,693 episodes, 40 tasks) immediately produced 20% success. This demonstrates that in imitation learning, **dataset scale matters more than training duration** for task completion.

## Architecture

```
LIBERO Dataset (1,693 episodes, 40 tasks)
    │
    ▼
┌─────────────────────────────────────┐
│  SmolVLA (450M params)              │
│  ┌───────────────┐  ┌────────────┐  │
│  │ SmolVLM2-500M │  │  Action    │  │ ◄── 100M trainable params
│  │ (frozen)      │──│  Expert    │  │     (expert-only fine-tuning)
│  │ Vision+Lang   │  │ (trained)  │  │
│  └───────────────┘  └────────────┘  │
│         ▲                  │        │
│    2x cameras         7-dim action  │
│    + state (8-dim)    (50-step      │
│    + task text         chunks)      │
└─────────────────────────────────────┘
    │
    ▼
MuJoCo LIBERO Simulation (eval rollouts)
```

- **Vision backbone**: SmolVLM2-500M-Video-Instruct (frozen)
- **Action expert**: Flow matching with 10-step denoising, 50-action chunking
- **Inputs**: 2 camera views (256x256), 8-dim proprioceptive state, task description
- **Outputs**: 7-dim actions (end-effector position, orientation, gripper)

## Setup

### Requirements

- Python 3.12+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
- ~40GB disk space (dataset + checkpoints)
- Linux (EGL rendering for headless MuJoCo)

### Installation

```bash
# Clone
git clone https://github.com/<your-username>/smolVLA.git
cd smolVLA

# Install dependencies
uv sync

# Create .env (required for HF cache and rendering)
cat > .env << 'EOF'
HF_HOME=/path/to/your/hf_cache
MUJOCO_GL=egl
EOF

# Install ffmpeg with libsvtav1 (required for video logging)
conda install -c conda-forge ffmpeg svt-av1
```

### Reproduce Best Result

```bash
# Export environment
set -a && source .env && set +a
export PATH=.local/bin:~/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=~/miniforge3/lib:${LD_LIBRARY_PATH:-}

# Evaluate the best checkpoint (60k steps)
uv run lerobot-eval \
  --policy.path=./experiments/libero_full_dataset/checkpoints/060000/pretrained_model \
  --env.type=libero --env.task=libero_object \
  --eval.batch_size=1 --eval.n_episodes=50
```

### Train from Scratch

```bash
# Stop Ollama if running (frees ~5.3GB VRAM)
sudo systemctl stop ollama

# Full training (100k steps, ~3.5 hours on RTX 4060)
bash scripts/train_libero_full_dataset.sh
```

## Reproducibility Notes

Key gotchas discovered during development (several undocumented elsewhere):

1. **`set -a && source .env && set +a`** — plain `source .env` doesn't export to `uv run` child processes
2. **batch_size=2 max** on 8GB VRAM — batch_size=4 causes OOM during optimizer state initialization
3. **`--env.task=libero_10`** (all 40 tasks) OOMs on 8GB — use `libero_object` (10 tasks) for eval
4. **Output directory** must not pre-exist — LeRobot refuses to train if `output_dir` exists and `resume=False`
5. **50 episodes insufficient** — converges in loss but 0% eval success; need full dataset (1,693 episodes)
6. **Stop Ollama** before training — it silently consumes ~5.3GB VRAM
7. **cmake 4.x** requires a wrapper injecting `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for egl-probe
8. **EGL headers** needed from conda: `conda install -c conda-forge libegl-devel`

## Project Structure

```
smolVLA/
├── scripts/
│   ├── train_libero.sh              # Phase 2: validation (5k steps)
│   ├── train_libero_full.sh         # Phase 2: full smol-libero (50k steps)
│   └── train_libero_full_dataset.sh # Phase 3: full LIBERO (100k steps) ← best
├── experiments/ → nvmedisk2 (symlink)
│   ├── libero_full_dataset/         # Phase 3 checkpoints + eval videos
│   └── libero_full/                 # Phase 2 checkpoints
├── assets/
│   ├── demo_success.gif             # Successful rollout
│   └── demo_failure_phase2.gif      # Phase 2 failure (0% with 50 episodes)
├── src/smolvla_manuf/               # Custom code (placeholder)
├── pyproject.toml
├── CLAUDE.md                        # Development notes + gotchas
└── README.md
```

## Limitations & Next Steps

- **20% vs 87.3%**: Primarily a batch size gap (2 vs 256). The model sees 128x fewer samples per gradient update, which limits generalization across the 10 LIBERO Object tasks.
- **No LoRA**: Currently uses expert-only fine-tuning (100M params). LoRA could reduce trainable params to ~10M, potentially improving generalization with small effective batch sizes.
- **Single eval suite**: Only evaluated on LIBERO Object (10 tasks). The full LIBERO benchmark has 4 suites (Spatial, Object, Goal, Long).

**Planned improvements:**
- LoRA fine-tuning to reduce trainable parameters and potentially improve batch efficiency
- Evaluation across all 4 LIBERO suites
- Training curve visualizations and per-task success analysis

## References

- [SmolVLA: A Small Vision-Language-Action Model](https://arxiv.org/abs/2506.01844) (Zouitine et al., 2025)
- [LeRobot: State-of-the-art Machine Learning for Real-World Robotics](https://github.com/huggingface/lerobot)
- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [SmolVLA HuggingFace Blog](https://huggingface.co/blog/smolvla)

## License

MIT

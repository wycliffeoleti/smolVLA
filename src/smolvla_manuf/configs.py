"""Typed training configurations for SmolVLA fine-tuning experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass

# Camera rename map: LIBERO key names → SmolVLA expected key names
_RENAME_MAP = json.dumps(
    {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
)


@dataclass
class TrainingConfig:
    """Typed configuration for a lerobot-train invocation.

    Use the classmethods for preset experiment configs:
        TrainingConfig.lora_r64()
        TrainingConfig.expert_full()
        TrainingConfig.smol_libero()
        TrainingConfig.validation()
    """

    dataset_repo_id: str
    output_dir: str
    steps: int
    batch_size: int = 2
    eval_batch_size: int = 1
    eval_n_episodes: int = 5
    eval_freq: int = 20_000
    save_freq: int = 20_000
    log_freq: int = 100
    seed: int = 1000
    wandb_project: str = "smolvla-libero"
    image_transforms: bool = True
    # Policy mode
    policy_type: str | None = "smolvla"
    policy_path: str | None = None
    load_vlm_weights: bool = True
    empty_cameras: int | None = None
    rename_cameras: bool = False
    # PEFT (LoRA)
    peft_method: str | None = None
    peft_r: int | None = None
    # Learning rate overrides
    learning_rate: float | None = None
    scheduler_decay_lr: float | None = None

    def to_cli_args(self) -> list[str]:
        """Generate lerobot-train CLI arguments from this config.

        Returns:
            List of ``--flag=value`` strings ready to pass to subprocess.
        """
        args: list[str] = []

        # Policy identity: path-based (LoRA) or type-based (expert)
        if self.policy_path:
            args.append(f"--policy.path={self.policy_path}")
        elif self.policy_type:
            args.append(f"--policy.type={self.policy_type}")
            if self.load_vlm_weights:
                args.append("--policy.load_vlm_weights=true")

        args.append("--policy.push_to_hub=false")

        if self.empty_cameras is not None:
            args.append(f"--policy.empty_cameras={self.empty_cameras}")

        if self.rename_cameras:
            args.append(f"--rename_map={_RENAME_MAP}")

        # PEFT
        if self.peft_method:
            args.append(f"--peft.method_type={self.peft_method}")
            if self.peft_r is not None:
                args.append(f"--peft.r={self.peft_r}")

        # Learning rate overrides
        if self.learning_rate is not None:
            args.append(f"--policy.optimizer_lr={self.learning_rate}")
        if self.scheduler_decay_lr is not None:
            args.append(f"--policy.scheduler_decay_lr={self.scheduler_decay_lr}")

        # Dataset + env (always present)
        args += [
            f"--dataset.repo_id={self.dataset_repo_id}",
            f"--dataset.image_transforms.enable={'true' if self.image_transforms else 'false'}",
            "--env.type=libero",
            "--env.task=libero_object",
            f"--output_dir={self.output_dir}",
            f"--steps={self.steps}",
            f"--batch_size={self.batch_size}",
            f"--log_freq={self.log_freq}",
            f"--save_freq={self.save_freq}",
            f"--eval_freq={self.eval_freq}",
            f"--eval.batch_size={self.eval_batch_size}",
            f"--eval.n_episodes={self.eval_n_episodes}",
            "--wandb.enable=true",
            f"--wandb.project={self.wandb_project}",
            f"--seed={self.seed}",
        ]

        return args

    @classmethod
    def validation(cls) -> TrainingConfig:
        """5k-step smoke test on smol-libero."""
        return cls(
            dataset_repo_id="HuggingFaceVLA/smol-libero",
            output_dir="./experiments/libero_validation",
            steps=5_000,
            eval_freq=2_500,
            save_freq=2_500,
        )

    @classmethod
    def smol_libero(cls) -> TrainingConfig:
        """50k-step expert-only training on smol-libero (50 episodes)."""
        return cls(
            dataset_repo_id="HuggingFaceVLA/smol-libero",
            output_dir="./experiments/libero_full",
            steps=50_000,
        )

    @classmethod
    def expert_full(cls) -> TrainingConfig:
        """100k-step expert-only training on full LIBERO (1,693 episodes)."""
        return cls(
            dataset_repo_id="HuggingFaceVLA/libero",
            output_dir="./experiments/libero_full_dataset",
            steps=100_000,
        )

    @classmethod
    def lora_r64(cls) -> TrainingConfig:
        """100k-step LoRA r=64 on full LIBERO — best result (32% success)."""
        return cls(
            dataset_repo_id="HuggingFaceVLA/libero",
            output_dir="./experiments/libero_lora",
            steps=100_000,
            policy_type=None,
            policy_path="lerobot/smolvla_base",
            load_vlm_weights=False,
            empty_cameras=1,
            rename_cameras=True,
            peft_method="LORA",
            peft_r=64,
            learning_rate=1e-3,
            scheduler_decay_lr=1e-4,
        )

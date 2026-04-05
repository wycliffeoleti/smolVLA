"""Tests for TrainingConfig and training launcher."""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import MagicMock, patch

import pytest

from smolvla_manuf.configs import TrainingConfig
from smolvla_manuf.train import check_ollama, run_training

# ---------------------------------------------------------------------------
# TrainingConfig — instantiation
# ---------------------------------------------------------------------------


def test_training_config_requires_dataset_and_output_and_steps() -> None:
    cfg = TrainingConfig(
        dataset_repo_id="HuggingFaceVLA/libero",
        output_dir="./experiments/test",
        steps=1000,
    )
    assert cfg.dataset_repo_id == "HuggingFaceVLA/libero"
    assert cfg.steps == 1000


def test_training_config_defaults_to_batch_size_2() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    assert cfg.batch_size == 2


def test_training_config_defaults_to_no_peft() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    assert cfg.peft_method is None
    assert cfg.peft_r is None


# ---------------------------------------------------------------------------
# to_cli_args — always-present flags
# ---------------------------------------------------------------------------


def test_cli_args_includes_dataset_repo_id() -> None:
    cfg = TrainingConfig(dataset_repo_id="HuggingFaceVLA/libero", output_dir="./out", steps=100)
    assert "--dataset.repo_id=HuggingFaceVLA/libero" in cfg.to_cli_args()


def test_cli_args_includes_steps() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="./out", steps=50000)
    assert "--steps=50000" in cfg.to_cli_args()


def test_cli_args_includes_batch_size() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="./out", steps=100)
    assert "--batch_size=2" in cfg.to_cli_args()


def test_cli_args_includes_output_dir() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="./experiments/lora", steps=100)
    assert "--output_dir=./experiments/lora" in cfg.to_cli_args()


def test_cli_args_includes_wandb_project() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    assert "--wandb.project=smolvla-libero" in cfg.to_cli_args()


def test_cli_args_returns_list_of_strings() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    args = cfg.to_cli_args()
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)


# ---------------------------------------------------------------------------
# to_cli_args — expert-only mode (no PEFT)
# ---------------------------------------------------------------------------


def test_cli_args_expert_includes_policy_type_smolvla() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    assert "--policy.type=smolvla" in cfg.to_cli_args()


def test_cli_args_expert_includes_load_vlm_weights() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    assert "--policy.load_vlm_weights=true" in cfg.to_cli_args()


def test_cli_args_expert_excludes_peft_flags() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    args = cfg.to_cli_args()
    assert not any("peft" in a for a in args)


def test_cli_args_expert_excludes_rename_map() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    args = cfg.to_cli_args()
    assert not any("rename_map" in a for a in args)


# ---------------------------------------------------------------------------
# to_cli_args — LoRA mode
# ---------------------------------------------------------------------------


def test_cli_args_lora_includes_peft_method() -> None:
    cfg = TrainingConfig(
        dataset_repo_id="x",
        output_dir="y",
        steps=100,
        policy_type=None,
        policy_path="lerobot/smolvla_base",
        peft_method="LORA",
        peft_r=64,
    )
    assert "--peft.method_type=LORA" in cfg.to_cli_args()


def test_cli_args_lora_includes_peft_r() -> None:
    cfg = TrainingConfig(
        dataset_repo_id="x",
        output_dir="y",
        steps=100,
        policy_type=None,
        policy_path="lerobot/smolvla_base",
        peft_method="LORA",
        peft_r=64,
    )
    assert "--peft.r=64" in cfg.to_cli_args()


def test_cli_args_lora_includes_policy_path() -> None:
    cfg = TrainingConfig(
        dataset_repo_id="x",
        output_dir="y",
        steps=100,
        policy_type=None,
        policy_path="lerobot/smolvla_base",
        peft_method="LORA",
        peft_r=64,
    )
    assert "--policy.path=lerobot/smolvla_base" in cfg.to_cli_args()


def test_cli_args_lora_excludes_policy_type() -> None:
    cfg = TrainingConfig(
        dataset_repo_id="x",
        output_dir="y",
        steps=100,
        policy_type=None,
        policy_path="lerobot/smolvla_base",
        peft_method="LORA",
        peft_r=64,
    )
    args = cfg.to_cli_args()
    assert not any(a.startswith("--policy.type=") for a in args)


def test_cli_args_lora_includes_rename_map() -> None:
    cfg = TrainingConfig(
        dataset_repo_id="x",
        output_dir="y",
        steps=100,
        policy_type=None,
        policy_path="lerobot/smolvla_base",
        rename_cameras=True,
        peft_method="LORA",
        peft_r=64,
    )
    args = cfg.to_cli_args()
    assert any("rename_map" in a for a in args)
    assert any("camera1" in a for a in args)


def test_cli_args_lora_includes_learning_rate() -> None:
    cfg = TrainingConfig(
        dataset_repo_id="x",
        output_dir="y",
        steps=100,
        policy_type=None,
        policy_path="lerobot/smolvla_base",
        peft_method="LORA",
        peft_r=64,
        learning_rate=1e-3,
    )
    assert "--policy.optimizer_lr=0.001" in cfg.to_cli_args()


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_training_config_is_json_serializable() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    data = json.dumps(dataclasses.asdict(cfg))
    assert json.loads(data)["dataset_repo_id"] == "x"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_preset_lora_r64_has_correct_peft_config() -> None:
    cfg = TrainingConfig.lora_r64()
    assert cfg.peft_method == "LORA"
    assert cfg.peft_r == 64
    assert cfg.learning_rate == pytest.approx(1e-3)


def test_preset_lora_r64_uses_full_libero_dataset() -> None:
    cfg = TrainingConfig.lora_r64()
    assert cfg.dataset_repo_id == "HuggingFaceVLA/libero"
    assert cfg.steps == 100_000


def test_preset_expert_full_uses_full_libero_dataset() -> None:
    cfg = TrainingConfig.expert_full()
    assert cfg.dataset_repo_id == "HuggingFaceVLA/libero"
    assert cfg.steps == 100_000
    assert cfg.peft_method is None


def test_preset_validation_uses_smol_libero_with_fewer_steps() -> None:
    cfg = TrainingConfig.validation()
    assert cfg.dataset_repo_id == "HuggingFaceVLA/smol-libero"
    assert cfg.steps < 10_000


def test_preset_smol_libero_uses_smol_dataset() -> None:
    cfg = TrainingConfig.smol_libero()
    assert cfg.dataset_repo_id == "HuggingFaceVLA/smol-libero"
    assert cfg.steps == 50_000


# ---------------------------------------------------------------------------
# check_ollama
# ---------------------------------------------------------------------------


def test_check_ollama_returns_false_when_not_running() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1  # systemctl returns 1 when service is inactive
    with patch("smolvla_manuf.train.subprocess.run", return_value=mock_result):
        assert check_ollama() is False


def test_check_ollama_returns_true_when_running() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0  # systemctl returns 0 when service is active
    with patch("smolvla_manuf.train.subprocess.run", return_value=mock_result):
        assert check_ollama() is True


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------


def test_run_training_raises_when_ollama_is_running() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100)
    with (
        patch("smolvla_manuf.train.check_ollama", return_value=True),
        pytest.raises(RuntimeError, match="Ollama"),
    ):
        run_training(cfg)


def test_run_training_calls_lerobot_train_command() -> None:
    cfg = TrainingConfig(dataset_repo_id="HuggingFaceVLA/libero", output_dir="./out", steps=100)
    with (
        patch("smolvla_manuf.train.check_ollama", return_value=False),
        patch("smolvla_manuf.train.subprocess.run") as mock_run,
    ):
        run_training(cfg)

    called_cmd = mock_run.call_args[0][0]
    assert called_cmd[:3] == ["uv", "run", "lerobot-train"]
    assert "--dataset.repo_id=HuggingFaceVLA/libero" in called_cmd
    assert "--steps=100" in called_cmd


def test_cli_args_respects_image_transforms_false() -> None:
    cfg = TrainingConfig(dataset_repo_id="x", output_dir="y", steps=100, image_transforms=False)
    assert "--dataset.image_transforms.enable=false" in cfg.to_cli_args()


def test_check_ollama_returns_false_when_systemctl_not_found() -> None:
    with patch("smolvla_manuf.train.subprocess.run", side_effect=FileNotFoundError):
        assert check_ollama() is False

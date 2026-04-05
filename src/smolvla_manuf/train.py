"""Training launcher with pre-flight checks for SmolVLA experiments."""

from __future__ import annotations

import logging
import shlex
import subprocess

from smolvla_manuf.configs import TrainingConfig

log = logging.getLogger(__name__)

_PRESETS = {
    "validation": TrainingConfig.validation,
    "smol_libero": TrainingConfig.smol_libero,
    "expert_full": TrainingConfig.expert_full,
    "lora_r64": TrainingConfig.lora_r64,
}


def check_ollama() -> bool:
    """Return True if the Ollama systemd service is currently active.

    Returns:
        True if Ollama is running (consuming ~5.3GB VRAM), False otherwise.
    """
    result = subprocess.run(
        ["systemctl", "is-active", "--quiet", "ollama"],
        capture_output=True,
    )
    return result.returncode == 0


def run_training(config: TrainingConfig) -> None:
    """Run lerobot-train with pre-flight VRAM checks.

    Args:
        config: Training configuration to execute.

    Raises:
        RuntimeError: If Ollama is active and consuming VRAM.
    """
    if check_ollama():
        raise RuntimeError(
            "Ollama is running (~5.3GB VRAM). Stop it first: sudo systemctl stop ollama"
        )

    cmd = ["uv", "run", "lerobot-train"] + config.to_cli_args()
    log.info("Running: %s", shlex.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """CLI entry point — select a preset config and launch training.

    Usage:
        smolvla-train --config lora_r64
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="SmolVLA training launcher")
    parser.add_argument(
        "--config",
        choices=list(_PRESETS),
        required=True,
        help="Preset training configuration to use",
    )
    args = parser.parse_args()
    run_training(_PRESETS[args.config]())

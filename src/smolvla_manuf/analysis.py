"""WandB experiment analysis — fetch runs, generate plots, export metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import wandb

matplotlib.use("Agg")  # non-interactive backend for file output

log = logging.getLogger(__name__)

# Dataset display names for plots
_DATASET_LABELS: dict[str, str] = {
    "HuggingFaceVLA/smol-libero": "smol-libero\n(50 eps)",
    "HuggingFaceVLA/libero": "LIBERO full\n(1,693 eps)",
}

# Method display names and colours
_METHOD_STYLE: dict[str, dict[str, str]] = {
    "expert": {"label": "Expert-only (100M params)", "color": "#4878CF"},
    "lora": {"label": "LoRA r=64 (3M params)", "color": "#D65F5F"},
}


@dataclass
class EvalCheckpoint:
    """Eval success rate at a training checkpoint."""

    step: int
    success_rate: float


@dataclass
class LossPoint:
    """Training loss at a given step."""

    step: int
    loss: float


@dataclass
class RunData:
    """Parsed representation of a single WandB training run."""

    id: str
    method: str  # "lora" | "expert"
    dataset: str
    steps: int
    final_eval: float
    eval_checkpoints: list[EvalCheckpoint]
    loss_history: list[LossPoint]


# ---------------------------------------------------------------------------
# WandB fetching
# ---------------------------------------------------------------------------


def _parse_run(run: Any) -> RunData:  # wandb.Run is untyped; Any avoids spurious mypy errors
    """Parse a single WandB Run object into RunData."""
    peft = run.config.get("peft")
    method = "lora" if peft and peft.get("method_type") == "LORA" else "expert"
    dataset: str = run.config.get("dataset", {}).get("repo_id", "unknown")
    steps: int = int(run.config.get("steps", 0))
    final_eval: float = float(run.summary.get("eval/pc_success", 0.0))

    history: pd.DataFrame = run.history(samples=10_000)

    _has_steps = "train/steps" in history.columns
    eval_rows = (
        history[history["eval/pc_success"].notna()]
        if "eval/pc_success" in history.columns and _has_steps
        else pd.DataFrame()
    )
    eval_checkpoints = [
        EvalCheckpoint(step=int(row["train/steps"]), success_rate=float(row["eval/pc_success"]))
        for _, row in eval_rows.iterrows()
    ]

    loss_rows = (
        history[history["train/loss"].notna()]
        if "train/loss" in history.columns and _has_steps
        else pd.DataFrame()
    )
    loss_history = [
        LossPoint(step=int(row["train/steps"]), loss=float(row["train/loss"]))
        for _, row in loss_rows.iterrows()
    ]

    return RunData(
        id=run.id,
        method=method,
        dataset=dataset,
        steps=steps,
        final_eval=final_eval,
        eval_checkpoints=eval_checkpoints,
        loss_history=loss_history,
    )


def fetch_runs(project: str) -> list[RunData]:
    """Fetch and parse all finished runs from a WandB project.

    Args:
        project: WandB project name (e.g. "smolvla-libero").

    Returns:
        List of parsed RunData, filtered to finished runs only.
    """
    api = wandb.Api()
    runs = api.runs(project)
    finished = [r for r in runs if r.state == "finished"]
    log.info("Fetched %d finished runs from %s", len(finished), project)
    return [_parse_run(r) for r in finished]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_metrics(runs: list[RunData], output_path: Path) -> None:
    """Export structured training results to a JSON file.

    Args:
        runs: Parsed run data to export.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [
            {
                "id": r.id,
                "method": r.method,
                "dataset": r.dataset,
                "steps": r.steps,
                "final_eval": r.final_eval,
                "eval_checkpoints": [asdict(c) for c in r.eval_checkpoints],
            }
            for r in runs
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2))
    log.info("Metrics exported to %s", output_path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save_fig(fig: matplotlib.figure.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_path)


def plot_loss_curves(runs: list[RunData], output_path: Path) -> None:
    """Plot training loss curves with 100-step rolling average.

    Overlays LoRA vs Expert for the full-LIBERO 100k-step runs.

    Args:
        runs: Parsed run data.
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    full_runs = [r for r in runs if r.dataset == "HuggingFaceVLA/libero" and r.steps >= 100_000]
    for run in full_runs:
        if not run.loss_history:
            continue
        style = _METHOD_STYLE.get(run.method, {"label": run.method, "color": "gray"})
        steps = [p.step for p in run.loss_history]
        losses = pd.Series([p.loss for p in run.loss_history]).rolling(100, min_periods=1).mean()
        ax.plot(
            steps, losses, label=style["label"], color=style["color"], linewidth=1.8, alpha=0.9
        )

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Loss (100-step rolling avg)", fontsize=12)
    ax.set_title("Training Loss: LoRA vs Expert-only (full LIBERO, RTX 4060)", fontsize=13, pad=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    _save_fig(fig, output_path)


def plot_eval_progression(runs: list[RunData], output_path: Path) -> None:
    """Plot eval success rate at each checkpoint for LoRA vs Expert.

    Args:
        runs: Parsed run data.
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    full_runs = [r for r in runs if r.dataset == "HuggingFaceVLA/libero" and r.steps >= 100_000]
    for run in full_runs:
        if not run.eval_checkpoints:
            continue
        style = _METHOD_STYLE.get(run.method, {"label": run.method, "color": "gray"})
        steps = [c.step // 1000 for c in run.eval_checkpoints]
        success = [c.success_rate for c in run.eval_checkpoints]
        ax.plot(
            steps,
            success,
            marker="o",
            label=style["label"],
            color=style["color"],
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Training step (×1k)", fontsize=12)
    ax.set_ylabel("LIBERO Object success rate (%)", fontsize=12)
    ax.set_title("Eval Success Rate by Checkpoint (full LIBERO, RTX 4060)", fontsize=13, pad=12)
    ax.set_ylim(0, 40)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    _save_fig(fig, output_path)


def plot_data_scaling(runs: list[RunData], output_path: Path) -> None:
    """Plot final eval success rate by dataset size.

    Args:
        runs: Parsed run data.
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    # Group by dataset × method, keep the run with the most steps
    grouped: dict[tuple[str, str], RunData] = {}
    for run in runs:
        key = (run.dataset, run.method)
        if key not in grouped or run.steps > grouped[key].steps:
            grouped[key] = run

    labels = [
        _DATASET_LABELS.get(r.dataset, r.dataset) + f"\n{r.method}" for r in grouped.values()
    ]
    values = [r.final_eval for r in grouped.values()]
    colors = [_METHOD_STYLE.get(r.method, {"color": "gray"})["color"] for r in grouped.values()]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.0f}%",
            ha="center",
            fontsize=11,
        )

    ax.set_ylabel("Final eval success rate (%)", fontsize=12)
    ax.set_title("Data Scaling: Dataset Size × Fine-tuning Method", fontsize=13, pad=12)
    if not values:
        _save_fig(fig, output_path)
        return
    ax.set_ylim(0, max(values) * 1.25 + 5)
    ax.grid(True, axis="y", alpha=0.4)
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_WANDB_PROJECT = "smolvla-libero"
_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
_RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def main() -> None:
    """CLI entry point — fetch runs, export metrics, generate all plots."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.info("Fetching runs from WandB project: %s", _WANDB_PROJECT)
    runs = fetch_runs(_WANDB_PROJECT)

    export_metrics(runs, _RESULTS_DIR / "metrics.json")

    plot_loss_curves(runs, _ASSETS_DIR / "loss_curves.png")
    plot_eval_progression(runs, _ASSETS_DIR / "eval_progression.png")
    plot_data_scaling(runs, _ASSETS_DIR / "data_scaling.png")

    log.info("Done. Plots saved to %s", _ASSETS_DIR)


__all__ = [
    "EvalCheckpoint",
    "LossPoint",
    "RunData",
    "export_metrics",
    "fetch_runs",
    "plot_data_scaling",
    "plot_eval_progression",
    "plot_loss_curves",
]

"""Tests for smolvla_manuf.analysis — WandB metric extraction and plot generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd
import pytest

from smolvla_manuf.analysis import (
    EvalCheckpoint,
    LossPoint,
    RunData,
    export_metrics,
    fetch_runs,
    plot_data_scaling,
    plot_eval_progression,
    plot_loss_curves,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_run(
    run_id: str,
    state: str = "finished",
    method: str | None = None,  # None → expert, 'LORA' → lora
    dataset: str = "HuggingFaceVLA/libero",
    steps: int = 100_000,
    final_eval: float = 20.0,
    eval_rows: list[tuple[int, float]] | None = None,
    loss_rows: list[tuple[int, float]] | None = None,
) -> MagicMock:
    """Build a mock wandb Run with realistic attributes."""
    run = MagicMock()
    run.id = run_id
    run.state = state
    run.config = {
        "peft": {"method_type": method, "r": 64} if method else None,
        "steps": steps,
        "dataset": {"repo_id": dataset},
    }
    run.summary = {"eval/pc_success": final_eval, "_step": steps}

    # Build history DataFrame (loss + occasional eval rows interleaved)
    loss_rows = loss_rows or [(i * 100, 1.0 - i * 0.01) for i in range(1, 11)]
    eval_rows = eval_rows or [(20_000, 8.0), (40_000, 8.0), (60_000, 20.0)]

    # Rows with only loss (no eval)
    loss_data = [
        {"train/steps": step, "train/loss": loss, "eval/pc_success": None}
        for step, loss in loss_rows
    ]
    # Rows with eval (also have loss values)
    eval_data = [
        {"train/steps": step, "train/loss": 0.5, "eval/pc_success": success}
        for step, success in eval_rows
    ]
    df = pd.DataFrame(loss_data + eval_data)
    run.history.return_value = df
    return run


@pytest.fixture
def two_runs() -> list[RunData]:
    """Pre-built RunData list for plot tests (no WandB API needed)."""
    loss = [LossPoint(step=s, loss=v) for s, v in [(1000, 0.9), (2000, 0.5), (3000, 0.2)]]
    ckpts = [
        EvalCheckpoint(step=20_000, success_rate=8.0),
        EvalCheckpoint(step=60_000, success_rate=20.0),
    ]
    expert = RunData(
        id="expert-id",
        method="expert",
        dataset="HuggingFaceVLA/libero",
        steps=100_000,
        final_eval=20.0,
        eval_checkpoints=ckpts,
        loss_history=loss,
    )
    lora = RunData(
        id="lora-id",
        method="lora",
        dataset="HuggingFaceVLA/libero",
        steps=100_000,
        final_eval=32.0,
        eval_checkpoints=ckpts,
        loss_history=loss,
    )
    return [expert, lora]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def test_eval_checkpoint_is_dataclass() -> None:
    ckpt = EvalCheckpoint(step=20_000, success_rate=8.0)
    assert ckpt.step == 20_000
    assert ckpt.success_rate == 8.0


def test_loss_point_is_dataclass() -> None:
    pt = LossPoint(step=100, loss=0.95)
    assert pt.step == 100
    assert pt.loss == 0.95


def test_run_data_is_dataclass() -> None:
    rd = RunData(
        id="abc",
        method="expert",
        dataset="HuggingFaceVLA/libero",
        steps=100_000,
        final_eval=20.0,
        eval_checkpoints=[],
        loss_history=[],
    )
    assert rd.id == "abc"
    assert rd.method == "expert"


# ---------------------------------------------------------------------------
# fetch_runs — filtering and parsing
# ---------------------------------------------------------------------------


def test_fetch_runs_skips_failed_runs() -> None:
    failed = _make_mock_run("fail-id", state="failed")
    finished = _make_mock_run("done-id", state="finished")
    api = MagicMock()
    api.runs.return_value = [failed, finished]

    with patch("smolvla_manuf.analysis.wandb.Api", return_value=api):
        result = fetch_runs("smolvla-libero")

    assert len(result) == 1
    assert result[0].id == "done-id"


def test_fetch_runs_parses_lora_method() -> None:
    lora_run = _make_mock_run("lora-id", method="LORA")
    api = MagicMock()
    api.runs.return_value = [lora_run]

    with patch("smolvla_manuf.analysis.wandb.Api", return_value=api):
        result = fetch_runs("smolvla-libero")

    assert result[0].method == "lora"


def test_fetch_runs_parses_expert_method() -> None:
    expert_run = _make_mock_run("exp-id", method=None)
    api = MagicMock()
    api.runs.return_value = [expert_run]

    with patch("smolvla_manuf.analysis.wandb.Api", return_value=api):
        result = fetch_runs("smolvla-libero")

    assert result[0].method == "expert"


def test_fetch_runs_parses_eval_checkpoints() -> None:
    run = _make_mock_run("r1", eval_rows=[(20_000, 8.0), (60_000, 20.0)])
    api = MagicMock()
    api.runs.return_value = [run]

    with patch("smolvla_manuf.analysis.wandb.Api", return_value=api):
        result = fetch_runs("smolvla-libero")

    assert len(result[0].eval_checkpoints) == 2
    assert result[0].eval_checkpoints[0].step == 20_000
    assert result[0].eval_checkpoints[0].success_rate == 8.0
    assert result[0].eval_checkpoints[1].success_rate == 20.0


def test_fetch_runs_parses_loss_history() -> None:
    loss_rows = [(100, 0.9), (200, 0.7), (300, 0.5)]
    run = _make_mock_run("r1", loss_rows=loss_rows)
    api = MagicMock()
    api.runs.return_value = [run]

    with patch("smolvla_manuf.analysis.wandb.Api", return_value=api):
        result = fetch_runs("smolvla-libero")

    # loss_history includes all rows with non-null train/loss (pure-loss rows + eval rows)
    assert len(result[0].loss_history) >= len(loss_rows)
    # first entry matches the first explicit loss row
    assert result[0].loss_history[0].step == 100
    assert result[0].loss_history[0].loss == pytest.approx(0.9)


def test_fetch_runs_parses_dataset() -> None:
    run = _make_mock_run("r1", dataset="HuggingFaceVLA/smol-libero")
    api = MagicMock()
    api.runs.return_value = [run]

    with patch("smolvla_manuf.analysis.wandb.Api", return_value=api):
        result = fetch_runs("smolvla-libero")

    assert result[0].dataset == "HuggingFaceVLA/smol-libero"


# ---------------------------------------------------------------------------
# export_metrics — JSON output
# ---------------------------------------------------------------------------


def test_export_metrics_creates_file(two_runs: list[RunData], tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    export_metrics(two_runs, out)
    assert out.exists()


def test_export_metrics_valid_json(two_runs: list[RunData], tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    export_metrics(two_runs, out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_export_metrics_schema(two_runs: list[RunData], tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    export_metrics(two_runs, out)
    data = json.loads(out.read_text())

    assert "runs" in data
    assert len(data["runs"]) == 2

    run = data["runs"][0]
    assert "id" in run
    assert "method" in run
    assert "dataset" in run
    assert "steps" in run
    assert "final_eval" in run
    assert "eval_checkpoints" in run

    ckpt = run["eval_checkpoints"][0]
    assert "step" in ckpt
    assert "success_rate" in ckpt


# ---------------------------------------------------------------------------
# Plot generation — file creation (CPU-only, no display)
# ---------------------------------------------------------------------------


def test_plot_loss_curves_creates_png(two_runs: list[RunData], tmp_path: Path) -> None:
    out = tmp_path / "loss_curves.png"
    plot_loss_curves(two_runs, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_eval_progression_creates_png(two_runs: list[RunData], tmp_path: Path) -> None:
    out = tmp_path / "eval_progression.png"
    plot_eval_progression(two_runs, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_data_scaling_creates_png(two_runs: list[RunData], tmp_path: Path) -> None:
    # Add a smol-libero run for scaling comparison
    smol = RunData(
        id="smol-id",
        method="expert",
        dataset="HuggingFaceVLA/smol-libero",
        steps=50_000,
        final_eval=0.0,
        eval_checkpoints=[],
        loss_history=[],
    )
    out = tmp_path / "data_scaling.png"
    plot_data_scaling(two_runs + [smol], out)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Bug fixes — regression tests
# ---------------------------------------------------------------------------


def test_plot_data_scaling_handles_empty_runs(tmp_path: Path) -> None:
    out = tmp_path / "scaling.png"
    plot_data_scaling([], out)  # must not raise ValueError: max() arg is an empty sequence
    assert out.exists()


def test_fetch_runs_handles_missing_train_steps_column() -> None:
    """_parse_run must not KeyError when history lacks 'train/steps'."""
    run = MagicMock()
    run.id = "r1"
    run.state = "finished"
    run.config = {"peft": None, "steps": 100, "dataset": {"repo_id": "HuggingFaceVLA/libero"}}
    run.summary = {"eval/pc_success": 10.0, "_step": 100}
    # History DataFrame missing the 'train/steps' column entirely
    df = pd.DataFrame({"train/loss": [0.5, 0.4], "eval/pc_success": [None, 10.0]})
    run.history.return_value = df
    api = MagicMock()
    api.runs.return_value = [run]

    with patch("smolvla_manuf.analysis.wandb.Api", return_value=api):
        result = fetch_runs("smolvla-libero")

    assert result[0].eval_checkpoints == []
    assert result[0].loss_history == []

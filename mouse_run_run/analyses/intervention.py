"""Paired behavioral comparison of baseline and intervened rollouts."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import torch

from mouse_run_run.analyses.base import AnalysisResult, write_analysis
from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, TerminalProgress


def compare_rollouts(
    baseline: RolloutArtifact,
    disturbed: RolloutArtifact,
    output: Path,
    *,
    progress: ProgressSink | None = None,
) -> AnalysisResult:
    _validate_pair(baseline, disturbed)
    tracker = ProgressTracker(
        "analyze:intervention",
        baseline.trajectory.batch_size,
        progress or TerminalProgress(),
    )
    baseline_rows = baseline.episodes.to_pylist()
    disturbed_rows = disturbed.episodes.to_pylist()
    rows: list[dict[str, object]] = []
    numeric_columns = [
        name
        for name in baseline.episodes.column_names
        if name.startswith("return.") or name.startswith("event.")
    ]
    action_changed: dict[str, torch.Tensor] = {}
    for agent_id in baseline.trajectory.agent_ids:
        action_changed[str(agent_id)] = (
            baseline.trajectory.agents[agent_id].actions
            != disturbed.trajectory.agents[agent_id].actions
        )
    for episode in range(baseline.trajectory.batch_size):
        for metric in numeric_columns:
            left = float(baseline_rows[episode][metric])
            right = float(disturbed_rows[episode][metric])
            rows.append(
                {
                    "episode_index": episode,
                    "metric": metric,
                    "baseline": left,
                    "disturbed": right,
                    "delta": right - left,
                }
            )
        for agent_id in baseline.trajectory.agent_ids:
            fraction = action_changed[str(agent_id)][:, episode].float().mean()
            rows.append(
                {
                    "episode_index": episode,
                    "metric": f"action_change_fraction.{agent_id}",
                    "baseline": 0.0,
                    "disturbed": float(fraction),
                    "delta": float(fraction),
                }
            )
        tracker.emit(episode + 1)
    result = write_analysis(
        output,
        name="intervention_comparison",
        rollout=disturbed,
        config={
            "baseline": str(baseline.path),
            "disturbed": str(disturbed.path),
            "paired_by": "episode_index",
        },
        table=pa.Table.from_pylist(rows),
        tensors={
            f"action_changed.{agent_id}": values
            for agent_id, values in action_changed.items()
        },
    )
    tracker.emit(baseline.trajectory.batch_size, state="completed")
    return result


def _validate_pair(baseline: RolloutArtifact, disturbed: RolloutArtifact) -> None:
    if baseline.manifest["experiment"] != disturbed.manifest["experiment"]:
        raise ValueError("paired rollouts use different experiments")
    if (
        baseline.trajectory.agent_ids != disturbed.trajectory.agent_ids
        or baseline.trajectory.horizon != disturbed.trajectory.horizon
        or baseline.trajectory.batch_size != disturbed.trajectory.batch_size
    ):
        raise ValueError("paired rollouts have incompatible agents or axes")
    for key, value in baseline.trajectory.world.items():
        if key not in disturbed.trajectory.world:
            raise ValueError(f"disturbed rollout is missing world field {key!r}")
        if not torch.equal(value[0], disturbed.trajectory.world[key][0]):
            raise ValueError(
                f"paired rollouts have different initial world field {key!r}; "
                "collect them with the same seed and batch size"
            )

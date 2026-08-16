"""Linear CKA across every pair of named agents in one rollout."""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import pyarrow as pa
import torch

from mouse_run_run.analyses.base import AnalysisResult, write_analysis
from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, TerminalProgress


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("linear CKA requires paired matrices (samples, features)")
    if left.shape[0] < 2:
        raise ValueError("linear CKA requires at least two samples")
    x = left.double() - left.double().mean(dim=0)
    y = right.double() - right.double().mean(dim=0)
    cross = (x.T @ y).square().sum()
    denominator = (x.T @ x).square().sum().sqrt() * (y.T @ y).square().sum().sqrt()
    if denominator <= 0:
        raise ValueError("linear CKA is undefined for constant representations")
    return cross / denominator


def run_linear_cka(
    rollout: RolloutArtifact,
    output: Path,
    *,
    site: str = "hidden",
    progress: ProgressSink | None = None,
) -> AnalysisResult:
    pairs = list(combinations_with_replacement(rollout.trajectory.agent_ids, 2))
    tracker = ProgressTracker("analyze:cka", len(pairs), progress or TerminalProgress())
    rows: list[dict[str, object]] = []
    matrix = torch.empty(len(rollout.trajectory.agent_ids), len(rollout.trajectory.agent_ids))
    agent_indices = {
        agent_id: index for index, agent_id in enumerate(rollout.trajectory.agent_ids)
    }
    mask = rollout.trajectory.active
    for index, (left_id, right_id) in enumerate(pairs, start=1):
        left = rollout.trajectory.agents[left_id].activations.get(site)
        right = rollout.trajectory.agents[right_id].activations.get(site)
        if left is None or right is None:
            raise ValueError(f"activation site {site!r} is missing for CKA")
        score = linear_cka(left[mask], right[mask])
        left_index = agent_indices[left_id]
        right_index = agent_indices[right_id]
        matrix[left_index, right_index] = score
        matrix[right_index, left_index] = score
        rows.append(
            {
                "agent_a": str(left_id),
                "agent_b": str(right_id),
                "site": site,
                "samples": int(mask.sum()),
                "linear_cka": float(score),
            }
        )
        tracker.emit(index, {"linear_cka": float(score)})
    result = write_analysis(
        output,
        name="linear_cka",
        rollout=rollout,
        config={"site": site},
        table=pa.Table.from_pylist(rows),
        tensors={"cka_matrix": matrix},
    )
    tracker.emit(len(pairs), state="completed")
    return result

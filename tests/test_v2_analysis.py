from __future__ import annotations

from pathlib import Path

import torch
import pyarrow.compute as pc

from mouse_run_run.analyses.cka import run_linear_cka
from mouse_run_run.analyses.intervention import compare_rollouts
from mouse_run_run.analyses.plsc import (
    FixedRank,
    PermutationThreshold,
    PLSCConfig,
    fit_plsc,
    load_fitted_plsc,
    save_fitted_plsc,
)
from mouse_run_run.artifacts.rollout import save_rollout
from mouse_run_run.core.simulation import AgentTrajectory, TrajectoryBatch
from mouse_run_run.core.types import AgentId


A = AgentId("a")
B = AgentId("b")


def _artifact(path: Path, *, disturbed: bool = False):
    generator = torch.Generator().manual_seed(4)
    time, episodes, width = 30, 5, 6
    shared = torch.randn(time, episodes, 2, generator=generator)
    left = torch.randn(time, episodes, width, generator=generator) * 0.1
    right = torch.randn(time, episodes, width, generator=generator) * 0.1
    left[..., :2] += shared
    right[..., :2] += shared
    actions = torch.zeros(time, episodes, dtype=torch.long)
    if disturbed:
        actions[:, 0] = 1
    def agent(hidden: torch.Tensor, offset: float) -> AgentTrajectory:
        return AgentTrajectory(
            observations=torch.zeros(time, episodes, 1),
            actions=actions.clone(),
            rewards=torch.full((time, episodes), offset),
            logits=torch.zeros(time, episodes, 2),
            values=torch.zeros(time, episodes),
            log_probs=torch.zeros(time, episodes),
            bootstrap_value=torch.zeros(episodes),
            activations={"hidden": hidden},
        )
    trajectory = TrajectoryBatch(
        agent_ids=(A, B),
        agents={A: agent(left, 1.0), B: agent(right, -1.0)},
        terminated=torch.zeros(time, episodes, dtype=torch.bool),
        truncated=torch.zeros(time, episodes, dtype=torch.bool),
        active=torch.ones(time, episodes, dtype=torch.bool),
        world={"state": torch.zeros(time + 1, episodes, 1)},
        events={"contact": actions.bool()},
        seed=8,
        episode_seeds=torch.full((episodes,), 8, dtype=torch.long),
    )
    return save_rollout(
        path,
        experiment="synthetic",
        checkpoint=Path("unused.safetensors"),
        trajectory=trajectory,
        config={},
        interventions=(({"name": "disturbed"},) if disturbed else ()),
    )


def test_plsc_subspace_round_trip_and_orthogonal_controls(tmp_path: Path) -> None:
    rollout = _artifact(tmp_path / "rollout")
    fitted = fit_plsc(
        rollout,
        PLSCConfig(A, B, selection=FixedRank(2), control_rank=2, control_seed=9),
    )
    path = tmp_path / "plsc"
    save_fitted_plsc(path, fitted, rollout=rollout.path)
    loaded = load_fitted_plsc(path)
    subspace = loaded.subspace(A)
    values = rollout.trajectory.agents[A].activations["hidden"][:, 0]
    shared = subspace.keep(values)
    unique = subspace.remove(values)
    reconstructed = subspace.mean + (shared - subspace.mean) + (unique - subspace.mean)
    assert torch.allclose(reconstructed, values.double(), atol=1e-6)
    top = loaded.subspace(A, "top_unique").basis
    random = loaded.subspace(A, "random_unique").basis
    assert torch.allclose(subspace.basis.T @ top, torch.zeros(2, 2, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(subspace.basis.T @ random, torch.zeros(2, 2, dtype=torch.float64), atol=1e-9)


def test_cka_and_intervention_comparison_write_parquet(tmp_path: Path) -> None:
    baseline = _artifact(tmp_path / "baseline")
    disturbed = _artifact(tmp_path / "disturbed", disturbed=True)
    cka = run_linear_cka(baseline, tmp_path / "cka")
    comparison = compare_rollouts(baseline, disturbed, tmp_path / "comparison")
    assert cka.table.num_rows == 3
    assert (cka.output / "results.parquet").is_file()
    assert comparison.table.num_rows > 0
    assert (
        comparison.table.filter(
            pc.equal(comparison.table["metric"], "action_change_fraction.a")
        ).num_rows
        == baseline.trajectory.batch_size
    )


def test_permutation_threshold_preserves_exact_noncontiguous_selection(tmp_path: Path) -> None:
    rollout = _artifact(tmp_path / "rollout")
    fitted = fit_plsc(
        rollout,
        PLSCConfig(
            A,
            B,
            selection=PermutationThreshold(
                permutations=40,
                null_model="time_shuffle",
                seed=2,
            ),
        ),
    )
    assert fitted.selected.dtype == torch.bool
    assert fitted.selected[:2].tolist() == [True, True]
    assert fitted.selected_indices.tolist() == torch.nonzero(
        fitted.singular_values > fitted.thresholds, as_tuple=False
    ).flatten().tolist()

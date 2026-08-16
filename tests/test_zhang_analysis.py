from __future__ import annotations

from pathlib import Path

import torch

from mouse_run_run.analyses.decoder import (
    DecoderConfig,
    episode_grouped_splits,
    run_diagnostic_decoder,
)
from mouse_run_run.analyses.paper_plsc import (
    PaperPLSCConfig,
    fit_paper_plsc,
    identical_action_mask,
    temporally_shifted_random_pc_basis,
)
from mouse_run_run.artifacts.rollout import save_rollout
from mouse_run_run.core.simulation import AgentTrajectory, TrajectoryBatch
from mouse_run_run.core.types import AgentId


A = AgentId("chaser")
B = AgentId("explorer")


def _paper_fixture(path: Path, *, no_events: bool = False, stuck: bool = False):
    generator = torch.Generator().manual_seed(41)
    time, episodes, width = 40, 6, 6
    phase = torch.linspace(0, 8 * torch.pi, time)
    shared = torch.stack((phase.sin(), phase.cos()), dim=-1)[:, None].repeat(1, episodes, 1)
    episode_offsets = torch.randn(1, episodes, 2, generator=generator) * 0.05
    shared = shared + episode_offsets
    left = torch.randn(time, episodes, width, generator=generator) * 0.08
    right = torch.randn(time, episodes, width, generator=generator) * 0.08
    left[..., :2] += shared
    right[..., :2] += shared
    actions_a = torch.randint(4, (time, episodes), generator=generator)
    actions_b = torch.randint(4, (time, episodes), generator=generator)
    collision = torch.zeros(time, episodes, dtype=torch.bool)
    approach = torch.zeros_like(collision)
    escape = torch.zeros_like(collision)
    if not no_events:
        collision[::7] = True
        approach[1::5] = True
        escape[2::5] = True

    def agent(hidden: torch.Tensor, actions: torch.Tensor) -> AgentTrajectory:
        return AgentTrajectory(
            observations=torch.zeros(time, episodes, 1),
            actions=actions,
            rewards=torch.zeros(time, episodes),
            logits=torch.zeros(time, episodes, 4),
            values=torch.zeros(time, episodes),
            log_probs=torch.zeros(time, episodes),
            bootstrap_value=torch.zeros(episodes),
            activations={"hidden": hidden},
        )

    chaser_positions = torch.zeros(time + 1, episodes, 2, dtype=torch.long)
    explorer_positions = torch.ones_like(chaser_positions)
    if not stuck:
        chaser_positions[..., 0] = torch.arange(time + 1).unsqueeze(1) % 10
        explorer_positions[..., 1] = torch.arange(time + 1).unsqueeze(1) % 10
    trajectory = TrajectoryBatch(
        agent_ids=(A, B),
        agents={A: agent(left, actions_a), B: agent(right, actions_b)},
        terminated=torch.zeros(time, episodes, dtype=torch.bool),
        truncated=torch.zeros(time, episodes, dtype=torch.bool),
        active=torch.ones(time, episodes, dtype=torch.bool),
        world={
            "chaser_position": chaser_positions,
            "explorer_position": explorer_positions,
            "distance": torch.ones(time + 1, episodes),
        },
        events={
            "collision": collision,
            "chaser_approach": approach,
            "explorer_escape": escape,
        },
        seed=1,
        episode_seeds=torch.arange(episodes),
    )
    return save_rollout(
        path,
        experiment="synthetic_zhang",
        checkpoint=Path("unused.safetensors"),
        trajectory=trajectory,
        config={},
    )


def test_paper_plsc_uses_dual_threshold_contiguous_prefix_and_delta_pcc(
    tmp_path: Path,
) -> None:
    rollout = _paper_fixture(tmp_path / "rollout")
    fitted = fit_paper_plsc(
        rollout,
        PaperPLSCConfig(
            A,
            B,
            permutations=40,
            percentile=0.95,
            min_shift=5,
            seed=7,
            null_batch_size=4,
        ),
    )
    expected = torch.cumprod(fitted.passes_both.long(), dim=0).bool()
    assert torch.equal(fitted.selected, expected)
    assert fitted.significant_count >= 1
    assert fitted.plsc1_pcc > fitted.null_plsc1_pcc
    assert fitted.delta_pcc == fitted.plsc1_pcc - fitted.null_plsc1_pcc
    assert torch.equal(
        fitted.passes_both,
        (fitted.covariance > fitted.covariance_threshold)
        & (fitted.projected_score_pcc > fitted.pcc_threshold),
    )


def test_identical_action_mask_and_episode_group_splits_do_not_leak(
    tmp_path: Path,
) -> None:
    rollout = _paper_fixture(tmp_path / "rollout")
    mask = identical_action_mask(rollout, A, B)
    assert torch.equal(
        mask,
        rollout.trajectory.agents[A].actions != rollout.trajectory.agents[B].actions,
    )
    labels = torch.tensor([0, 1] * 12)
    groups = torch.arange(6).repeat_interleave(4)
    for train, test in episode_grouped_splits(labels, groups, folds=3, seed=2):
        assert set(groups[train].tolist()).isdisjoint(groups[test].tolist())


def test_temporally_shifted_control_is_in_shared_complement(tmp_path: Path) -> None:
    rollout = _paper_fixture(tmp_path / "rollout")
    fitted = fit_paper_plsc(
        rollout,
        PaperPLSCConfig(
            A,
            B,
            permutations=4,
            percentile=0.75,
            min_shift=5,
            null_batch_size=2,
        ),
    )
    control = temporally_shifted_random_pc_basis(
        fitted,
        rollout,
        agent_id=A,
        shared_rank=2,
        control_rank=2,
        seed=3,
    )
    shared = fitted.bases[A][:, :2]
    assert torch.allclose(
        shared.T @ control,
        torch.zeros(2, 2, dtype=torch.float64),
        atol=1e-9,
    )
    assert torch.allclose(
        control.T @ control,
        torch.eye(2, dtype=torch.float64),
        atol=1e-9,
    )


def test_decoder_reports_insufficient_events_without_supplementing(
    tmp_path: Path,
) -> None:
    rollout = _paper_fixture(tmp_path / "rollout", no_events=True)
    table = run_diagnostic_decoder(
        rollout,
        tmp_path / "decoder",
        config=DecoderConfig(folds=2, shuffled_controls=2),
    )
    assert set(table["status"].to_pylist()) == {"insufficient"}
    assert all(value is None for value in table["balanced_accuracy"].to_pylist())


def test_degenerate_episode_diagnostic_marks_long_noncollision_stuck_run(
    tmp_path: Path,
) -> None:
    rollout = _paper_fixture(tmp_path / "rollout", stuck=True)
    assert rollout.episodes["quality.degenerate"].to_pylist() == [True] * 6
    assert rollout.episodes[
        "quality.longest_noncollision_stuck_run"
    ].to_pylist() == [6] * 6

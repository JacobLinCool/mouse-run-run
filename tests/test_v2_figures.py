from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from experiments.chase_grid.experiment import ChaseExperimentConfig, build_experiment
from experiments.chase_grid.model import ModelConfig
from mouse_run_run.analyses.flow_field import (
    angle_histogram_rows,
    flow_field_rows,
    movement_samples,
    polar_rows,
)
from mouse_run_run.analyses.statistics import permutation_comparison, significance_stars
from mouse_run_run.core.config import PPOConfig, TrainingConfig
from mouse_run_run.core.experiment import RuntimeConfig
from mouse_run_run.core.simulation import AgentTrajectory, TrajectoryBatch
from mouse_run_run.figures import sources
from mouse_run_run.figures.fig5 import Fig5Config, render_fig5
from mouse_run_run.figures.sources import CHASER_MATCHUP, EXPLORER_MATCHUP
from mouse_run_run.rollouts import collect_experiment_rollout
from mouse_run_run.training.runner import train_experiment


CHECKPOINTS = (1, 2)


def _trajectory(chaser: list[tuple[int, int]], partner: tuple[int, int]) -> TrajectoryBatch:
    steps = len(chaser) - 1
    positions = torch.tensor(chaser, dtype=torch.long).unsqueeze(1)
    partner_positions = torch.tensor(partner, dtype=torch.long).repeat(steps + 1, 1, 1)
    visible = torch.ones(steps + 1, 1, dtype=torch.bool)
    agent = AgentTrajectory(
        observations=torch.zeros(steps, 1, 1),
        actions=torch.zeros(steps, 1, dtype=torch.long),
        rewards=torch.zeros(steps, 1),
        logits=torch.zeros(steps, 1, 4),
        values=torch.zeros(steps, 1),
        log_probs=torch.zeros(steps, 1),
        bootstrap_value=torch.zeros(1),
    )
    return TrajectoryBatch(
        agent_ids=("chaser",),
        agents={"chaser": agent},
        terminated=torch.zeros(steps, 1, dtype=torch.bool),
        truncated=torch.zeros(steps, 1, dtype=torch.bool),
        active=torch.ones(steps, 1, dtype=torch.bool),
        world={
            "chaser_position": positions,
            "explorer_position": partner_positions,
            "chaser_partner_visible": visible,
        },
        events={},
        seed=0,
        episode_seeds=torch.zeros(1, dtype=torch.long),
    )


def test_movement_towards_partner_has_zero_angle() -> None:
    # The chaser starts three columns left of the partner and walks right.
    trajectory = _trajectory([(5, 2), (5, 3), (5, 4)], (5, 5))
    samples = movement_samples(trajectory)
    assert len(samples) == 2
    assert torch.allclose(samples.angle, torch.zeros(2, dtype=torch.float64))
    assert samples.offset.tolist() == [[-3.0, 0.0], [-2.0, 0.0]]
    assert samples.movement.tolist() == [[1.0, 0.0], [1.0, 0.0]]
    field = {(row["offset_x"], row["offset_y"]): row for row in flow_field_rows(samples, vision_radius=3)}
    assert field[(-3, 0)]["movement_x"] == pytest.approx(1.0)
    assert field[(-3, 0)]["samples"] == 1
    assert field[(1, 1)]["samples"] == 0
    histogram = angle_histogram_rows(samples)
    assert histogram[0]["label"] == "1-30"
    assert histogram[0]["probability"] == pytest.approx(1.0)
    polar = polar_rows(samples)
    assert polar[0]["bin_center"] == pytest.approx(0.0)
    assert polar[0]["probability"] == pytest.approx(1.0)


def test_movement_away_from_partner_is_a_straight_angle() -> None:
    trajectory = _trajectory([(5, 2), (5, 1)], (5, 5))
    samples = movement_samples(trajectory)
    assert samples.angle.abs().tolist() == pytest.approx([180.0])
    assert angle_histogram_rows(samples)[-1]["label"] == "151-180"
    assert angle_histogram_rows(samples)[-1]["probability"] == pytest.approx(1.0)


def test_movement_samples_require_recorded_world() -> None:
    trajectory = _trajectory([(5, 2), (5, 3)], (5, 5))
    stripped = TrajectoryBatch(
        agent_ids=trajectory.agent_ids,
        agents=trajectory.agents,
        terminated=trajectory.terminated,
        truncated=trajectory.truncated,
        active=trajectory.active,
        world={},
        events={},
        seed=0,
        episode_seeds=trajectory.episode_seeds,
    )
    with pytest.raises(ValueError, match="record_world"):
        movement_samples(stripped)


def test_permutation_comparison_is_exact_for_small_groups() -> None:
    separated = permutation_comparison([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])
    assert separated.exact
    assert separated.p_value == pytest.approx(0.1)
    assert significance_stars(separated.p_value) == "ns"
    identical = permutation_comparison([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert identical.p_value > 0.5
    wide = permutation_comparison(list(range(20, 40)), list(range(20)), permutations=2_000)
    assert not wide.exact
    assert wide.p_value < 0.01


def _study_directory(tmp_path: Path) -> Path:
    experiment = build_experiment(
        ChaseExperimentConfig(
            chaser_model=ModelConfig(hidden_size=8),
            explorer_model=ModelConfig(hidden_size=8),
            training=TrainingConfig(
                updates=max(CHECKPOINTS),
                horizon=6,
                checkpoint_every=1,
                ppo=PPOConfig(epochs=1),
            ),
        )
    )
    study = tmp_path / "study"
    rows: list[dict[str, object]] = []
    for condition in ("social", "non_social"):
        for seed in (0, 1):
            unit = study / "units" / condition / f"seed_{seed:04d}"
            train_experiment(
                experiment,
                runtime=RuntimeConfig(batch_size=2, seed=seed + 1),
                run_dir=unit / "train",
            )
            for update in CHECKPOINTS:
                checkpoint = unit / "train" / "checkpoints" / f"update_{update:06d}.safetensors"
                artifact = collect_experiment_rollout(
                    experiment,
                    checkpoint,
                    unit / "behavior" / f"update_{update:06d}" / CHASER_MATCHUP,
                    runtime=RuntimeConfig(batch_size=2, seed=seed + 10),
                    episodes=2,
                    horizon=6,
                    deterministic=False,
                )
                for matchup in (CHASER_MATCHUP, EXPLORER_MATCHUP):
                    rows.extend(
                        {
                            "condition": condition,
                            "seed": seed,
                            "checkpoint_update": update,
                            "matchup": matchup,
                            **episode,
                        }
                        for episode in artifact.episodes.to_pylist()
                    )
    table = pa.Table.from_pylist(rows)
    (study / "tables").mkdir(parents=True)
    pq.write_table(table, study / "tables" / "behavior.parquet")
    return study


def test_render_fig5_writes_every_panel_source_table(tmp_path: Path) -> None:
    study = _study_directory(tmp_path)
    result = render_fig5(
        study,
        tmp_path / "figures",
        config=Fig5Config(
            early_checkpoint=CHECKPOINTS[0],
            late_checkpoint=CHECKPOINTS[-1],
            permutations=200,
            formats=("png",),
        ),
    )
    assert [path.name for path in result.figure] == ["fig5.png"]
    assert (tmp_path / "figures" / "fig5.png").stat().st_size > 10_000
    written = {path.stem for path in result.source_data}
    assert written == {
        "fig5_training",
        "fig5_random_opponent",
        "fig5_random_opponent_seeds",
        "fig5_random_opponent_comparisons",
        "fig5_flow_field",
        "fig5_polar",
        "fig5_angles",
        "fig5_angle_comparisons",
    }
    manifest = json.loads((tmp_path / "figures" / "manifest.json").read_text())
    assert manifest["config"]["early_checkpoint"] == CHECKPOINTS[0]
    assert manifest["notes"] == []
    assert manifest["rendering"]["flow_arrow_longest_step"] > 0


def test_training_sources_average_over_seeds(tmp_path: Path) -> None:
    study = tmp_path / "study"
    for condition, values in (("social", (2.0, 4.0)), ("non_social", (1.0, 1.0))):
        for seed, value in enumerate(values):
            path = study / "units" / condition / f"seed_{seed:04d}" / "train"
            path.mkdir(parents=True)
            (path / "metrics.jsonl").write_text(
                json.dumps({"update": 1, "event.collision": value}) + "\n",
                encoding="utf-8",
            )
    rows = sources.training_rows(study, ["event.collision"])
    summary = {row["condition"]: row for row in sources.summarize_curves(rows)}
    assert summary["social"]["mean"] == pytest.approx(3.0)
    assert summary["social"]["sem"] == pytest.approx(1.0)
    assert summary["non_social"]["sem"] == pytest.approx(0.0)
    assert summary["social"]["seeds"] == 2

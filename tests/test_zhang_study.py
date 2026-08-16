from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from experiments.zhang_2025_rnn.study import (
    definition,
    development_definition,
    protocol_definition,
    smoke_definition,
)
from mouse_run_run.artifacts.rollout import load_rollout
from mouse_run_run.studies.runner import plan_study, run_study
from tests.fixtures.zhang_study import development_smoke_definition


TARGET = "experiments.zhang_2025_rnn.study:smoke_definition"
DEVELOPMENT_SMOKE_TARGET = (
    "tests.fixtures.zhang_study:development_smoke_definition"
)


def test_confirmatory_plan_is_goal_directed_and_uses_one_fixed_checkpoint() -> None:
    plan = plan_study(definition)
    assert plan["purpose"] == "confirmatory"
    assert plan["seeds"] == list(range(100, 110))
    assert plan["units"] == 20
    assert plan["environment_steps_per_unit"] == 8_000_000
    assert plan["total_environment_steps"] == 160_000_000
    assert plan["optimizer_minibatches_per_agent_update"] == 1
    assert plan["optimizer_steps_per_update"] == 2
    assert plan["total_optimizer_steps"] == 80_000
    assert plan["checkpoint_updates"] == [500, 1_000, 1_500, 2_000]
    assert plan["recipes"]["behavior"]["checkpoint_updates"] == [2_000]
    assert plan["recipes"]["neural"]["checkpoint_updates"] == [2_000]
    assert plan["estimated_artifacts"] == {
        "training_checkpoints_including_latest": 100,
        "random_opponent_rollouts": 40,
        "neural_rollouts": 40,
        "paper_plsc_fits_including_identical_action_controls": 50,
        "diagnostic_decoder_rollouts": 20,
        "causal_rollouts": 40,
    }


def test_development_plan_is_behavior_only_checkpoint_sweep() -> None:
    plan = plan_study(development_definition)
    assert plan["purpose"] == "development"
    assert plan["enabled_stages"] == ["train", "behavior"]
    assert plan["units"] == 6
    assert plan["recipes"]["behavior"]["checkpoint_updates"] == list(
        range(500, 5_001, 500)
    )
    assert plan["estimated_artifacts"]["random_opponent_rollouts"] == 120
    assert plan["estimated_artifacts"]["neural_rollouts"] == 0
    assert plan["estimated_artifacts"]["diagnostic_decoder_rollouts"] == 0
    assert plan["estimated_artifacts"]["causal_rollouts"] == 0


def test_development_definition_rejects_neural_stage_before_writing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "invalid-development-stage"
    with pytest.raises(ValueError, match="does not enable stage 'neural'"):
        run_study(
            "experiments.zhang_2025_rnn.study:development_definition",
            development_definition,
            output=output,
            device="cpu",
            backend="torch",
            parallelism=1,
            stage="neural",
            resume=False,
        )
    assert not output.exists()


def test_protocol_sensitivity_plan_retains_expensive_reference_workload() -> None:
    plan = plan_study(protocol_definition)
    assert plan["purpose"] == "protocol_sensitivity"
    assert plan["units"] == 20
    assert plan["environment_steps_per_unit"] == 80_000_000
    assert plan["total_environment_steps"] == 1_600_000_000
    assert plan["optimizer_steps_per_update"] == 2_040
    assert plan["total_optimizer_steps"] == 816_000_000
    assert plan["checkpoint_updates"] == list(range(1_000, 20_001, 1_000))
    assert plan["recipes"]["neural"]["checkpoint_updates"] == list(
        range(15_000, 20_001, 1_000)
    )


def test_development_smoke_runs_only_enabled_stages_and_pairs_checkpoints(
    tmp_path: Path,
) -> None:
    output = tmp_path / "development"
    result = run_study(
        DEVELOPMENT_SMOKE_TARGET,
        development_smoke_definition,
        output=output,
        device="cpu",
        backend="torch",
        parallelism=1,
        stage="all",
        resume=False,
    )
    assert result.stages == ("train", "behavior")
    assert not (output / "tables" / "neural.parquet").exists()
    assert not (output / "tables" / "causal.parquet").exists()
    contrasts = pq.read_table(
        output / "tables" / "behavior_checkpoint_contrasts.parquet"
    )
    assert contrasts["checkpoint_update"].to_pylist() == [1, 2]

    behavior = output / "units" / "social" / "seed_0000" / "behavior"
    matchup = "trained_chaser_vs_uniform_explorer"
    first = load_rollout(behavior / "update_000001" / matchup).trajectory
    second = load_rollout(behavior / "update_000002" / matchup).trajectory
    assert torch.equal(first.episode_seeds, second.episode_seeds)
    assert torch.equal(
        first.world["chaser_position"][0],
        second.world["chaser_position"][0],
    )
    assert torch.equal(
        first.world["explorer_position"][0],
        second.world["explorer_position"][0],
    )


def test_tiny_cpu_study_runs_every_stage_and_keeps_gates_not_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "study"
    result = run_study(
        TARGET,
        smoke_definition,
        output=output,
        device="cpu",
        backend="torch",
        parallelism=1,
        stage="all",
        resume=False,
    )
    assert result.report.is_file()
    for name in (
        "behavior",
        "behavior_checkpoint_summary",
        "behavior_checkpoint_contrasts",
        "neural",
        "decoder",
        "causal",
        "plsc_seed_summary",
        "nature_comparison",
    ):
        assert (output / "tables" / f"{name}.parquet").is_file()
    assert pq.read_table(output / "tables" / "behavior.parquet").num_rows > 0
    assert (
        pq.read_table(output / "tables" / "behavior_checkpoint_summary.parquet")
        .column("checkpoint_update")
        .to_pylist()
    ) == [1, 1, 1, 1, 1, 1, 1, 1]
    assert pq.read_table(
        output / "tables" / "behavior_checkpoint_contrasts.parquet"
    ).num_rows == 1
    assert set(
        pq.read_table(output / "tables" / "behavior.parquet")[
            "checkpoint_update"
        ].to_pylist()
    ) == {1}
    behavior_root = output / "units" / "social" / "seed_0000" / "behavior"
    assert (
        behavior_root
        / "update_000001"
        / "trained_chaser_vs_uniform_explorer"
        / "tensors.safetensors"
    ).is_file()
    assert pq.read_table(output / "tables" / "decoder.parquet").num_rows > 0
    comparison = pq.read_table(output / "tables" / "nature_comparison.parquet")
    assert set(comparison["role"].to_pylist()) == {"comparison_only"}
    assert set(comparison["scientific_status"].to_pylist()) == {"NOT_RUN"}
    assert all(value is None for value in comparison["observed_mean"].to_pylist())
    gates = json.loads((output / "gates.json").read_text(encoding="utf-8"))["gates"]
    assert gates == {
        "behavior": "NOT_RUN",
        "plsc": "NOT_RUN",
        "causal": "NOT_RUN",
        "decoder": "DIAGNOSTIC_ONLY",
    }
    causal_root = output / "units" / "social" / "seed_0000" / "causal"
    assert (causal_root / "control_basis.safetensors").is_file()
    for condition in (
        "baseline",
        "shared_readout",
        "shifted_random_pc_readout",
        "shared_recurrent",
    ):
        assert (causal_root / condition / "tensors.safetensors").is_file()
        assert (causal_root / condition / "episodes.parquet").is_file()

    resumed = run_study(
        TARGET,
        smoke_definition,
        output=output,
        device="cpu",
        backend="torch",
        parallelism=1,
        stage="all",
        resume=True,
    )
    assert resumed.report == result.report

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.chase_grid.experiment import ChaseExperimentConfig, build_experiment
from experiments.chase_grid.model import ModelConfig
from mouse_run_run.artifacts.checkpoint import load_checkpoint, read_checkpoint_metadata
from mouse_run_run.artifacts.rollout import load_rollout
from mouse_run_run.core.config import PPOConfig, TrainingConfig
from mouse_run_run.core.experiment import RuntimeConfig
from mouse_run_run.rollouts import collect_experiment_rollout
from mouse_run_run.training.runner import train_experiment


def _experiment(updates: int):
    return build_experiment(
        ChaseExperimentConfig(
            chaser_model=ModelConfig(hidden_size=8),
            explorer_model=ModelConfig(hidden_size=8),
            training=TrainingConfig(
                updates=updates,
                horizon=4,
                checkpoint_every=1,
                ppo=PPOConfig(epochs=1),
            ),
        )
    )


def test_resume_matches_uninterrupted_training_exactly(tmp_path: Path) -> None:
    runtime = RuntimeConfig(batch_size=3, seed=12)
    uninterrupted = train_experiment(
        _experiment(2), runtime=runtime, run_dir=tmp_path / "uninterrupted"
    )
    first = train_experiment(_experiment(1), runtime=runtime, run_dir=tmp_path / "first")
    resumed = train_experiment(
        _experiment(2),
        runtime=runtime,
        run_dir=tmp_path / "resumed",
        resume_from=first.checkpoint,
    )
    left = load_file(str(uninterrupted.checkpoint))
    right = load_file(str(resumed.checkpoint))
    policy_keys = [key for key in left if key.startswith("policy.")]
    assert policy_keys
    assert all(torch.equal(left[key], right[key]) for key in policy_keys)
    metadata = read_checkpoint_metadata(resumed.checkpoint)
    assert metadata.update == 2
    assert set(metadata.learner_state["kl_coefficients"]) == {"chaser", "explorer"}


def test_rollout_round_trip_uses_safetensors_and_parquet(tmp_path: Path) -> None:
    experiment = _experiment(1)
    trained = train_experiment(
        experiment,
        runtime=RuntimeConfig(batch_size=2, seed=2),
        run_dir=tmp_path / "run",
    )
    artifact = collect_experiment_rollout(
        experiment,
        trained.checkpoint,
        tmp_path / "rollout",
        runtime=RuntimeConfig(batch_size=2, seed=4),
        episodes=3,
        horizon=4,
        deterministic=True,
    )
    loaded = load_rollout(artifact.path)
    assert (artifact.path / "tensors.safetensors").is_file()
    assert (artifact.path / "episodes.parquet").is_file()
    assert loaded.trajectory.batch_size == 3
    assert loaded.trajectory.world["chaser_position"].shape[:2] == (5, 3)


def test_v2_checkpoint_reader_rejects_unversioned_safetensors(tmp_path: Path) -> None:
    legacy = tmp_path / "old.safetensors"
    save_file({"weight": torch.ones(1)}, str(legacy))
    with pytest.raises(ValueError, match="legacy checkpoints are archived only"):
        load_checkpoint(legacy, policies={})


def test_active_runtime_has_one_cli_and_no_pickle_serialization() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["scripts"] == {"mrr": "mouse_run_run.cli:main"}
    source_roots = (
        root / "mouse_run_run",
        root / "experiments" / "chase_grid",
        root / "experiments" / "zhang_2025_rnn",
    )
    forbidden = ("torch.save(", "torch.load(", "pickle.dump(", "pickle.load(")
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in forbidden), path

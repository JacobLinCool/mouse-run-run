from __future__ import annotations

import json
from pathlib import Path

from experiments.chase_grid.experiment import ChaseExperimentConfig, build_experiment
from experiments.chase_grid.model import ModelConfig
from mouse_run_run.core.config import PPOConfig, TrainingConfig
from mouse_run_run.core.experiment import RuntimeConfig
from mouse_run_run.replay.payload import build_replay_payload
from mouse_run_run.rollouts import collect_experiment_rollout
from mouse_run_run.training.runner import train_experiment


def test_saved_rollout_builds_finite_replay_payload(tmp_path: Path) -> None:
    experiment = build_experiment(
        ChaseExperimentConfig(
            chaser_model=ModelConfig(hidden_size=6),
            explorer_model=ModelConfig(hidden_size=6),
            training=TrainingConfig(updates=1, horizon=3, ppo=PPOConfig(epochs=1)),
        )
    )
    trained = train_experiment(
        experiment,
        runtime=RuntimeConfig(batch_size=2, seed=1),
        run_dir=tmp_path / "run",
    )
    rollout = collect_experiment_rollout(
        experiment,
        trained.checkpoint,
        tmp_path / "rollout",
        runtime=RuntimeConfig(batch_size=2, seed=7),
        episodes=2,
        horizon=3,
        deterministic=True,
    )
    payload = build_replay_payload(rollout, experiment, episode=0)
    json.dumps(payload, allow_nan=False)
    assert len(payload["frames"]) == 4
    assert payload["neural"]["chaser"]["hidden"]["width"] == 6

from dataclasses import asdict

import torch

from mouse_run_run.analysis import load_rollout
from mouse_run_run.degenerate import episode_degeneracy
from mouse_run_run.env import GridWorldConfig
from mouse_run_run.plsc import compute_plsc
from mouse_run_run.policy import build_policy
from mouse_run_run.serialization import (
    TrainingState,
    load_checkpoint,
    load_training_state,
    read_checkpoint_metadata,
    save_checkpoint,
    save_rollout,
)


def test_checkpoint_and_rollout_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    config = GridWorldConfig(max_steps=5)
    chaser = build_policy("rnn", config.observation_size, hidden_size=8)
    explorer = build_policy("rnn", config.observation_size, hidden_size=8)
    chaser_optimizer = torch.optim.Adam(chaser.parameters(), lr=1e-3)
    explorer_optimizer = torch.optim.Adam(explorer.parameters(), lr=1e-3)
    minibatch_generator = torch.Generator().manual_seed(123)
    checkpoint = tmp_path / "checkpoint.safetensors"

    save_checkpoint(
        checkpoint,
        config={
            "env": asdict(config),
            "architecture": "rnn",
            "hidden_size": 8,
            "batch_size": 2,
            "seed": 0,
        },
        metrics={"collisions_per_episode": 0.0},
        chaser_state=chaser.state_dict(),
        explorer_state=explorer.state_dict(),
        training_state=TrainingState(
            update=1,
            optimizer_states={
                "chaser": chaser_optimizer.state_dict(),
                "explorer": explorer_optimizer.state_dict(),
            },
            learner_state={"chaser_kl_coeff": 0.2, "explorer_kl_coeff": 0.2},
            cpu_rng_state=torch.get_rng_state(),
            minibatch_rng_state=minibatch_generator.get_state(),
        ),
    )

    loaded_config, loaded_metrics, chaser_state, explorer_state = load_checkpoint(checkpoint)
    metadata_config, metadata_metrics = read_checkpoint_metadata(checkpoint)
    training_state = load_training_state(checkpoint)

    assert loaded_config == metadata_config
    assert loaded_metrics == metadata_metrics
    assert loaded_config["architecture"] == "rnn"
    assert "rnn.weight_ih_l0" in chaser_state
    assert "rnn.weight_ih_l0" in explorer_state
    assert training_state is not None
    assert training_state.update == 1
    assert set(training_state.optimizer_states) == {"chaser", "explorer"}
    assert training_state.learner_state["chaser_kl_coeff"] == 0.2
    assert torch.equal(training_state.minibatch_rng_state, minibatch_generator.get_state())

    rollout_path = tmp_path / "rollout.safetensors"
    save_rollout(
        rollout_path,
        metadata={"schema_version": 3, "note": "test"},
        rollout={
            "episode_degenerate": torch.tensor([False, True]),
            "chaser_hidden": torch.zeros(2, 2, 3),
            "explorer_hidden": torch.ones(2, 2, 3),
        },
    )

    rollout_metadata, tensors = load_rollout(rollout_path)
    assert rollout_metadata["schema_version"] == 3
    assert tensors["episode_degenerate"].tolist() == [False, True]
    assert tensors["chaser_hidden"].shape == (2, 2, 3)


def test_episode_degeneracy_detects_sustained_stuck_run() -> None:
    positions = torch.tensor(
        [
            [[0, 0]],
            [[0, 0]],
            [[0, 0]],
            [[0, 0]],
        ]
    )
    result = episode_degeneracy(
        positions,
        positions,
        torch.zeros(3, 1, dtype=torch.long),
        torch.zeros(3, 1, dtype=torch.long),
        threshold_fraction=0.5,
    )

    assert result["degenerate"].tolist() == [True]
    assert result["same_state_run_steps"].tolist() == [3]


def test_plsc_smoke_returns_finite_shared_structure() -> None:
    torch.manual_seed(0)
    chaser = torch.randn(24, 4)
    explorer = chaser + 0.05 * torch.randn(24, 4)

    result = compute_plsc(chaser, explorer, permutations=8, alpha=0.1, seed=0)

    assert result.singular_values.shape == (4,)
    assert result.null.shape == (8, 4)
    assert result.cross_covariance.shape == (4, 4)
    assert result.significant_dims >= 0
    assert torch.isfinite(result.singular_values).all()

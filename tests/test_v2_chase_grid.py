from __future__ import annotations

import torch

from experiments.chase_grid.environment import (
    CHASER,
    EXPLORER,
    ChaseGridConfig,
    ChaseGridEnvironment,
)
from experiments.chase_grid.model import ModelConfig, build_policy
from experiments.chase_grid.reward import RewardConfig
from mouse_run_run.core.experiment import PolicyBuildContext, RuntimeConfig


def test_seeded_reset_is_reproducible_and_never_overlaps_agents() -> None:
    runtime = RuntimeConfig(batch_size=32)
    config = ChaseGridConfig(grid_size=5)
    left = ChaseGridEnvironment(config, runtime, torch.device("cpu"))
    right = ChaseGridEnvironment(config, runtime, torch.device("cpu"))
    left.reset(generator=torch.Generator().manual_seed(9))
    right.reset(generator=torch.Generator().manual_seed(9))
    assert torch.equal(left.chaser_position, right.chaser_position)
    assert torch.equal(left.explorer_position, right.explorer_position)
    assert not ((left.chaser_position == left.explorer_position).all(dim=1)).any()


def test_reward_isolated_in_reward_config_and_collision_has_precedence() -> None:
    config = ChaseGridConfig(
        grid_size=3,
        vision_radius=1,
        max_steps=3,
        reward=RewardConfig(
            chaser_step=-2.0,
            explorer_step=-3.0,
            chaser_new_field=4.0,
            explorer_new_field=5.0,
            social_chaser_collision=7.0,
            social_explorer_collision=-11.0,
        ),
    )
    env = ChaseGridEnvironment(config, RuntimeConfig(batch_size=1), torch.device("cpu"))
    env.reset(generator=torch.Generator().manual_seed(1))
    env.chaser_position[:] = torch.tensor([[1, 0]])
    env.explorer_position[:] = torch.tensor([[1, 1]])
    result = env.step({CHASER: torch.tensor([1]), EXPLORER: torch.tensor([0])})
    assert result.events["chaser_collision"].item()
    assert result.rewards[CHASER].item() == 7.0
    assert result.rewards[EXPLORER].item() == -11.0


def test_rnn_and_cnn_rnn_share_the_same_policy_contract() -> None:
    context = PolicyBuildContext(CHASER, (2, 5, 5), 4, torch.device("cpu"))
    for kind in ("rnn", "cnn_rnn"):
        policy = build_policy(ModelConfig(kind=kind, hidden_size=7), context)
        state = policy.initial_state(3, torch.device("cpu"))
        features = policy.advance(torch.randn(3, 2, 5, 5), state)
        readout = policy.readout(features)
        assert features.activations["hidden"].shape == (3, 7)
        assert readout.logits.shape == (3, 4)
        assert readout.value.shape == (3,)


def test_non_social_hides_partner_input_but_preserves_physical_fov_event() -> None:
    config = ChaseGridConfig(
        grid_size=10,
        vision_radius=3,
        partner_visibility="none",
        spawn_mode="exclude_last",
    )
    env = ChaseGridEnvironment(config, RuntimeConfig(batch_size=4), torch.device("cpu"))
    env.reset(generator=torch.Generator().manual_seed(23))
    assert int(env.chaser_position.max()) <= 8
    assert int(env.explorer_position.max()) <= 8
    env.chaser_position[:] = torch.tensor([[4, 4]]).expand(4, -1)
    env.explorer_position[:] = torch.tensor([[4, 5]]).expand(4, -1)
    observations = env._observations()
    assert not observations[CHASER][:, 1].bool().any()
    transition = env.step(
        {
            CHASER: torch.zeros(4, dtype=torch.long),
            EXPLORER: torch.zeros(4, dtype=torch.long),
        }
    )
    assert transition.events["chaser_partner_visible"].all()

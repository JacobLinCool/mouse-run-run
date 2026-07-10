"""Parity tests for the fused rollout math and the one-hot observation encoders.

The launch gate requires fused_agent_rollout=True, so all real training data
flows through _FusedAgentPair/_FusedSSMPair and forward_one_hot_indices, while
the PPO update re-evaluates the same trajectories through the plain policy
modules (policy.sequence). Any fused/plain mismatch silently corrupts PPO
ratios, so these tests pin every reimplementation to the canonical forward
pass on CPU.
"""

import pytest
import torch

from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig
from mouse_run_run.policy import PolicyBase, build_policy
from mouse_run_run.training_config import TrainConfig
from mouse_run_run.training_rollout import (
    _build_grid_observations,
    _flat_position,
    _FusedAgentPair,
    _FusedSSMPair,
    _observation_from_flat,
    collect_rollout,
)

DEVICE = torch.device("cpu")
GRID_SIZE = 10
GRID_CELLS = GRID_SIZE * GRID_SIZE
OBSERVATION_SIZE = 2 * GRID_CELLS
HIDDEN_SIZE = 32
TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def _policy_pair(architecture: str, seed: int = 0) -> tuple[PolicyBase, PolicyBase]:
    torch.manual_seed(seed)
    chaser = build_policy(architecture, OBSERVATION_SIZE, hidden_size=HIDDEN_SIZE)
    explorer = build_policy(architecture, OBSERVATION_SIZE, hidden_size=HIDDEN_SIZE)
    return chaser, explorer


def _random_step_inputs(
    generator: torch.Generator,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chaser_flat = torch.randint(GRID_CELLS, (batch_size,), generator=generator)
    explorer_flat = torch.randint(GRID_CELLS, (batch_size,), generator=generator)
    visible = torch.rand(batch_size, generator=generator) < 0.5
    return chaser_flat, explorer_flat, visible


def _random_observation_sequence(
    steps: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    frames = []
    for _ in range(steps):
        own_flat, other_flat, visible = _random_step_inputs(generator, batch_size)
        frames.append(_observation_from_flat(own_flat, other_flat, visible, GRID_CELLS))
    return torch.stack(frames)


def _step_loop(
    policy: PolicyBase,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = policy.initial_hidden(observations.shape[1], observations.device)
    logits, values, hiddens = [], [], []
    for step in range(observations.shape[0]):
        output = policy(observations[step], state)
        logits.append(output.logits)
        values.append(output.value)
        hiddens.append(output.hidden)
        state = output.state
    return torch.stack(logits), torch.stack(values), torch.stack(hiddens)


@pytest.mark.parametrize(
    ("architecture", "fused_class"),
    [("rnn", _FusedAgentPair), ("ssm", _FusedSSMPair)],
)
@torch.no_grad()
def test_fused_pair_matches_plain_forward(architecture: str, fused_class: type) -> None:
    batch_size = 5
    steps = 20
    chaser, explorer = _policy_pair(architecture, seed=1)
    fused = fused_class(
        chaser,
        explorer,
        observation_size=OBSERVATION_SIZE,
        batch_size=batch_size,
        device=DEVICE,
    )
    chaser_state = chaser.initial_hidden(batch_size, DEVICE)
    explorer_state = explorer.initial_hidden(batch_size, DEVICE)
    generator = torch.Generator().manual_seed(11)

    for _ in range(steps):
        chaser_flat, explorer_flat, visible = _random_step_inputs(generator, batch_size)
        fused_chaser, fused_explorer = fused.step(chaser_flat, explorer_flat, visible)
        chaser_output = chaser(
            _observation_from_flat(chaser_flat, explorer_flat, visible, GRID_CELLS),
            chaser_state,
        )
        explorer_output = explorer(
            _observation_from_flat(explorer_flat, chaser_flat, visible, GRID_CELLS),
            explorer_state,
        )
        for fused_output, plain_output in (
            (fused_chaser, chaser_output),
            (fused_explorer, explorer_output),
        ):
            torch.testing.assert_close(fused_output.logits, plain_output.logits, **TOLERANCE)
            torch.testing.assert_close(fused_output.value, plain_output.value, **TOLERANCE)
            torch.testing.assert_close(fused_output.hidden, plain_output.hidden, **TOLERANCE)
        chaser_state = chaser_output.state
        explorer_state = explorer_output.state


@torch.no_grad()
def test_forward_one_hot_indices_matches_full_observation_forward() -> None:
    batch_size = 7
    steps = 15
    policy, _ = _policy_pair("rnn", seed=2)
    index_hidden = policy.initial_hidden(batch_size, DEVICE)
    full_hidden = policy.initial_hidden(batch_size, DEVICE)
    generator = torch.Generator().manual_seed(22)

    for _ in range(steps):
        own_flat, other_flat, visible = _random_step_inputs(generator, batch_size)
        index_output = policy.forward_one_hot_indices(
            own_index=own_flat,
            other_index=GRID_CELLS + other_flat,
            other_visible=visible,
            hidden=index_hidden,
        )
        full_output = policy(
            _observation_from_flat(own_flat, other_flat, visible, GRID_CELLS),
            full_hidden,
        )
        torch.testing.assert_close(index_output.logits, full_output.logits, **TOLERANCE)
        torch.testing.assert_close(index_output.value, full_output.value, **TOLERANCE)
        torch.testing.assert_close(index_output.hidden, full_output.hidden, **TOLERANCE)
        index_hidden = index_output.state
        full_hidden = full_output.state


# steps=12 drives the SSM's chunked linear scan (chunk length 3); steps=13 is
# prime so _chunked_linear_scan falls back to the sequential loop.
@pytest.mark.parametrize(
    ("architecture", "steps"),
    [("rnn", 12), ("mlp", 12), ("ssm", 12), ("ssm", 13), ("transformer", 12)],
)
@torch.no_grad()
def test_sequence_eval_matches_step_iterated_forward(architecture: str, steps: int) -> None:
    batch_size = 4
    torch.manual_seed(3)
    policy = build_policy(architecture, OBSERVATION_SIZE, hidden_size=HIDDEN_SIZE)
    observations = _random_observation_sequence(steps, batch_size, seed=33)

    step_logits, step_values, step_hiddens = _step_loop(policy, observations)
    hiddens = policy.hidden_sequence(observations)
    logits, values, final_hidden = policy.sequence(observations)

    torch.testing.assert_close(hiddens, step_hiddens, **TOLERANCE)
    torch.testing.assert_close(logits, step_logits, **TOLERANCE)
    torch.testing.assert_close(values, step_values, **TOLERANCE)
    torch.testing.assert_close(final_hidden, step_hiddens[-1], **TOLERANCE)


@torch.no_grad()
def test_one_hot_observation_encoders_agree() -> None:
    batch_size = 64
    torch.manual_seed(4)
    env = BatchedChaseEnv(GridWorldConfig(), batch_size, DEVICE)
    env.reset_state()
    # Craft boundary states: max-offset exactly at the vision radius (visible)
    # and one past it (hidden).
    env.chaser_position[-2:] = torch.tensor([[0, 0], [0, 0]])
    env.explorer_position[-2:] = torch.tensor([[3, 3], [4, 0]])

    visible = env.partner_visible()
    assert bool(visible[-2]) and not bool(visible[-1])
    assert bool(visible.any()) and bool((~visible).any())

    env_chaser_obs, env_explorer_obs = env.observations()
    chaser_flat = _flat_position(env.chaser_position, GRID_SIZE)
    explorer_flat = _flat_position(env.explorer_position, GRID_SIZE)

    flat_chaser_obs = _observation_from_flat(chaser_flat, explorer_flat, visible, GRID_CELLS)
    flat_explorer_obs = _observation_from_flat(explorer_flat, chaser_flat, visible, GRID_CELLS)
    assert torch.equal(flat_chaser_obs, env_chaser_obs)
    assert torch.equal(flat_explorer_obs, env_explorer_obs)

    grid_chaser_obs = _build_grid_observations(
        own_positions=env.chaser_position.unsqueeze(0),
        other_positions=env.explorer_position.unsqueeze(0),
        other_visible=visible.unsqueeze(0),
        grid_size=GRID_SIZE,
        device=DEVICE,
    )
    grid_explorer_obs = _build_grid_observations(
        own_positions=env.explorer_position.unsqueeze(0),
        other_positions=env.chaser_position.unsqueeze(0),
        other_visible=visible.unsqueeze(0),
        grid_size=GRID_SIZE,
        device=DEVICE,
    )
    assert torch.equal(grid_chaser_obs[0], env_chaser_obs)
    assert torch.equal(grid_explorer_obs[0], env_explorer_obs)


@pytest.mark.parametrize("architecture", ["rnn", "ssm"])
def test_collect_rollout_fused_matches_unfused(architecture: str) -> None:
    batch_size = 8
    env_config = GridWorldConfig(max_steps=20)
    chaser, explorer = _policy_pair(architecture, seed=5)

    rollouts = {}
    for fused in (False, True):
        config = TrainConfig(
            batch_size=batch_size,
            architecture=architecture,
            hidden_size=HIDDEN_SIZE,
            fused_agent_rollout=fused,
            env=env_config,
        )
        env = BatchedChaseEnv(env_config, batch_size, DEVICE)
        # Same seed aligns the spawn and action-sampling RNG streams, so the
        # trajectories only diverge if the fused forward math does.
        torch.manual_seed(55)
        rollouts[fused] = collect_rollout(
            config=config,
            env=env,
            chaser=chaser,
            explorer=explorer,
            device=DEVICE,
            capture_metrics=False,
        )

    for agent in ("chaser", "explorer"):
        fused_agent = getattr(rollouts[True], agent)
        plain_agent = getattr(rollouts[False], agent)
        assert torch.equal(fused_agent.actions, plain_agent.actions)
        assert torch.equal(fused_agent.observations, plain_agent.observations)
        assert torch.equal(fused_agent.rewards, plain_agent.rewards)
        torch.testing.assert_close(fused_agent.old_logits, plain_agent.old_logits, **TOLERANCE)
        torch.testing.assert_close(
            fused_agent.old_log_probs, plain_agent.old_log_probs, **TOLERANCE
        )
        torch.testing.assert_close(fused_agent.old_values, plain_agent.old_values, **TOLERANCE)
        torch.testing.assert_close(fused_agent.returns, plain_agent.returns, **TOLERANCE)
        # Advantages are standardized, so the std division amplifies rounding.
        torch.testing.assert_close(
            fused_agent.advantages, plain_agent.advantages, atol=1e-4, rtol=1e-4
        )

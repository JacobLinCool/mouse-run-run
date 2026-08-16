from __future__ import annotations

import torch

from mouse_run_run.core.config import PPOConfig
from mouse_run_run.core.simulation import AgentTrajectory, TrajectoryBatch
from mouse_run_run.core.types import AgentId
from mouse_run_run.training.ppo import _advantages


AGENT = AgentId("agent")


def test_gae_stops_at_truncation_and_ignores_bootstrap_after_boundary() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.zeros_like(rewards)
    trajectory = AgentTrajectory(
        observations=torch.zeros(2, 1, 1),
        actions=torch.zeros(2, 1, dtype=torch.long),
        rewards=rewards,
        logits=torch.zeros(2, 1, 2),
        values=values,
        log_probs=torch.zeros(2, 1),
        bootstrap_value=torch.tensor([100.0]),
    )
    rollout = TrajectoryBatch(
        agent_ids=(AGENT,),
        agents={AGENT: trajectory},
        terminated=torch.zeros(2, 1, dtype=torch.bool),
        truncated=torch.tensor([[False], [True]]),
        active=torch.ones(2, 1, dtype=torch.bool),
        world={},
        events={},
        seed=0,
        episode_seeds=torch.tensor([0]),
    )
    advantage, returns = _advantages(
        rollout,
        PPOConfig(gamma=0.9, gae_lambda=1.0),
    )
    assert torch.allclose(advantage[AGENT], torch.tensor([[1.9], [1.0]]))
    assert torch.equal(returns[AGENT], advantage[AGENT])


def test_inactive_steps_do_not_contribute_advantage() -> None:
    trajectory = AgentTrajectory(
        observations=torch.zeros(3, 1, 1),
        actions=torch.zeros(3, 1, dtype=torch.long),
        rewards=torch.tensor([[1.0], [100.0], [100.0]]),
        logits=torch.zeros(3, 1, 2),
        values=torch.zeros(3, 1),
        log_probs=torch.zeros(3, 1),
        bootstrap_value=torch.zeros(1),
    )
    rollout = TrajectoryBatch(
        agent_ids=(AGENT,),
        agents={AGENT: trajectory},
        terminated=torch.tensor([[True], [True], [True]]),
        truncated=torch.zeros(3, 1, dtype=torch.bool),
        active=torch.tensor([[True], [False], [False]]),
        world={},
        events={},
        seed=0,
        episode_seeds=torch.tensor([0]),
    )
    advantage, _ = _advantages(rollout, PPOConfig())
    assert torch.equal(advantage[AGENT], torch.tensor([[1.0], [0.0], [0.0]]))

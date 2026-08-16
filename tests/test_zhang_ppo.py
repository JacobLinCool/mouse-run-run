from __future__ import annotations

import pytest
import torch
from dataclasses import replace

from experiments.chase_grid.experiment import ChaseExperimentConfig, build_experiment
from experiments.chase_grid.model import ModelConfig
from experiments.zhang_2025_rnn.study import GOAL_DIRECTED_PPO, PROTOCOL_PPO
from mouse_run_run.core.config import PPOConfig, TrainingConfig
from mouse_run_run.core.experiment import PolicyBuildContext, RuntimeConfig
from mouse_run_run.core.simulation import SimulationConfig, SimulationEngine
from mouse_run_run.training.ppo import (
    IndependentPPO,
    _epoch_minibatches,
    updated_kl_coefficient,
)


def test_goal_directed_ppo_is_one_full_batch_update_per_agent() -> None:
    assert GOAL_DIRECTED_PPO.learning_rate == 5e-5
    assert GOAL_DIRECTED_PPO.gamma == 0.99
    assert GOAL_DIRECTED_PPO.gae_lambda == 1.0
    assert GOAL_DIRECTED_PPO.epochs == 1
    assert GOAL_DIRECTED_PPO.minibatch_size == 4_000
    assert GOAL_DIRECTED_PPO.sequence_length == 20
    assert GOAL_DIRECTED_PPO.clip_epsilon == 0.3
    assert GOAL_DIRECTED_PPO.value_coefficient == 1.0
    assert GOAL_DIRECTED_PPO.value_clip == 10.0
    assert GOAL_DIRECTED_PPO.entropy_coefficient == 0.0
    assert GOAL_DIRECTED_PPO.initial_kl_coefficient == 0.2
    assert GOAL_DIRECTED_PPO.kl_target == 0.01
    assert GOAL_DIRECTED_PPO.recurrent_l2_coefficient == 0.3
    assert GOAL_DIRECTED_PPO.max_gradient_norm is None
    valid = torch.ones(200, 20, dtype=torch.bool)
    groups = _epoch_minibatches(valid, 4_000, torch.Generator().manual_seed(4))
    assert len(groups) == 1
    assert int(valid[groups[0]].sum()) == 4_000


def test_protocol_ppo_contract_remains_explicit_sensitivity_recipe() -> None:
    assert PROTOCOL_PPO.epochs == 30
    assert PROTOCOL_PPO.minibatch_size == 128
    assert PROTOCOL_PPO.sequence_length == 20
    assert PROTOCOL_PPO.recurrent_l2_coefficient == 3.0


def test_nonoverlapping_tbptt_covers_all_4000_steps_once() -> None:
    valid = torch.ones(200, 20, dtype=torch.bool)
    groups = _epoch_minibatches(valid, 128, torch.Generator().manual_seed(4))
    flattened = torch.cat(groups)
    assert len(groups) == 34
    assert flattened.numel() == 200
    assert flattened.unique().numel() == 200
    assert sum(int(valid[group].sum()) for group in groups) == 4_000
    assert max(int(valid[group].sum()) for group in groups) <= 128


def test_adaptive_kl_matches_ray_threshold_rule() -> None:
    assert updated_kl_coefficient(0.2, sampled_kl=0.03, target=0.01) == pytest.approx(0.3)
    assert updated_kl_coefficient(0.2, sampled_kl=0.004, target=0.01) == 0.1
    assert updated_kl_coefficient(0.2, sampled_kl=0.01, target=0.01) == 0.2


def test_value_error_clip_and_unsquared_recurrent_l2_enter_loss() -> None:
    ppo = PPOConfig(
        learning_rate=1e-12,
        epochs=1,
        minibatch_size=16,
        sequence_length=4,
        value_clip=0.25,
        recurrent_l2_coefficient=0.5,
    )
    experiment = build_experiment(
        ChaseExperimentConfig(
            chaser_model=ModelConfig(hidden_size=4),
            explorer_model=ModelConfig(hidden_size=4),
            training=TrainingConfig(updates=1, horizon=4, ppo=ppo),
        )
    )
    device = torch.device("cpu")
    policies = {
        agent_id: experiment.policy_factories[agent_id](
            PolicyBuildContext(
                agent_id,
                experiment.observation_shapes[agent_id],
                experiment.action_counts[agent_id],
                device,
            )
        )
        for agent_id in experiment.agent_ids
    }
    rollout = SimulationEngine(
        experiment.make_environment(RuntimeConfig(batch_size=2), device), policies
    ).collect(SimulationConfig(horizon=4, seed=3, action_seed=4))
    rollout = replace(
        rollout,
        agents={
            agent_id: replace(
                trajectory,
                rewards=torch.full_like(trajectory.rewards, 100.0),
            )
            for agent_id, trajectory in rollout.agents.items()
        },
    )
    expected_l2 = {
        agent_id: 0.5 * float(policy.recurrent_weight_norm().detach())
        for agent_id, policy in policies.items()
    }
    metrics = IndependentPPO(policies, ppo, seed=9).update(rollout)
    for agent_id, values in metrics.agents.items():
        assert values.value_loss == pytest.approx(0.25)
        assert values.recurrent_l2 == pytest.approx(expected_l2[agent_id])

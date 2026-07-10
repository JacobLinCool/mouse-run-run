from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mouse_run_run.env import GridWorldConfig
from mouse_run_run.health import NonFiniteTrainingError
from mouse_run_run.policy import build_policy
from mouse_run_run.ppo import (
    _rllib_policy_update,
    _rllib_recurrent_minibatch_specs,
    _updated_kl_coeff,
)
from mouse_run_run.serialization import load_checkpoint, load_training_state
from mouse_run_run.train import train
from mouse_run_run.training_config import TrainConfig, apply_preset_defaults
from mouse_run_run.training_types import AgentRollout


def test_official_code_preset_matches_rllib_2_2_defaults() -> None:
    target = SimpleNamespace(
        preset="official_code",
        learning_rate=None,
        gae_lambda=None,
        ppo_epochs=None,
        clip_epsilon=None,
        entropy_coef=None,
        value_coef=None,
        value_clip=None,
        recurrent_l2_coef=None,
        grad_clip=None,
        sgd_minibatch_size=None,
        max_seq_len=None,
        kl_coeff=None,
        kl_target=None,
        learner_mode=None,
        rnn_initialization=None,
    )

    apply_preset_defaults(target)

    assert target.learning_rate == 5e-5
    assert target.gae_lambda == 1.0
    assert target.ppo_epochs == 30
    assert target.clip_epsilon == 0.3
    assert target.entropy_coef == 0.0
    assert target.value_coef == 1.0
    assert target.grad_clip is None
    assert target.sgd_minibatch_size == 128
    assert target.max_seq_len == 20
    assert target.kl_coeff == 0.2
    assert target.learner_mode == "rllib_2_2"
    assert target.rnn_initialization == "pytorch_default"


def test_recurrent_minibatches_match_rllib_unpadded_sequence_slicing() -> None:
    specs = _rllib_recurrent_minibatch_specs((20,) * 200, 128)

    assert len(specs) == 33
    assert specs[:3] == [(0, 7, 8), (6, 13, 8), (12, 19, 8)]
    assert specs[-1] == (192, 199, 8)


def test_rllib_kl_coefficient_adaptation() -> None:
    assert _updated_kl_coeff(0.2, sampled_kl=0.03, target=0.01) == pytest.approx(0.3)
    assert _updated_kl_coeff(0.2, sampled_kl=0.004, target=0.01) == 0.1
    assert _updated_kl_coeff(0.2, sampled_kl=0.01, target=0.01) == 0.2


def test_rllib_policy_loss_matches_reference_formula() -> None:
    torch.manual_seed(4)
    policy = build_policy(
        "rnn",
        input_size=6,
        hidden_size=3,
        rnn_initialization="pytorch_default",
    )
    observations = torch.randn(4, 1, 6)
    state = policy.initial_hidden(1, torch.device("cpu"))
    states = []
    logits = []
    values = []
    for step in range(4):
        states.append(state)
        output = policy(observations[step], state)
        logits.append(output.logits)
        values.append(output.value)
        state = output.state
    old_logits = torch.stack(logits).detach()
    old_values = torch.stack(values).detach()
    actions = torch.tensor([[0], [1], [2], [3]])
    old_action_log_probs = old_logits.log_softmax(dim=-1).gather(
        -1,
        actions.unsqueeze(-1),
    ).squeeze(-1)
    advantages = torch.tensor([[1.0], [-2.0], [3.0], [-4.0]])
    returns = old_values + torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    rollout = AgentRollout(
        observations=observations,
        actions=actions,
        old_log_probs=old_action_log_probs,
        old_logits=old_logits,
        old_values=old_values,
        rewards=torch.zeros_like(old_values),
        advantages=advantages,
        returns=returns,
        state_inputs=torch.stack(states),
    )
    config = TrainConfig(
        batch_size=1,
        hidden_size=3,
        ppo_epochs=1,
        clip_epsilon=0.3,
        entropy_coef=0.0,
        value_coef=1.0,
        value_clip=10.0,
        recurrent_l2_coef=0.0,
        grad_clip=None,
        sgd_minibatch_size=4,
        max_seq_len=2,
        learner_mode="rllib_2_2",
        rnn_initialization="pytorch_default",
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0)

    stats = _rllib_policy_update(
        config=config,
        agent_rollout=rollout,
        policy=policy,
        optimizer=optimizer,
        kl_coeff=0.2,
        minibatch_generator=torch.Generator().manual_seed(0),
        capture_metrics=True,
        policy_name="test",
    )

    expected_entropy = -(
        old_logits.log_softmax(dim=-1).exp() * old_logits.log_softmax(dim=-1)
    ).sum(dim=-1).mean()
    expected_value_loss = (old_values - returns).square().clamp(max=10.0).mean()
    assert stats.policy_loss == pytest.approx(-advantages.mean().item(), abs=1e-6)
    assert stats.value_loss == pytest.approx(expected_value_loss.item(), abs=1e-6)
    assert stats.entropy == pytest.approx(expected_entropy.item(), abs=1e-6)
    assert stats.kl == pytest.approx(0.0, abs=1e-7)


def test_rllib_device_accumulated_guard_rejects_nonfinite_minibatch() -> None:
    policy = build_policy(
        "rnn",
        input_size=2,
        hidden_size=2,
        rnn_initialization="pytorch_default",
    )
    old_logits = torch.zeros(2, 1, 4)
    rollout = AgentRollout(
        observations=torch.zeros(2, 1, 2),
        actions=torch.zeros(2, 1, dtype=torch.long),
        old_log_probs=torch.full((2, 1), -torch.log(torch.tensor(4.0))),
        old_logits=old_logits,
        old_values=torch.zeros(2, 1),
        rewards=torch.zeros(2, 1),
        advantages=torch.full((2, 1), float("nan")),
        returns=torch.zeros(2, 1),
        state_inputs=torch.zeros(2, 1, 2),
    )
    config = TrainConfig(
        batch_size=1,
        hidden_size=2,
        ppo_epochs=1,
        recurrent_l2_coef=0.0,
        grad_clip=None,
        sgd_minibatch_size=2,
        max_seq_len=1,
        learner_mode="rllib_2_2",
        rnn_initialization="pytorch_default",
        finite_guard=True,
    )

    with pytest.raises(NonFiniteTrainingError, match="accumulated_loss_or_gradient"):
        _rllib_policy_update(
            config=config,
            agent_rollout=rollout,
            policy=policy,
            optimizer=torch.optim.Adam(policy.parameters(), lr=0.0),
            kl_coeff=0.2,
            minibatch_generator=torch.Generator().manual_seed(0),
            capture_metrics=True,
            policy_name="test",
        )


def test_tiny_official_learner_run_persists_independent_policy_state(tmp_path) -> None:
    checkpoint = tmp_path / "official.safetensors"

    metrics = train(
        TrainConfig(
            updates=1,
            batch_size=2,
            hidden_size=8,
            gae_lambda=1.0,
            learning_rate=5e-5,
            ppo_epochs=2,
            clip_epsilon=0.3,
            entropy_coef=0.0,
            value_coef=1.0,
            value_clip=10.0,
            recurrent_l2_coef=3.0,
            grad_clip=None,
            sgd_minibatch_size=4,
            max_seq_len=2,
            kl_coeff=0.2,
            kl_target=0.01,
            learner_mode="rllib_2_2",
            rnn_initialization="pytorch_default",
            device="cpu",
            checkpoint=checkpoint,
            log_every=1,
            env=GridWorldConfig(
                grid_size=5,
                max_steps=4,
                spawn_mode="official_exclude_last",
            ),
        )
    )

    state = load_training_state(checkpoint)
    assert state is not None
    assert set(state.optimizer_states) == {"chaser", "explorer"}
    assert state.optimizer_states["chaser"]["state"]
    assert state.optimizer_states["explorer"]["state"]
    assert metrics.chaser_kl_coeff > 0
    assert metrics.explorer_kl_coeff > 0


def test_official_learner_resume_is_exact_on_cpu(tmp_path) -> None:
    base = TrainConfig(
        updates=2,
        batch_size=2,
        hidden_size=8,
        gae_lambda=1.0,
        learning_rate=5e-5,
        ppo_epochs=2,
        clip_epsilon=0.3,
        entropy_coef=0.0,
        value_coef=1.0,
        value_clip=10.0,
        recurrent_l2_coef=3.0,
        grad_clip=None,
        sgd_minibatch_size=4,
        max_seq_len=2,
        kl_coeff=0.2,
        kl_target=0.01,
        learner_mode="rllib_2_2",
        rnn_initialization="pytorch_default",
        seed=11,
        device="cpu",
        log_every=1,
        env=GridWorldConfig(grid_size=5, max_steps=4),
    )
    direct_path = tmp_path / "direct.safetensors"
    phase_one_path = tmp_path / "phase-one.safetensors"
    resumed_path = tmp_path / "resumed.safetensors"

    train(replace(base, checkpoint=direct_path))
    train(replace(base, updates=1, checkpoint=phase_one_path))
    train(
        replace(
            base,
            checkpoint=resumed_path,
            resume_from=phase_one_path,
        )
    )

    _, _, direct_chaser, direct_explorer = load_checkpoint(direct_path)
    _, _, resumed_chaser, resumed_explorer = load_checkpoint(resumed_path)
    for key in direct_chaser:
        assert torch.equal(direct_chaser[key], resumed_chaser[key])
    for key in direct_explorer:
        assert torch.equal(direct_explorer[key], resumed_explorer[key])
    direct_state = load_training_state(direct_path)
    resumed_state = load_training_state(resumed_path)
    assert direct_state is not None and resumed_state is not None
    assert direct_state.learner_state == resumed_state.learner_state
    assert torch.equal(direct_state.cpu_rng_state, resumed_state.cpu_rng_state)
    assert torch.equal(
        direct_state.minibatch_rng_state,
        resumed_state.minibatch_rng_state,
    )

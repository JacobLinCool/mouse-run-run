"""Math-level pinning tests for the PPO training core.

Covers pieces the integration tests never check directly:

- ``gae``: hand-derived advantages/returns for gae_lambda in {0.95, 1.0}
  and done-flag masking of the bootstrap/advantage recursion.
- ``normalize_advantages``: the RLlib ``min_std=1e-4`` clamp branch vs the
  default ``std + 1e-8`` path.
- The full-batch learner behind the modern_fast/paper_text presets
  (``ppo_update`` with ``learner_mode="full_batch"``): loss composition,
  clip engagement, value coefficient, entropy sign, and descent on the
  composed objective.
- The rllib_2_2 learner: the KL-penalty term engages exactly when
  configured, and padded positions of unequal-length sequences contribute
  zero loss and zero gradient.

All randomness is seeded; everything runs on CPU.
"""

import copy

import pytest
import torch

from mouse_run_run.policy import PolicyBase, build_policy
from mouse_run_run.ppo import _rllib_policy_update, ppo_update
from mouse_run_run.training_config import TRAINING_PRESETS, TrainConfig
from mouse_run_run.training_rollout import gae, normalize_advantages
from mouse_run_run.training_types import (
    AgentRollout,
    LearnerState,
    Rollout,
    empty_metrics,
)

ACTIONS = 4


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
#
# gamma = 0.9. The implementation bootstraps from the stored V(s_{t+1})
# inside the horizon and from 0 past the horizon (next_value starts at 0),
# and a done flag at step t zeroes both the bootstrap term and the
# advantage recursion:
#
#     delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
#     A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
#
# With r = [1.0, 0.5, 2.0, 1.0], V = [0.5, 1.0, 1.5, 2.0],
# dones = [F, T, F, F]:
#
#     t=3: delta = 1.0 + 0.9*0   - 2.0 = -1.0   A3 = -1.0
#     t=2: delta = 2.0 + 0.9*2.0 - 1.5 =  2.3   A2 = 2.3 + 0.9*lam*(-1.0)
#     t=1: done => delta = 0.5 - 1.0   = -0.5   A1 = -0.5  (recursion cut)
#     t=0: delta = 1.0 + 0.9*1.0 - 0.5 =  1.4   A0 = 1.4 + 0.9*lam*(-0.5)
#
#     lam = 1.00: A = [0.95,   -0.5, 1.4,   -1.0]
#     lam = 0.95: A = [0.9725, -0.5, 1.445, -1.0]
#
# and returns = A + V in both cases. As a cross-check for lam = 1.0, the
# episode segment at steps 2..3 telescopes to the plain discounted sum:
# R2 = r2 + gamma*r3 = 2.0 + 0.9 = 2.9.
@pytest.mark.parametrize(
    ("gae_lambda", "expected_advantages"),
    [
        (1.0, [0.95, -0.5, 1.4, -1.0]),
        (0.95, [0.9725, -0.5, 1.445, -1.0]),
    ],
)
def test_gae_matches_hand_derived_values(
    gae_lambda: float,
    expected_advantages: list[float],
) -> None:
    rewards = torch.tensor([[1.0], [0.5], [2.0], [1.0]])
    values = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
    dones = torch.tensor([[False], [True], [False], [False]])

    advantages, returns = gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.9,
        gae_lambda=gae_lambda,
    )

    expected = torch.tensor(expected_advantages).unsqueeze(1)
    torch.testing.assert_close(advantages, expected, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(returns, expected + values, rtol=0.0, atol=1e-6)


def test_gae_done_flag_stops_bootstrap_propagation() -> None:
    """Steps before an episode boundary must ignore everything after it."""
    rewards = torch.tensor([[0.1], [-0.2], [0.3], [0.4]])
    values = torch.tensor([[0.5], [0.6], [0.7], [0.8]])
    tail_rewards = rewards.clone()
    tail_rewards[2:] = torch.tensor([[9.0], [-3.0]])
    tail_values = values.clone()
    tail_values[2:] = torch.tensor([[-5.0], [2.0]])
    dones = torch.tensor([[False], [True], [False], [False]])

    base_adv, base_ret = gae(
        rewards=rewards, values=values, dones=dones, gamma=0.99, gae_lambda=0.95
    )
    tail_adv, tail_ret = gae(
        rewards=tail_rewards, values=tail_values, dones=dones, gamma=0.99, gae_lambda=0.95
    )

    # Pre-boundary steps are bitwise identical, post-boundary steps differ.
    assert torch.equal(base_adv[:2], tail_adv[:2])
    assert torch.equal(base_ret[:2], tail_ret[:2])
    assert not torch.allclose(base_adv[2:], tail_adv[2:])

    # Sanity: with the done flag removed, the tail does leak backwards, so
    # the equality above is really the done mask at work.
    no_dones = torch.zeros_like(dones)
    open_adv, _ = gae(
        rewards=rewards, values=values, dones=no_dones, gamma=0.99, gae_lambda=0.95
    )
    open_tail_adv, _ = gae(
        rewards=tail_rewards, values=tail_values, dones=no_dones, gamma=0.99, gae_lambda=0.95
    )
    assert not torch.allclose(open_adv[:2], open_tail_adv[:2])


# ---------------------------------------------------------------------------
# Advantage normalization
# ---------------------------------------------------------------------------
def test_normalize_advantages_default_path_uses_biased_std_plus_epsilon() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])

    normalized = normalize_advantages(values)

    expected = (values - 2.5) / (torch.tensor(1.25).sqrt() + 1e-8)
    torch.testing.assert_close(normalized, expected, rtol=1e-6, atol=1e-7)


def test_normalize_advantages_clamps_tiny_std_to_rllib_min_std() -> None:
    # Biased std of [0, 2e-5] is 1e-5 < min_std, so the denominator clamps
    # to 1e-4 and the output is (+-1e-5) / 1e-4 = +-0.1 instead of +-1.0.
    values = torch.tensor([0.0, 2e-5])

    normalized = normalize_advantages(values, min_std=1e-4)

    torch.testing.assert_close(
        normalized,
        torch.tensor([-0.1, 0.1]),
        rtol=1e-5,
        atol=1e-7,
    )


def test_normalize_advantages_min_std_leaves_normal_spread_untouched() -> None:
    # Biased std of [0, 2] is exactly 1.0 > min_std: no clamp, plain z-score.
    values = torch.tensor([0.0, 2.0])

    normalized = normalize_advantages(values, min_std=1e-4)

    torch.testing.assert_close(normalized, torch.tensor([-1.0, 1.0]), rtol=0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Full-batch learner (modern_fast / paper_text presets)
# ---------------------------------------------------------------------------
def test_modern_presets_use_the_full_batch_learner() -> None:
    assert TRAINING_PRESETS["modern_fast"].learner_mode == "full_batch"
    assert TRAINING_PRESETS["paper_text"].learner_mode == "full_batch"


def _evaluate(
    policy: PolicyBase,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference re-derivation of the full-batch action evaluation."""
    logits, values, _ = policy.sequence(observations)
    log_probs = logits.log_softmax(dim=-1)
    action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return action_log_probs, values, entropy


def _full_batch_rollout(
    chaser: PolicyBase,
    explorer: PolicyBase,
    *,
    steps: int,
    batch: int,
    obs_size: int,
    neutral: bool = False,
) -> Rollout:
    """Synthetic rollout at the policies' current parameters.

    Default: stored old_log_probs are offset by +-0.5 from the current log
    probs so the first epoch sees ratios exp(+-0.5) ~ {1.65, 0.61}, well
    outside a 0.2 clip band. ``neutral=True`` instead zeroes advantages and
    sets returns to the current values so only the entropy bonus (or a KL
    penalty) can produce gradients.
    """
    observations = torch.randn(steps, batch, obs_size)
    actions = torch.randint(0, ACTIONS, (steps, batch))
    offsets = 0.5 * (-1.0) ** torch.arange(steps * batch, dtype=torch.float32)
    offsets = offsets.reshape(steps, batch)
    agents = []
    for policy in (chaser, explorer):
        with torch.no_grad():
            log_probs, values, _ = _evaluate(policy, observations, actions)
        if neutral:
            old_log_probs = log_probs.clone()
            advantages = torch.zeros(steps, batch)
            returns = values.clone()
        else:
            old_log_probs = log_probs - offsets
            advantages = torch.randn(steps, batch) * 1.5
            returns = values + torch.randn(steps, batch) * 1.5
        agents.append(
            AgentRollout(
                observations=observations,
                actions=actions,
                old_log_probs=old_log_probs,
                old_logits=torch.zeros(steps, batch, ACTIONS),
                old_values=values.clone(),
                rewards=torch.zeros(steps, batch),
                advantages=advantages,
                returns=returns,
            )
        )
    return Rollout(chaser=agents[0], explorer=agents[1], metrics=empty_metrics())


def _full_batch_reference(
    config: TrainConfig,
    rollout: Rollout,
    chaser: PolicyBase,
    explorer: PolicyBase,
) -> dict[str, torch.Tensor]:
    """Independent recomputation of the full-batch composed loss."""
    policy_losses = []
    unclipped_losses = []
    value_errors = []
    entropies = []
    for policy, agent in ((chaser, rollout.chaser), (explorer, rollout.explorer)):
        log_prob, value, entropy = _evaluate(policy, agent.observations, agent.actions)
        ratio = (log_prob - agent.old_log_probs).exp()
        clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        policy_losses.append(
            -torch.minimum(ratio * agent.advantages, clipped_ratio * agent.advantages).mean()
        )
        unclipped_losses.append(-(ratio * agent.advantages).mean())
        error = (value - agent.returns).square()
        if config.value_clip > 0:
            error = error.clamp(max=config.value_clip)
        value_errors.append(error.mean())
        entropies.append(entropy.mean())
    policy_loss = policy_losses[0] + policy_losses[1]
    value_loss = 0.5 * (value_errors[0] + value_errors[1])
    entropy = entropies[0] + entropies[1]
    recurrent_l2 = config.recurrent_l2_coef * (
        chaser.recurrent_weight_norm() + explorer.recurrent_weight_norm()
    )
    loss = (
        policy_loss
        + config.value_coef * value_loss
        - config.entropy_coef * entropy
        + recurrent_l2
    )
    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "unclipped_policy_loss": unclipped_losses[0] + unclipped_losses[1],
        "value_loss": value_loss,
        "entropy": entropy,
    }


def _run_full_batch_update(
    config: TrainConfig,
    rollout: Rollout,
    chaser: PolicyBase,
    explorer: PolicyBase,
    *,
    lr: float,
):
    return ppo_update(
        config=config,
        rollout=rollout,
        chaser=chaser,
        explorer=explorer,
        chaser_optimizer=torch.optim.SGD(chaser.parameters(), lr=lr),
        explorer_optimizer=torch.optim.SGD(explorer.parameters(), lr=lr),
        learner_state=LearnerState(
            chaser_kl_coeff=config.kl_coeff,
            explorer_kl_coeff=config.kl_coeff,
        ),
        minibatch_generator=torch.Generator().manual_seed(0),
        capture_metrics=True,
    )


def _full_batch_config(**overrides) -> TrainConfig:
    base: dict = dict(
        batch_size=3,
        hidden_size=6,
        ppo_epochs=1,
        clip_epsilon=0.2,
        entropy_coef=0.05,
        value_coef=0.7,
        value_clip=2.0,
        recurrent_l2_coef=0.3,
        grad_clip=None,
        learner_mode="full_batch",
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_full_batch_loss_composition_and_gradients_match_reference() -> None:
    torch.manual_seed(11)
    chaser = build_policy("rnn", input_size=5, hidden_size=6)
    explorer = build_policy("rnn", input_size=5, hidden_size=6)
    rollout = _full_batch_rollout(chaser, explorer, steps=6, batch=3, obs_size=5)
    config = _full_batch_config()
    ref_chaser = copy.deepcopy(chaser)
    ref_explorer = copy.deepcopy(explorer)

    metrics = _run_full_batch_update(config, rollout, chaser, explorer, lr=0.0)

    assert metrics is not None
    reference = _full_batch_reference(config, rollout, ref_chaser, ref_explorer)

    # Sanity: the constructed batch really exercises both clip directions
    # and the value_clip clamp, so the assertions below are non-vacuous.
    for agent in (rollout.chaser, rollout.explorer):
        with torch.no_grad():
            log_prob, value, _ = _evaluate(ref_chaser, agent.observations, agent.actions)
        ratio = (log_prob - agent.old_log_probs).exp()
        assert ((ratio > 1.2) & (agent.advantages > 0)).any()
        assert ((ratio < 0.8) & (agent.advantages < 0)).any()
    with torch.no_grad():
        _, chaser_value, _ = _evaluate(ref_chaser, rollout.chaser.observations, rollout.chaser.actions)
    assert ((chaser_value - rollout.chaser.returns).square() > config.value_clip).any()

    # Reported components are the unweighted pieces of the composed loss,
    # and the policy loss is the clipped surrogate, not the raw one.
    assert metrics.policy_loss == pytest.approx(reference["policy_loss"].item(), abs=1e-5)
    assert metrics.value_loss == pytest.approx(reference["value_loss"].item(), abs=1e-5)
    assert metrics.entropy == pytest.approx(reference["entropy"].item(), abs=1e-5)
    assert abs(metrics.policy_loss - reference["unclipped_policy_loss"].item()) > 1e-3
    expected_kl = 0.5 * (
        (rollout.chaser.old_log_probs - _evaluate(ref_chaser, rollout.chaser.observations, rollout.chaser.actions)[0]).mean()
        + (rollout.explorer.old_log_probs - _evaluate(ref_explorer, rollout.explorer.observations, rollout.explorer.actions)[0]).mean()
    )
    assert metrics.approx_kl == pytest.approx(expected_kl.item(), abs=1e-5)

    # Gradient composition: backward of the independently composed loss
    # (clip + value_coef * value + (-entropy_coef) * entropy + recurrent L2)
    # must match the production gradients left in place by the lr=0 update.
    reference_grad = _full_batch_reference(config, rollout, ref_chaser, ref_explorer)
    reference_grad["loss"].backward()
    for policy, ref_policy in ((chaser, ref_chaser), (explorer, ref_explorer)):
        for (name, parameter), (ref_name, ref_parameter) in zip(
            policy.named_parameters(), ref_policy.named_parameters()
        ):
            assert name == ref_name
            assert parameter.grad is not None and ref_parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, rtol=1e-5, atol=1e-7
            )


def test_full_batch_update_step_decreases_composed_loss() -> None:
    torch.manual_seed(23)
    chaser = build_policy("rnn", input_size=5, hidden_size=6)
    explorer = build_policy("rnn", input_size=5, hidden_size=6)
    rollout = _full_batch_rollout(chaser, explorer, steps=6, batch=3, obs_size=5)
    config = _full_batch_config()

    with torch.no_grad():
        loss_before = _full_batch_reference(config, rollout, chaser, explorer)["loss"].item()
    _run_full_batch_update(config, rollout, chaser, explorer, lr=1e-3)
    with torch.no_grad():
        loss_after = _full_batch_reference(config, rollout, chaser, explorer)["loss"].item()

    assert loss_after < loss_before


def test_full_batch_entropy_bonus_increases_policy_entropy() -> None:
    """The entropy term must be a bonus (subtracted from the loss).

    With zero advantages and returns equal to the current values, the
    surrogate and value terms have exactly zero gradient, so a single SGD
    step moves parameters purely along +entropy_coef * d(entropy)/d(theta).
    """
    torch.manual_seed(31)
    chaser = build_policy("rnn", input_size=5, hidden_size=6)
    explorer = build_policy("rnn", input_size=5, hidden_size=6)
    rollout = _full_batch_rollout(chaser, explorer, steps=6, batch=3, obs_size=5, neutral=True)
    config = _full_batch_config(entropy_coef=0.05, recurrent_l2_coef=0.0, value_clip=0.0)

    def mean_entropy() -> float:
        with torch.no_grad():
            chaser_entropy = _evaluate(chaser, rollout.chaser.observations, rollout.chaser.actions)[2]
            explorer_entropy = _evaluate(explorer, rollout.explorer.observations, rollout.explorer.actions)[2]
        return (chaser_entropy.mean() + explorer_entropy.mean()).item()

    entropy_before = mean_entropy()
    before = [parameter.detach().clone() for parameter in chaser.parameters()]
    _run_full_batch_update(config, rollout, chaser, explorer, lr=0.05)

    moved = max(
        (parameter.detach() - old).abs().max().item()
        for parameter, old in zip(chaser.parameters(), before)
    )
    assert moved > 0.0
    assert mean_entropy() > entropy_before


# ---------------------------------------------------------------------------
# Recurrent (rllib_2_2) learner blind spots
# ---------------------------------------------------------------------------
def _rllib_config(**overrides) -> TrainConfig:
    base: dict = dict(
        batch_size=2,
        hidden_size=3,
        ppo_epochs=1,
        clip_epsilon=0.3,
        entropy_coef=0.0,
        value_coef=1.0,
        value_clip=10.0,
        recurrent_l2_coef=0.0,
        grad_clip=None,
        learner_mode="rllib_2_2",
        rnn_initialization="pytorch_default",
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_rllib_kl_penalty_engages_only_when_configured() -> None:
    """kl_coeff gates the KL penalty; every other term has zero gradient.

    Advantages are zero, returns equal the exact padded-evaluation values,
    and entropy/L2 coefficients are zero, so the only possible gradient
    source is kl_coeff * KL(old_logits || policy).
    """
    torch.manual_seed(41)
    policy = build_policy("rnn", input_size=4, hidden_size=3, rnn_initialization="pytorch_default")
    steps, episodes = 4, 2
    observations = torch.randn(steps, episodes, 4)
    with torch.no_grad():
        # Same teacher-forced path _evaluate_recurrent_sequences takes for a
        # single full-length chunk with a zero initial state.
        hidden, _ = policy.rnn(observations, torch.zeros(1, episodes, 3))
        logits = policy.action_layer(hidden)
        values = policy.value_layer(hidden).squeeze(-1)
    actions = torch.randint(0, ACTIONS, (steps, episodes))
    log_probs = logits.log_softmax(dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    rollout = AgentRollout(
        observations=observations,
        actions=actions,
        old_log_probs=log_probs,
        # Old distribution = uniform, different from the policy => KL > 0.
        old_logits=torch.zeros(steps, episodes, ACTIONS),
        old_values=values.clone(),
        rewards=torch.zeros(steps, episodes),
        advantages=torch.zeros(steps, episodes),
        returns=values.clone(),
        state_inputs=torch.zeros(steps, episodes, 3),
    )
    config = _rllib_config(sgd_minibatch_size=8, max_seq_len=4)

    def kl_to_uniform(candidate) -> float:
        with torch.no_grad():
            hidden_now, _ = candidate.rnn(observations, torch.zeros(1, episodes, 3))
            new_log_probs = candidate.action_layer(hidden_now).log_softmax(dim=-1)
        uniform_log_prob = torch.zeros(steps, episodes, ACTIONS).log_softmax(dim=-1)
        return (
            (uniform_log_prob.exp() * (uniform_log_prob - new_log_probs)).sum(dim=-1).mean().item()
        )

    def run(kl_coeff: float):
        candidate = copy.deepcopy(policy)
        stats = _rllib_policy_update(
            config=config,
            agent_rollout=rollout,
            policy=candidate,
            optimizer=torch.optim.SGD(candidate.parameters(), lr=0.1),
            kl_coeff=kl_coeff,
            minibatch_generator=torch.Generator().manual_seed(0),
            capture_metrics=True,
            policy_name="test",
        )
        delta = max(
            (parameter.detach() - reference.detach()).abs().max().item()
            for parameter, reference in zip(candidate.parameters(), policy.parameters())
        )
        return stats, delta, candidate

    stats_off, delta_off, _ = run(kl_coeff=0.0)
    stats_on, delta_on, updated = run(kl_coeff=0.2)

    assert stats_off.kl > 0.0 and stats_on.kl > 0.0
    # Without the penalty nothing moves; with it, parameters move and the
    # step descends on the KL to the old distribution.
    assert delta_off < 1e-8
    assert delta_on > 1e-6
    assert kl_to_uniform(updated) < kl_to_uniform(policy)


def test_rllib_padded_positions_contribute_zero_loss_and_gradient() -> None:
    """Unequal-length sequences: padding is invisible to loss and gradients.

    steps=5 with max_seq_len=4 chunks each episode into a 4-step and a
    1-step sequence; the second chunk carries 3 padded positions. The
    reference below never materializes padded steps at all: it steps the
    policy over the 10 real positions only (restarting from the recorded
    hidden state at each chunk boundary, i.e. truncated BPTT) and averages
    over those 10. If any padded position leaked into a mean or a gradient,
    the stats and parameter gradients could not match it.
    """
    torch.manual_seed(53)
    policy = build_policy("rnn", input_size=4, hidden_size=3, rnn_initialization="pytorch_default")
    steps, episodes, max_seq_len = 5, 2, 4
    observations = torch.randn(steps, episodes, 4)
    actions = torch.randint(0, ACTIONS, (steps, episodes))

    state_inputs = torch.empty(steps, episodes, 3)
    logits_steps = []
    value_steps = []
    with torch.no_grad():
        hidden = policy.initial_hidden(episodes, torch.device("cpu"))
        for step in range(steps):
            state_inputs[step] = hidden
            output = policy(observations[step], hidden)
            logits_steps.append(output.logits)
            value_steps.append(output.value)
            hidden = output.state
    logits = torch.stack(logits_steps)
    values = torch.stack(value_steps)
    log_probs = logits.log_softmax(dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    offsets = 0.5 * (-1.0) ** torch.arange(steps * episodes, dtype=torch.float32)
    offsets = offsets.reshape(steps, episodes)
    rollout = AgentRollout(
        observations=observations,
        actions=actions,
        old_log_probs=log_probs - offsets,
        old_logits=logits + 0.3 * torch.randn(steps, episodes, ACTIONS),
        old_values=values.clone(),
        rewards=torch.zeros(steps, episodes),
        advantages=torch.randn(steps, episodes),
        returns=values + torch.randn(steps, episodes),
        state_inputs=state_inputs,
    )
    kl_coeff = 0.2
    # sequence_lengths become (4, 1, 4, 1); sgd_minibatch_size=10 collects
    # all four chunks (all 10 real steps) into a single minibatch.
    config = _rllib_config(
        entropy_coef=0.01,
        recurrent_l2_coef=3.0,
        sgd_minibatch_size=10,
        max_seq_len=max_seq_len,
    )
    reference_policy = copy.deepcopy(policy)

    stats = _rllib_policy_update(
        config=config,
        agent_rollout=rollout,
        policy=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.0),
        kl_coeff=kl_coeff,
        minibatch_generator=torch.Generator().manual_seed(0),
        capture_metrics=True,
        policy_name="test",
    )

    # Reference over the 10 real positions only.
    ref_logits_steps = []
    ref_value_steps = []
    hidden = None
    for step in range(steps):
        if step % max_seq_len == 0:
            # Chunk boundary: restart from the recorded (gradient-free)
            # state, exactly like the padded evaluation's initial_states.
            hidden = state_inputs[step]
        output = reference_policy(observations[step], hidden)
        ref_logits_steps.append(output.logits)
        ref_value_steps.append(output.value)
        hidden = output.state
    ref_logits = torch.stack(ref_logits_steps)
    ref_values = torch.stack(ref_value_steps)
    new_log_probs = ref_logits.log_softmax(dim=-1)
    action_log_probs = new_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    ratio = (action_log_probs - rollout.old_log_probs).exp()
    surrogate = torch.minimum(
        rollout.advantages * ratio,
        rollout.advantages * ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon),
    )
    assert not torch.allclose(surrogate, rollout.advantages * ratio)  # clip binds
    policy_loss = -surrogate.mean()
    value_loss = (ref_values - rollout.returns).square().clamp(0.0, config.value_clip).mean()
    old_dist_log_probs = rollout.old_logits.log_softmax(dim=-1)
    mean_kl = (
        (old_dist_log_probs.exp() * (old_dist_log_probs - new_log_probs)).sum(dim=-1).mean()
    )
    entropy = -(new_log_probs.exp() * new_log_probs).sum(dim=-1).mean()
    loss = (
        policy_loss
        + config.value_coef * value_loss
        - config.entropy_coef * entropy
        + kl_coeff * mean_kl
        + config.recurrent_l2_coef * reference_policy.recurrent_weight_norm()
    )

    assert stats.policy_loss == pytest.approx(policy_loss.item(), abs=1e-5)
    assert stats.value_loss == pytest.approx(value_loss.item(), abs=1e-5)
    assert stats.entropy == pytest.approx(entropy.item(), abs=1e-5)
    assert stats.kl == pytest.approx(mean_kl.item(), abs=1e-5)

    loss.backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        policy.named_parameters(), reference_policy.named_parameters()
    ):
        assert name == ref_name
        assert parameter.grad is not None and ref_parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, rtol=1e-4, atol=1e-6
        )

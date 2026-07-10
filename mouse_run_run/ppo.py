from dataclasses import dataclass

import torch
from torch import nn

from mouse_run_run.health import (
    assert_finite_module,
    assert_finite_tensor,
    assert_finite_tensors,
    module_max_abs,
)
from mouse_run_run.policy import PolicyBase, RNNActorCritic
from mouse_run_run.training_config import TrainConfig
from mouse_run_run.training_types import AgentRollout, LearnerState, Rollout, RolloutMetrics


# Upstream source IDs in comments resolve to fixed commit/tag permalinks in
# experiments/paper_marl_official_dynamics_2026/SPEC.md,
# "Implementation Source Registry".
@dataclass(frozen=True)
class _RecurrentSequenceBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_logits: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    initial_states: torch.Tensor
    valid: torch.Tensor
    sequence_lengths: tuple[int, ...]


@dataclass(frozen=True)
class _PolicyUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    grad_norm: float


def ppo_update(
    *,
    config: TrainConfig,
    rollout: Rollout,
    chaser: PolicyBase,
    explorer: PolicyBase,
    chaser_optimizer: torch.optim.Optimizer,
    explorer_optimizer: torch.optim.Optimizer,
    learner_state: LearnerState,
    minibatch_generator: torch.Generator,
    capture_metrics: bool,
) -> RolloutMetrics | None:
    if config.learner_mode == "rllib_2_2":
        result = _rllib_ppo_update(
            config=config,
            rollout=rollout,
            chaser=chaser,
            explorer=explorer,
            chaser_optimizer=chaser_optimizer,
            explorer_optimizer=explorer_optimizer,
            learner_state=learner_state,
            minibatch_generator=minibatch_generator,
            capture_metrics=capture_metrics,
        )
    elif config.learner_mode == "full_batch":
        result = _full_batch_ppo_update(
            config=config,
            rollout=rollout,
            chaser=chaser,
            explorer=explorer,
            chaser_optimizer=chaser_optimizer,
            explorer_optimizer=explorer_optimizer,
            learner_state=learner_state,
            capture_metrics=capture_metrics,
        )
    else:  # pragma: no cover - TrainConfig's Literal prevents this in typed callers.
        raise ValueError(f"Unsupported learner mode: {config.learner_mode}")

    if config.finite_guard:
        assert_finite_module(chaser, location="ppo_update", prefix="chaser")
        assert_finite_module(explorer, location="ppo_update", prefix="explorer")
    return result


def _full_batch_ppo_update(
    *,
    config: TrainConfig,
    rollout: Rollout,
    chaser: PolicyBase,
    explorer: PolicyBase,
    chaser_optimizer: torch.optim.Optimizer,
    explorer_optimizer: torch.optim.Optimizer,
    learner_state: LearnerState,
    capture_metrics: bool,
) -> RolloutMetrics | None:
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    chaser_grad_norms: list[float] = []
    explorer_grad_norms: list[float] = []
    chaser_parameters = list(chaser.parameters())
    explorer_parameters = list(explorer.parameters())

    for ppo_epoch in range(config.ppo_epochs):
        chaser_log_prob, chaser_value, chaser_entropy = _evaluate_actions(
            chaser,
            rollout.chaser.observations,
            rollout.chaser.actions,
        )
        explorer_log_prob, explorer_value, explorer_entropy = _evaluate_actions(
            explorer,
            rollout.explorer.observations,
            rollout.explorer.actions,
        )

        chaser_policy_loss = _clipped_policy_loss(
            log_prob=chaser_log_prob,
            old_log_prob=rollout.chaser.old_log_probs,
            advantage=rollout.chaser.advantages,
            clip_epsilon=config.clip_epsilon,
        )
        explorer_policy_loss = _clipped_policy_loss(
            log_prob=explorer_log_prob,
            old_log_prob=rollout.explorer.old_log_probs,
            advantage=rollout.explorer.advantages,
            clip_epsilon=config.clip_epsilon,
        )
        chaser_value_error = (chaser_value - rollout.chaser.returns).square()
        explorer_value_error = (explorer_value - rollout.explorer.returns).square()
        if config.value_clip > 0:
            chaser_value_error = chaser_value_error.clamp(max=config.value_clip)
            explorer_value_error = explorer_value_error.clamp(max=config.value_clip)
        value_loss = 0.5 * (chaser_value_error.mean() + explorer_value_error.mean())
        entropy = chaser_entropy.mean() + explorer_entropy.mean()
        policy_loss = chaser_policy_loss + explorer_policy_loss
        recurrent_l2 = config.recurrent_l2_coef * (
            chaser.recurrent_weight_norm() + explorer.recurrent_weight_norm()
        )
        loss = (
            policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy
            + recurrent_l2
        )
        if config.finite_guard:
            assert_finite_tensors(
                {
                    "chaser_log_prob": chaser_log_prob,
                    "explorer_log_prob": explorer_log_prob,
                    "chaser_value": chaser_value,
                    "explorer_value": explorer_value,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "recurrent_l2": recurrent_l2,
                    "loss": loss,
                },
                location=f"ppo_epoch_{ppo_epoch}",
            )

        chaser_optimizer.zero_grad(set_to_none=True)
        explorer_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        chaser_grad_norm = _gradient_norm(
            chaser_parameters,
            max_norm=config.grad_clip,
            error_if_nonfinite=config.finite_guard,
        )
        explorer_grad_norm = _gradient_norm(
            explorer_parameters,
            max_norm=config.grad_clip,
            error_if_nonfinite=config.finite_guard,
        )
        chaser_optimizer.step()
        explorer_optimizer.step()

        if capture_metrics:
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            chaser_grad_norms.append(float(chaser_grad_norm.item()))
            explorer_grad_norms.append(float(explorer_grad_norm.item()))
            with torch.no_grad():
                chaser_kl = rollout.chaser.old_log_probs - chaser_log_prob
                explorer_kl = rollout.explorer.old_log_probs - explorer_log_prob
                approx_kls.append(
                    0.5 * (chaser_kl.mean().item() + explorer_kl.mean().item())
                )

    if not capture_metrics:
        return None
    return _metrics_from_updates(
        rollout=rollout,
        chaser=chaser,
        explorer=explorer,
        learner_state=learner_state,
        policy_loss=_mean(policy_losses),
        value_loss=_mean(value_losses),
        entropy=_mean(entropies),
        kl=_mean(approx_kls),
        chaser_grad_norm=_mean(chaser_grad_norms),
        explorer_grad_norm=_mean(explorer_grad_norms),
    )


def _rllib_ppo_update(
    *,
    config: TrainConfig,
    rollout: Rollout,
    chaser: PolicyBase,
    explorer: PolicyBase,
    chaser_optimizer: torch.optim.Optimizer,
    explorer_optimizer: torch.optim.Optimizer,
    learner_state: LearnerState,
    minibatch_generator: torch.Generator,
    capture_metrics: bool,
) -> RolloutMetrics | None:
    if not isinstance(chaser, RNNActorCritic) or not isinstance(explorer, RNNActorCritic):
        raise ValueError("rllib_2_2 learner mode requires RNNActorCritic policies")
    if config.sgd_minibatch_size <= config.max_seq_len:
        raise ValueError("sgd_minibatch_size must be larger than max_seq_len")

    # [OFFICIAL-TRAIN], [RAY-SGD-SLICING]: RLlib iterates the released policy
    # dict in insertion order, policy1/explorer then policy2/chaser. Each
    # PPOTorchPolicy owns its optimizer and adaptive KL coefficient.
    explorer_stats = _rllib_policy_update(
        config=config,
        agent_rollout=rollout.explorer,
        policy=explorer,
        optimizer=explorer_optimizer,
        kl_coeff=learner_state.explorer_kl_coeff,
        minibatch_generator=minibatch_generator,
        capture_metrics=capture_metrics,
        policy_name="explorer",
    )
    chaser_stats = _rllib_policy_update(
        config=config,
        agent_rollout=rollout.chaser,
        policy=chaser,
        optimizer=chaser_optimizer,
        kl_coeff=learner_state.chaser_kl_coeff,
        minibatch_generator=minibatch_generator,
        capture_metrics=capture_metrics,
        policy_name="chaser",
    )

    learner_state.chaser_kl_coeff = _updated_kl_coeff(
        learner_state.chaser_kl_coeff,
        sampled_kl=chaser_stats.kl,
        target=config.kl_target,
    )
    learner_state.explorer_kl_coeff = _updated_kl_coeff(
        learner_state.explorer_kl_coeff,
        sampled_kl=explorer_stats.kl,
        target=config.kl_target,
    )

    if not capture_metrics:
        return None
    return _metrics_from_updates(
        rollout=rollout,
        chaser=chaser,
        explorer=explorer,
        learner_state=learner_state,
        policy_loss=0.5 * (chaser_stats.policy_loss + explorer_stats.policy_loss),
        value_loss=0.5 * (chaser_stats.value_loss + explorer_stats.value_loss),
        entropy=0.5 * (chaser_stats.entropy + explorer_stats.entropy),
        kl=0.5 * (chaser_stats.kl + explorer_stats.kl),
        chaser_grad_norm=chaser_stats.grad_norm,
        explorer_grad_norm=explorer_stats.grad_norm,
    )


def _rllib_policy_update(
    *,
    config: TrainConfig,
    agent_rollout: AgentRollout,
    policy: RNNActorCritic,
    optimizer: torch.optim.Optimizer,
    kl_coeff: float,
    minibatch_generator: torch.Generator,
    capture_metrics: bool,
    policy_name: str,
) -> _PolicyUpdateStats:
    sequences = _as_recurrent_sequences(agent_rollout, config.max_seq_len)
    minibatch_specs = _rllib_recurrent_minibatch_specs(
        sequences.sequence_lengths,
        config.sgd_minibatch_size,
    )
    # RLlib reconstructs the same padded recurrent views on every epoch. They
    # are immutable, so materialize them once; this removes Python/device
    # allocation overhead without changing slice boundaries or update order.
    minibatches = [
        _slice_recurrent_minibatch(sequences, spec)
        for spec in minibatch_specs
    ]
    parameters = list(policy.parameters())
    policy_losses: list[torch.Tensor] = []
    value_losses: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    kls: list[torch.Tensor] = []
    grad_norms: list[torch.Tensor] = []
    all_finite = torch.ones((), dtype=torch.bool, device=sequences.actions.device)

    for epoch in range(config.ppo_epochs):
        order = torch.randperm(len(minibatches), generator=minibatch_generator).tolist()
        for batch_index in order:
            minibatch = minibatches[batch_index]
            logits, values = _evaluate_recurrent_sequences(policy, minibatch)
            new_log_probs = logits.log_softmax(dim=-1)
            action_log_probs = new_log_probs.gather(
                dim=-1,
                index=minibatch.actions.unsqueeze(-1),
            ).squeeze(-1)
            # [RAY-PPO-LOSS]: clipped surrogate, squared value-error clipping,
            # exact categorical KL, entropy, and KL-penalized loss composition.
            ratio = (action_log_probs - minibatch.old_log_probs).exp()
            surrogate = torch.minimum(
                minibatch.advantages * ratio,
                minibatch.advantages
                * ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon),
            )
            policy_loss = -_masked_mean(surrogate, minibatch.valid)

            value_error = (values - minibatch.returns).square()
            if config.value_clip > 0:
                value_error = value_error.clamp(0.0, config.value_clip)
            value_loss = _masked_mean(value_error, minibatch.valid)

            old_log_probs = minibatch.old_logits.log_softmax(dim=-1)
            old_probs = old_log_probs.exp()
            categorical_kl = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1)
            mean_kl = _masked_mean(categorical_kl, minibatch.valid)
            entropy = _masked_mean(
                -(new_log_probs.exp() * new_log_probs).sum(dim=-1),
                minibatch.valid,
            )
            recurrent_l2 = config.recurrent_l2_coef * policy.recurrent_weight_norm()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + kl_coeff * mean_kl
                + recurrent_l2
            )

            if config.finite_guard:
                # Keep the guard on-device across all 990 optimizer steps for
                # this policy. Calling .item() per scalar/minibatch would force
                # thousands of CUDA host synchronizations per update.
                finite_values = torch.stack(
                    [policy_loss, value_loss, entropy, mean_kl, recurrent_l2, loss]
                ).detach()
                all_finite &= torch.isfinite(finite_values).all()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = _gradient_norm(
                parameters,
                max_norm=config.grad_clip,
                error_if_nonfinite=False,
            )
            if config.finite_guard:
                all_finite &= torch.isfinite(grad_norm.detach())
            optimizer.step()

            # RLlib updates its KL coefficient from the learner's mean KL over
            # all minibatches. Keep KL regardless of reporting cadence.
            kls.append(mean_kl.detach())
            if capture_metrics:
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropies.append(entropy.detach())
                grad_norms.append(grad_norm.detach())

    if config.finite_guard:
        sentinel = torch.where(
            all_finite,
            torch.zeros((), device=all_finite.device),
            torch.full((), float("nan"), device=all_finite.device),
        )
        assert_finite_tensor(
            sentinel,
            location=f"{policy_name}_ppo_update",
            name="accumulated_loss_or_gradient",
        )

    mean_kl_value = _tensor_mean(kls)
    return _PolicyUpdateStats(
        policy_loss=_tensor_mean(policy_losses) if capture_metrics else 0.0,
        value_loss=_tensor_mean(value_losses) if capture_metrics else 0.0,
        entropy=_tensor_mean(entropies) if capture_metrics else 0.0,
        kl=mean_kl_value,
        grad_norm=_tensor_mean(grad_norms) if capture_metrics else 0.0,
    )


def _as_recurrent_sequences(
    rollout: AgentRollout,
    max_seq_len: int,
) -> _RecurrentSequenceBatch:
    if rollout.state_inputs is None:
        raise ValueError("recurrent state inputs were not recorded")
    steps, episodes = rollout.actions.shape
    chunks = (steps + max_seq_len - 1) // max_seq_len
    width = min(max_seq_len, steps)
    padded_steps = chunks * width

    def chunk(tensor: torch.Tensor) -> torch.Tensor:
        episode_major = tensor.transpose(0, 1)
        if padded_steps > steps:
            episode_major = torch.cat(
                [
                    episode_major,
                    torch.zeros(
                        episodes,
                        padded_steps - steps,
                        *tensor.shape[2:],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    ),
                ],
                dim=1,
            )
        return episode_major.reshape(episodes * chunks, width, *tensor.shape[2:])

    per_episode_lengths = [width] * (chunks - 1) + [steps - (chunks - 1) * width]
    sequence_lengths = tuple(per_episode_lengths * episodes)
    valid = torch.arange(width, device=rollout.actions.device).unsqueeze(0) < torch.tensor(
        sequence_lengths,
        device=rollout.actions.device,
    ).unsqueeze(1)
    sequence_starts = torch.arange(0, steps, max_seq_len, device=rollout.actions.device)
    initial_states = rollout.state_inputs[sequence_starts].permute(1, 0, 2).reshape(
        episodes * chunks,
        -1,
    )
    return _RecurrentSequenceBatch(
        observations=chunk(rollout.observations),
        actions=chunk(rollout.actions),
        old_log_probs=chunk(rollout.old_log_probs),
        old_logits=chunk(rollout.old_logits),
        returns=chunk(rollout.returns),
        advantages=chunk(rollout.advantages),
        initial_states=initial_states,
        valid=valid,
        sequence_lengths=sequence_lengths,
    )


def _rllib_recurrent_minibatch_specs(
    sequence_lengths: tuple[int, ...],
    minibatch_size: int,
) -> list[tuple[int, int, int]]:
    """Match Ray 2.2 SampleBatch slicing; see [RAY-SGD-SLICING].

    In the official 4,000-step batch, 200 sequences of length 20 and a nominal
    minibatch size of 128 produce 33 slices. An over-boundary sequence is
    truncated to eight steps and then reintroduced in full in the next slice;
    the final 32 timesteps are not selected by the released implementation.
    """
    if minibatch_size <= 0:
        return [(0, len(sequence_lengths), sequence_lengths[-1])]
    if max(sequence_lengths) >= minibatch_size:
        raise ValueError("sgd_minibatch_size must be larger than max_seq_len")

    specs: list[tuple[int, int, int]] = []
    accumulated = 0
    start_sequence = 0
    sequence_index = 0
    while sequence_index < len(sequence_lengths):
        sequence_length = sequence_lengths[sequence_index]
        accumulated += sequence_length
        if accumulated >= minibatch_size:
            stop_sequence = sequence_index + 1
            overhead = accumulated - minibatch_size
            final_length = sequence_length - overhead
            specs.append((start_sequence, stop_sequence, final_length))
            if overhead > 0:
                sequence_index -= 1
            accumulated = 0
            start_sequence = sequence_index + 1
        sequence_index += 1
    if not specs:
        raise ValueError("train batch must contain at least one SGD minibatch")
    return specs


def _slice_recurrent_minibatch(
    batch: _RecurrentSequenceBatch,
    spec: tuple[int, int, int],
) -> _RecurrentSequenceBatch:
    start, stop, final_length = spec
    lengths = list(batch.sequence_lengths[start:stop])
    lengths[-1] = final_length
    width = batch.observations.shape[1]
    valid = torch.arange(width, device=batch.actions.device).unsqueeze(0) < torch.tensor(
        lengths,
        device=batch.actions.device,
    ).unsqueeze(1)

    def masked_slice(tensor: torch.Tensor) -> torch.Tensor:
        values = tensor[start:stop]
        mask = valid.reshape(*valid.shape, *([1] * (values.ndim - 2)))
        return torch.where(mask, values, torch.zeros_like(values))

    return _RecurrentSequenceBatch(
        observations=masked_slice(batch.observations),
        actions=masked_slice(batch.actions),
        old_log_probs=masked_slice(batch.old_log_probs),
        old_logits=masked_slice(batch.old_logits),
        returns=masked_slice(batch.returns),
        advantages=masked_slice(batch.advantages),
        initial_states=batch.initial_states[start:stop],
        valid=valid,
        sequence_lengths=tuple(lengths),
    )


def _evaluate_recurrent_sequences(
    policy: RNNActorCritic,
    batch: _RecurrentSequenceBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = batch.observations.transpose(0, 1)
    hidden, _ = policy.rnn(observations, batch.initial_states.unsqueeze(0))
    hidden = hidden.transpose(0, 1)
    return policy.action_layer(hidden), policy.value_layer(hidden).squeeze(-1)


def _updated_kl_coeff(current: float, *, sampled_kl: float, target: float) -> float:
    """Apply Ray 2.2 KLCoeffMixin's per-policy rule; see [RAY-KL-ADAPT]."""
    if sampled_kl > 2.0 * target:
        return current * 1.5
    if sampled_kl < 0.5 * target:
        return current * 0.5
    return current


def _metrics_from_updates(
    *,
    rollout: Rollout,
    chaser: PolicyBase,
    explorer: PolicyBase,
    learner_state: LearnerState,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    kl: float,
    chaser_grad_norm: float,
    explorer_grad_norm: float,
) -> RolloutMetrics:
    return RolloutMetrics(
        collisions_per_episode=rollout.metrics.collisions_per_episode,
        chaser_return=rollout.metrics.chaser_return,
        explorer_return=rollout.metrics.explorer_return,
        chaser_partner_vision=rollout.metrics.chaser_partner_vision,
        explorer_partner_vision=rollout.metrics.explorer_partner_vision,
        chaser_new_fields=rollout.metrics.chaser_new_fields,
        explorer_new_fields=rollout.metrics.explorer_new_fields,
        final_distance=rollout.metrics.final_distance,
        chaser_subspace_norm=rollout.metrics.chaser_subspace_norm,
        explorer_subspace_norm=rollout.metrics.explorer_subspace_norm,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=kl,
        chaser_grad_norm=chaser_grad_norm,
        explorer_grad_norm=explorer_grad_norm,
        max_param_abs=module_max_abs(chaser, explorer),
        chaser_kl_coeff=learner_state.chaser_kl_coeff,
        explorer_kl_coeff=learner_state.explorer_kl_coeff,
    )


def _gradient_norm(
    parameters: list[nn.Parameter],
    *,
    max_norm: float | None,
    error_if_nonfinite: bool,
) -> torch.Tensor:
    if max_norm is not None:
        return nn.utils.clip_grad_norm_(
            parameters,
            max_norm,
            error_if_nonfinite=error_if_nonfinite,
        )
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.zeros(())
    total = torch.stack(norms).norm(2)
    if error_if_nonfinite:
        assert_finite_tensor(total, location="ppo_gradient", name="global_grad_norm")
    return total


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return values[valid].sum() / valid.sum()


def _tensor_mean(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    return torch.stack(values).mean().item()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate_actions(
    policy: PolicyBase,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values, _ = policy.sequence(observations)
    log_probs = logits.log_softmax(dim=-1)
    action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return action_log_probs, values, entropy


def _clipped_policy_loss(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    ratio = (log_prob - old_log_prob).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()

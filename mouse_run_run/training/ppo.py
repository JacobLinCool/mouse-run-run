"""Independent recurrent PPO with complete non-overlapping TBPTT coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from mouse_run_run.core.config import PPOConfig
from mouse_run_run.core.policy import PolicyModule
from mouse_run_run.core.simulation import AgentTrajectory, TrajectoryBatch
from mouse_run_run.core.types import AgentId, TensorMap


@dataclass(frozen=True)
class AgentPPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    categorical_kl: float
    kl_coefficient: float
    recurrent_l2: float
    gradient_norm: float
    return_mean: float
    minibatches: float
    active_samples_per_epoch: float


@dataclass(frozen=True)
class PPOMetrics:
    agents: dict[AgentId, AgentPPOMetrics]

    def flat(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for agent_id, metrics in self.agents.items():
            for key, value in vars(metrics).items():
                values[f"{agent_id}.{key}"] = float(value)
        return values


@dataclass(frozen=True)
class RecurrentSequenceBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_logits: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    valid: torch.Tensor
    initial_states: TensorMap

    @property
    def sequences(self) -> int:
        return self.valid.shape[0]


class IndependentPPO:
    """One optimizer and adaptive KL state per independently owned policy."""

    def __init__(
        self,
        policies: dict[AgentId, PolicyModule],
        config: PPOConfig,
        *,
        seed: int = 0,
    ) -> None:
        config.validate()
        self.policies = policies
        self.config = config
        self.optimizers = {
            agent_id: torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
            for agent_id, policy in policies.items()
        }
        self.kl_coefficients = {
            agent_id: float(config.initial_kl_coefficient) for agent_id in policies
        }
        self.minibatch_generator = torch.Generator().manual_seed(seed)

    def update(self, rollout: TrajectoryBatch) -> PPOMetrics:
        if tuple(self.policies) != rollout.agent_ids:
            raise ValueError("PPO policies and rollout agents differ")
        advantages, returns = _advantages(rollout, self.config)
        results: dict[AgentId, AgentPPOMetrics] = {}
        for agent_id in rollout.agent_ids:
            results[agent_id] = self._update_agent(
                agent_id,
                rollout.agents[agent_id],
                rollout.active,
                advantages[agent_id],
                returns[agent_id],
            )
        return PPOMetrics(agents=results)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "kl_coefficients": {
                str(agent_id): coefficient
                for agent_id, coefficient in self.kl_coefficients.items()
            }
        }

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        return {"minibatch_rng": self.minibatch_generator.get_state().contiguous()}

    def restore_checkpoint(
        self,
        metadata: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ) -> None:
        encoded = metadata.get("kl_coefficients")
        if not isinstance(encoded, dict) or tuple(encoded) != tuple(
            str(agent_id) for agent_id in self.policies
        ):
            raise ValueError("checkpoint adaptive KL agents do not match learner")
        if set(tensors) != {"minibatch_rng"}:
            raise ValueError("checkpoint learner tensors must contain only minibatch_rng")
        self.kl_coefficients = {
            agent_id: float(encoded[str(agent_id)]) for agent_id in self.policies
        }
        self.minibatch_generator.set_state(tensors["minibatch_rng"].cpu())

    def _update_agent(
        self,
        agent_id: AgentId,
        trajectory: AgentTrajectory,
        active: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> AgentPPOMetrics:
        policy = self.policies[agent_id]
        optimizer = self.optimizers[agent_id]
        normalized = _normalize_advantages(advantages, active)
        sequences = _as_recurrent_sequences(
            trajectory,
            normalized,
            returns,
            active,
            self.config.sequence_length,
        )
        active_samples = int(sequences.valid.sum())
        if active_samples < 1:
            raise ValueError(f"PPO rollout for {agent_id} has no active samples")

        accumulators: dict[str, list[torch.Tensor]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "categorical_kl": [],
            "recurrent_l2": [],
            "gradient_norm": [],
        }
        minibatch_count = 0
        for _ in range(self.config.epochs):
            groups = _epoch_minibatches(
                sequences.valid,
                self.config.minibatch_size,
                self.minibatch_generator,
            )
            if sum(int(sequences.valid[group].sum()) for group in groups) != active_samples:
                raise AssertionError("recurrent minibatches did not cover each active sample once")
            for group in groups:
                batch = _select_sequences(sequences, group)
                evaluated = policy.evaluate_sequence(
                    batch.observations.transpose(0, 1),
                    initial_state={
                        key: value for key, value in batch.initial_states.items()
                    },
                )
                logits = evaluated.logits.transpose(0, 1)
                values = evaluated.values.transpose(0, 1)
                new_log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = new_log_probs.gather(
                    -1, batch.actions.unsqueeze(-1)
                ).squeeze(-1)
                ratio = (action_log_probs - batch.old_log_probs).exp()
                surrogate = torch.minimum(
                    ratio * batch.advantages,
                    ratio.clamp(
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * batch.advantages,
                )
                policy_loss = -_masked_mean(surrogate, batch.valid)
                squared_error = (values - batch.returns).square()
                if self.config.value_clip > 0.0:
                    squared_error = squared_error.clamp(max=self.config.value_clip)
                value_loss = _masked_mean(squared_error, batch.valid)
                old_log_distribution = F.log_softmax(batch.old_logits, dim=-1)
                old_distribution = old_log_distribution.exp()
                categorical_kl = (
                    old_distribution * (old_log_distribution - new_log_probs)
                ).sum(dim=-1)
                mean_kl = _masked_mean(categorical_kl, batch.valid)
                entropy = _masked_mean(
                    -(new_log_probs.exp() * new_log_probs).sum(dim=-1),
                    batch.valid,
                )
                recurrent_l2 = (
                    self.config.recurrent_l2_coefficient
                    * policy.recurrent_weight_norm()
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.kl_coefficients[agent_id] * mean_kl
                    + recurrent_l2
                )
                finite = torch.stack(
                    [policy_loss, value_loss, entropy, mean_kl, recurrent_l2, loss]
                )
                if not bool(torch.isfinite(finite).all()):
                    raise FloatingPointError(f"non-finite PPO loss for agent {agent_id}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = _gradient_norm(policy, self.config.max_gradient_norm)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"non-finite PPO gradient for agent {agent_id}")
                optimizer.step()
                for name, value in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                    ("categorical_kl", mean_kl),
                    ("recurrent_l2", recurrent_l2),
                    ("gradient_norm", gradient_norm),
                ):
                    accumulators[name].append(value.detach())
                minibatch_count += 1

        sampled_kl = float(torch.stack(accumulators["categorical_kl"]).mean())
        self.kl_coefficients[agent_id] = updated_kl_coefficient(
            self.kl_coefficients[agent_id],
            sampled_kl=sampled_kl,
            target=self.config.kl_target,
        )
        mask = active.to(trajectory.rewards.dtype)
        return AgentPPOMetrics(
            policy_loss=_mean(accumulators["policy_loss"]),
            value_loss=_mean(accumulators["value_loss"]),
            entropy=_mean(accumulators["entropy"]),
            categorical_kl=sampled_kl,
            kl_coefficient=self.kl_coefficients[agent_id],
            recurrent_l2=_mean(accumulators["recurrent_l2"]),
            gradient_norm=_mean(accumulators["gradient_norm"]),
            return_mean=float((trajectory.rewards * mask).sum(dim=0).mean()),
            minibatches=float(minibatch_count),
            active_samples_per_epoch=float(active_samples),
        )


def _as_recurrent_sequences(
    trajectory: AgentTrajectory,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    active: torch.Tensor,
    sequence_length: int,
) -> RecurrentSequenceBatch:
    if not trajectory.state_inputs:
        raise ValueError("recurrent PPO requires recorded trajectory boundary states")
    steps, episodes = active.shape
    chunks = (steps + sequence_length - 1) // sequence_length
    padded_steps = chunks * sequence_length

    def chunk(value: torch.Tensor) -> torch.Tensor:
        episode_major = value.transpose(0, 1)
        if padded_steps > steps:
            padding = torch.zeros(
                episodes,
                padded_steps - steps,
                *value.shape[2:],
                dtype=value.dtype,
                device=value.device,
            )
            episode_major = torch.cat((episode_major, padding), dim=1)
        return episode_major.reshape(
            episodes * chunks,
            sequence_length,
            *value.shape[2:],
        )

    starts = torch.arange(0, steps, sequence_length, device=active.device)
    initial_states = {
        key: value[starts].transpose(0, 1).reshape(
            episodes * chunks, *value.shape[2:]
        )
        for key, value in trajectory.state_inputs.items()
    }
    return RecurrentSequenceBatch(
        observations=chunk(trajectory.observations),
        actions=chunk(trajectory.actions),
        old_logits=chunk(trajectory.logits),
        old_log_probs=chunk(trajectory.log_probs),
        old_values=chunk(trajectory.values),
        advantages=chunk(advantages),
        returns=chunk(returns),
        valid=chunk(active),
        initial_states=initial_states,
    )


def _epoch_minibatches(
    valid: torch.Tensor,
    minibatch_size: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Pack whole TBPTT segments without overlap or dropped active timesteps."""
    costs = valid.sum(dim=1).cpu()
    usable = torch.nonzero(costs > 0, as_tuple=False).flatten()
    order = usable[torch.randperm(usable.numel(), generator=generator)]
    groups: list[torch.Tensor] = []
    current: list[int] = []
    total = 0
    for index in order.tolist():
        cost = int(costs[index])
        if cost > minibatch_size:
            raise ValueError("one recurrent segment exceeds PPO minibatch_size")
        if current and total + cost > minibatch_size:
            groups.append(torch.tensor(current, dtype=torch.long, device=valid.device))
            current = []
            total = 0
        current.append(index)
        total += cost
    if current:
        groups.append(torch.tensor(current, dtype=torch.long, device=valid.device))
    if not groups:
        raise ValueError("PPO found no non-empty recurrent minibatches")
    flattened = torch.cat(groups)
    if flattened.unique().numel() != usable.numel():
        raise AssertionError("PPO recurrent sequence coverage is not one-to-one")
    return groups


def _select_sequences(
    batch: RecurrentSequenceBatch,
    indices: torch.Tensor,
) -> RecurrentSequenceBatch:
    return RecurrentSequenceBatch(
        observations=batch.observations[indices],
        actions=batch.actions[indices],
        old_logits=batch.old_logits[indices],
        old_log_probs=batch.old_log_probs[indices],
        old_values=batch.old_values[indices],
        advantages=batch.advantages[indices],
        returns=batch.returns[indices],
        valid=batch.valid[indices],
        initial_states={key: value[indices] for key, value in batch.initial_states.items()},
    )


def updated_kl_coefficient(current: float, *, sampled_kl: float, target: float) -> float:
    if sampled_kl > 2.0 * target:
        return current * 1.5
    if sampled_kl < 0.5 * target:
        return current * 0.5
    return current


def _normalize_advantages(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    selected = values[active]
    result = torch.zeros_like(values)
    if selected.numel() == 1:
        result[active] = selected
    elif selected.numel() > 1:
        result[active] = (selected - selected.mean()) / selected.std(
            correction=0
        ).clamp_min(1e-4)
    return result


def _gradient_norm(policy: PolicyModule, max_norm: float | None) -> torch.Tensor:
    parameters = [parameter for parameter in policy.parameters() if parameter.grad is not None]
    if not parameters:
        return torch.zeros(())
    if max_norm is not None:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return torch.stack([parameter.grad.detach().norm() for parameter in parameters]).norm()


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return values[valid].mean()


def _mean(values: list[torch.Tensor]) -> float:
    return float(torch.stack(values).mean())


def _advantages(
    rollout: TrajectoryBatch,
    config: PPOConfig,
) -> tuple[dict[AgentId, torch.Tensor], dict[AgentId, torch.Tensor]]:
    advantages: dict[AgentId, torch.Tensor] = {}
    returns: dict[AgentId, torch.Tensor] = {}
    done = rollout.terminated | rollout.truncated
    for agent_id in rollout.agent_ids:
        trajectory = rollout.agents[agent_id]
        advantage = torch.zeros_like(trajectory.rewards)
        next_advantage = torch.zeros(rollout.batch_size, device=trajectory.rewards.device)
        next_value = trajectory.bootstrap_value
        for step in reversed(range(rollout.horizon)):
            nonterminal = (~done[step]).to(trajectory.rewards.dtype)
            delta = (
                trajectory.rewards[step]
                + config.gamma * next_value * nonterminal
                - trajectory.values[step]
            )
            next_advantage = (
                delta
                + config.gamma * config.gae_lambda * nonterminal * next_advantage
            ) * rollout.active[step].to(delta.dtype)
            advantage[step] = next_advantage
            next_value = trajectory.values[step]
        advantages[agent_id] = advantage
        returns[agent_id] = advantage + trajectory.values
    return advantages, returns

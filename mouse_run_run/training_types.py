from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutMetrics:
    collisions_per_episode: float
    chaser_return: float
    explorer_return: float
    chaser_partner_vision: float
    explorer_partner_vision: float
    chaser_new_fields: float
    explorer_new_fields: float
    final_distance: float
    chaser_subspace_norm: float
    explorer_subspace_norm: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    chaser_grad_norm: float
    explorer_grad_norm: float
    max_param_abs: float
    chaser_kl_coeff: float
    explorer_kl_coeff: float


@dataclass(frozen=True)
class AgentRollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_logits: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    state_inputs: torch.Tensor | None = None


@dataclass(frozen=True)
class Rollout:
    chaser: AgentRollout
    explorer: AgentRollout
    metrics: RolloutMetrics


@dataclass
class LearnerState:
    """Mutable PPO state that RLlib keeps independently for each policy."""

    chaser_kl_coeff: float
    explorer_kl_coeff: float




def tensor_mean(values: list[torch.Tensor]) -> float:
    """Mean of a list of scalar tensors as a float; 0.0 for an empty list."""
    if not values:
        return 0.0
    return torch.stack(values).mean().item()


def empty_metrics() -> RolloutMetrics:
    return RolloutMetrics(
        collisions_per_episode=0.0,
        chaser_return=0.0,
        explorer_return=0.0,
        chaser_partner_vision=0.0,
        explorer_partner_vision=0.0,
        chaser_new_fields=0.0,
        explorer_new_fields=0.0,
        final_distance=0.0,
        chaser_subspace_norm=0.0,
        explorer_subspace_norm=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        approx_kl=0.0,
        chaser_grad_norm=0.0,
        explorer_grad_norm=0.0,
        max_param_abs=0.0,
        chaser_kl_coeff=0.0,
        explorer_kl_coeff=0.0,
    )

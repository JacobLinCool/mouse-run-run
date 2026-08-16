"""Small reusable policy implementations used by evaluation recipes."""

from __future__ import annotations

import torch

from mouse_run_run.core.policy import (
    ActivationSite,
    PolicyFeatures,
    PolicyModule,
    PolicyReadout,
    SequenceEvaluation,
)
from mouse_run_run.core.types import TensorMap, TensorTree


class UniformRandomPolicy(PolicyModule):
    """A parameter-free categorical policy with exactly uniform logits."""

    def __init__(self, action_count: int) -> None:
        super().__init__()
        if action_count < 2:
            raise ValueError("uniform policy requires at least two actions")
        self.action_count = action_count

    @property
    def activation_sites(self) -> dict[str, ActivationSite]:
        return {}

    def initial_state(self, batch_size: int, device: torch.device) -> TensorTree:
        return {"_marker": torch.empty(batch_size, 0, device=device)}

    def advance(self, observation: torch.Tensor, state: TensorTree) -> PolicyFeatures:
        if set(state) != {"_marker"}:
            raise ValueError("uniform policy received invalid marker state")
        marker = torch.empty(observation.shape[0], 0, device=observation.device)
        return PolicyFeatures(activations={}, next_state={"_marker": marker})

    def readout(
        self,
        features: PolicyFeatures,
        activation_overrides: TensorMap | None = None,
    ) -> PolicyReadout:
        if activation_overrides:
            raise ValueError("uniform policy has no activation sites")
        batch = _batch_size(features)
        device = _device(features)
        return PolicyReadout(
            logits=torch.zeros(batch, self.action_count, device=device),
            value=torch.zeros(batch, device=device),
        )

    def carry_state(
        self,
        features: PolicyFeatures,
        activation_overrides: TensorMap | None = None,
    ) -> TensorTree:
        if activation_overrides:
            raise ValueError("uniform policy has no activation sites")
        return features.next_state

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        *,
        initial_state: TensorTree | None = None,
    ) -> SequenceEvaluation:
        time, batch = observations.shape[:2]
        return SequenceEvaluation(
            logits=torch.zeros(time, batch, self.action_count, device=observations.device),
            values=torch.zeros(time, batch, device=observations.device),
        )


def _batch_size(features: PolicyFeatures) -> int:
    # The simulation only calls this policy after advance.  A zero-width
    # marker keeps batch/device information tensor-only without exposing an
    # analysis or intervention site.
    return int(features.next_state["_marker"].shape[0])


def _device(features: PolicyFeatures) -> torch.device:
    return features.next_state["_marker"].device

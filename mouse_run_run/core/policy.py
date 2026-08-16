"""Policy contracts with explicit, named representation intervention sites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn

from mouse_run_run.core.types import TensorMap, TensorTree


@dataclass(frozen=True)
class ActivationSite:
    name: str
    width: int
    analysis: bool = True
    readout: bool = False
    recurrent: bool = False


@dataclass(frozen=True)
class PolicyFeatures:
    activations: TensorMap
    next_state: TensorTree


@dataclass(frozen=True)
class PolicyReadout:
    logits: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class SequenceEvaluation:
    logits: torch.Tensor
    values: torch.Tensor
    activations: TensorMap = field(default_factory=dict)


class PolicyModule(nn.Module, ABC):
    """A policy separates state advance from action/value readout.

    `activation_overrides` are validated model-owned replacements.  Readout
    overrides affect only the current action/value computation.  Carry-state
    overrides are the recurrent intervention boundary and must be reflected in
    the returned tensor-only state tree.
    """

    @property
    @abstractmethod
    def activation_sites(self) -> dict[str, ActivationSite]: ...

    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> TensorTree: ...

    @abstractmethod
    def advance(self, observation: torch.Tensor, state: TensorTree) -> PolicyFeatures: ...

    @abstractmethod
    def readout(
        self,
        features: PolicyFeatures,
        activation_overrides: TensorMap | None = None,
    ) -> PolicyReadout: ...

    @abstractmethod
    def carry_state(
        self,
        features: PolicyFeatures,
        activation_overrides: TensorMap | None = None,
    ) -> TensorTree: ...

    @abstractmethod
    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        *,
        initial_state: TensorTree | None = None,
    ) -> SequenceEvaluation: ...

    def recurrent_weight_norm(self) -> torch.Tensor:
        """Unsquared recurrent-weight norm used by the Zhang/RLlib learner."""
        parameter = next(self.parameters(), None)
        device = torch.device("cpu") if parameter is None else parameter.device
        return torch.zeros((), device=device)

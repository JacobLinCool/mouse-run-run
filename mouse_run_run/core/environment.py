"""Contract implemented by each experiment environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from mouse_run_run.core.types import AgentId, TensorMap


@dataclass(frozen=True)
class EnvReset:
    observations: dict[AgentId, torch.Tensor]
    world: TensorMap


@dataclass(frozen=True)
class EnvTransition:
    observations: dict[AgentId, torch.Tensor]
    rewards: dict[AgentId, torch.Tensor]
    terminated: torch.Tensor
    truncated: torch.Tensor
    active: torch.Tensor
    world: TensorMap
    events: TensorMap

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


class MultiAgentEnvironment(Protocol):
    """Stateful batched environment with a stable set of named agents."""

    @property
    def agent_ids(self) -> tuple[AgentId, ...]: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def reset(self, *, generator: torch.Generator) -> EnvReset: ...

    def step(self, actions: dict[AgentId, torch.Tensor]) -> EnvTransition: ...

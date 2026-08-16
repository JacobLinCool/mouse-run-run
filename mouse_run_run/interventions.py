"""Reusable rollout interventions backed by fitted analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mouse_run_run.analyses.plsc import StandardizedSubspace
from mouse_run_run.core.intervention import InterventionContext, InterventionTarget
from mouse_run_run.core.types import AgentId


SubspaceOperation = Literal["remove", "keep"]


@dataclass(frozen=True)
class SubspaceIntervention:
    name: str
    subspace: StandardizedSubspace
    target: InterventionTarget
    operation: SubspaceOperation = "remove"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("intervention name must be non-empty")
        if self.target not in ("readout", "recurrent"):
            raise ValueError(f"invalid intervention target {self.target!r}")
        if self.operation not in ("remove", "keep"):
            raise ValueError(f"invalid subspace operation {self.operation!r}")

    @property
    def agent_ids(self) -> frozenset[AgentId]:
        return frozenset({self.subspace.agent_id})

    @property
    def site(self) -> str:
        return self.subspace.site

    def apply(self, value: torch.Tensor, context: InterventionContext) -> torch.Tensor:
        if context.agent_id != self.subspace.agent_id or context.site.name != self.site:
            raise ValueError("subspace intervention received a mismatched context")
        if self.operation == "remove":
            return self.subspace.remove(value)
        if self.operation == "keep":
            return self.subspace.keep(value)
        raise ValueError(f"unknown subspace operation {self.operation!r}")

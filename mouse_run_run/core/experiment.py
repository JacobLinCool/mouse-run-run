"""Typed Python experiment composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from mouse_run_run.core.config import TrainingConfig
from mouse_run_run.core.environment import MultiAgentEnvironment
from mouse_run_run.core.policy import PolicyModule
from mouse_run_run.core.types import AgentId, TensorMap


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    environment_backend: str = "torch"
    batch_size: int = 32
    seed: int = 0

    def validate(self) -> None:
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError(f"unsupported runtime device: {self.device!r}")
        if self.environment_backend not in ("torch", "triton"):
            raise ValueError(
                f"unsupported environment backend: {self.environment_backend!r}"
            )
        if self.batch_size < 1:
            raise ValueError("runtime batch_size must be positive")
        if self.seed < 0:
            raise ValueError("runtime seed must be non-negative")


@dataclass(frozen=True)
class PolicyBuildContext:
    agent_id: AgentId
    observation_shape: tuple[int, ...]
    action_count: int
    device: torch.device


EnvironmentFactory = Callable[[RuntimeConfig, torch.device], MultiAgentEnvironment]
PolicyFactory = Callable[[PolicyBuildContext], PolicyModule]


class ReplayRenderer(Protocol):
    def frame(self, world: TensorMap, *, batch_index: int, step: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Experiment:
    name: str
    agent_ids: tuple[AgentId, ...]
    observation_shapes: Mapping[AgentId, tuple[int, ...]]
    action_counts: Mapping[AgentId, int]
    make_environment: EnvironmentFactory
    policy_factories: Mapping[AgentId, PolicyFactory]
    training: TrainingConfig
    renderer: ReplayRenderer | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.name.replace("-", "_").isidentifier():
            raise ValueError(f"invalid experiment name: {self.name!r}")
        if not self.agent_ids or len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("experiment agent_ids must be non-empty and unique")
        if any(not str(agent_id).isidentifier() for agent_id in self.agent_ids):
            raise ValueError("agent_ids must be valid Python identifiers")
        for label, mapping in (
            ("observation_shapes", self.observation_shapes),
            ("action_counts", self.action_counts),
            ("policy_factories", self.policy_factories),
        ):
            if tuple(mapping) != self.agent_ids:
                raise ValueError(
                    f"{label} keys {tuple(mapping)!r} must exactly match {self.agent_ids!r}"
                )
        for agent_id in self.agent_ids:
            shape = self.observation_shapes[agent_id]
            if not shape or any(dimension < 1 for dimension in shape):
                raise ValueError(f"{agent_id} has invalid observation shape {shape!r}")
            if self.action_counts[agent_id] < 2:
                raise ValueError(f"{agent_id} action count must be at least two")
        self.training.validate()


@dataclass(frozen=True)
class ExperimentDefinition:
    """A typed config value plus the pure function that composes an experiment."""

    config: Any
    build: Callable[[Any], Experiment]

    def resolve(self, config: Any | None = None) -> Experiment:
        experiment = self.build(self.config if config is None else config)
        experiment.validate()
        return experiment

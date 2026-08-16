"""Strict safetensors checkpoints; no pickle and no legacy readers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from mouse_run_run.core.policy import PolicyModule
from mouse_run_run.core.types import AgentId


CHECKPOINT_FORMAT = "mrr-checkpoint-v3"


class CheckpointableLearner(Protocol):
    def checkpoint_metadata(self) -> dict[str, Any]: ...

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]: ...

    def restore_checkpoint(
        self,
        metadata: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ) -> None: ...


@dataclass(frozen=True)
class CheckpointMetadata:
    experiment: str
    update: int
    config: dict[str, Any]
    metrics: dict[str, float]
    learner_state: dict[str, Any]


def save_checkpoint(
    path: Path,
    *,
    experiment: str,
    update: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, float],
    policies: Mapping[AgentId, PolicyModule],
    optimizers: Mapping[AgentId, torch.optim.Optimizer],
    learner: CheckpointableLearner,
) -> None:
    if update < 0:
        raise ValueError("checkpoint update must be non-negative")
    if tuple(policies) != tuple(optimizers):
        raise ValueError("checkpoint policy and optimizer agents differ")
    tensors: dict[str, torch.Tensor] = {}
    for agent_id, policy in policies.items():
        for key, value in policy.state_dict().items():
            tensors[f"policy.{agent_id}.{key}"] = _cpu_contiguous(value)

    optimizer_metadata: dict[str, Any] = {}
    for agent_id, optimizer in optimizers.items():
        state = optimizer.state_dict()
        state_keys: dict[str, list[str]] = {}
        for parameter_index, values in state["state"].items():
            tensor_keys: list[str] = []
            for key, value in values.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"optimizer {agent_id}.{parameter_index}.{key} is not tensor-valued"
                    )
                tensors[
                    f"optimizer.{agent_id}.state.{parameter_index}.{key}"
                ] = _cpu_contiguous(value)
                tensor_keys.append(key)
            state_keys[str(parameter_index)] = tensor_keys
        optimizer_metadata[str(agent_id)] = {
            "param_groups": state["param_groups"],
            "state_keys": state_keys,
        }
    learner_tensors = learner.checkpoint_tensors()
    if not learner_tensors:
        raise ValueError("checkpoint learner state must contain tensors")
    for key, value in learner_tensors.items():
        if not key or "." in key:
            raise ValueError(f"invalid learner checkpoint tensor key {key!r}")
        tensors[f"learner.{key}"] = _cpu_contiguous(value)
    tensors["rng.cpu"] = torch.random.get_rng_state().contiguous()
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"rng.cuda.{index}"] = state.contiguous().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    save_file(
        tensors,
        str(temporary),
        metadata={
            "format": CHECKPOINT_FORMAT,
            "experiment": experiment,
            "update": str(update),
            "config": _json(config),
            "metrics": _json(metrics),
            "optimizers": _json(optimizer_metadata),
            "learner": _json(learner.checkpoint_metadata()),
        },
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    policies: Mapping[AgentId, PolicyModule],
    optimizers: Mapping[AgentId, torch.optim.Optimizer] | None = None,
    learner: CheckpointableLearner | None = None,
    restore_rng: bool = False,
) -> CheckpointMetadata:
    metadata = _raw_metadata(path)
    tensors = load_file(str(path), device="cpu")
    for agent_id, policy in policies.items():
        prefix = f"policy.{agent_id}."
        state = {
            key.removeprefix(prefix): value
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError(f"checkpoint has no policy for agent {agent_id!r}")
        policy.load_state_dict(state, strict=True)
    expected_policy_prefixes = {f"policy.{agent_id}." for agent_id in policies}
    for key in tensors:
        if key.startswith("policy.") and not any(
            key.startswith(prefix) for prefix in expected_policy_prefixes
        ):
            raise ValueError(f"checkpoint contains unexpected policy tensor {key!r}")

    if optimizers is not None:
        if tuple(optimizers) != tuple(policies):
            raise ValueError("optimizer keys must exactly match policy keys")
        optimizer_metadata = json.loads(metadata["optimizers"])
        if tuple(optimizer_metadata) != tuple(str(agent) for agent in policies):
            raise ValueError("checkpoint optimizer agents do not match policies")
        for agent_id, optimizer in optimizers.items():
            encoded = optimizer_metadata[str(agent_id)]
            state: dict[int, dict[str, torch.Tensor]] = {}
            for parameter_index, keys in encoded["state_keys"].items():
                state[int(parameter_index)] = {
                    key: tensors[
                        f"optimizer.{agent_id}.state.{parameter_index}.{key}"
                    ]
                    for key in keys
                }
            optimizer.load_state_dict(
                {"state": state, "param_groups": encoded["param_groups"]}
            )
    if learner is not None:
        learner.restore_checkpoint(
            json.loads(metadata["learner"]),
            {
                key.removeprefix("learner."): value
                for key, value in tensors.items()
                if key.startswith("learner.")
            },
        )
    if restore_rng:
        torch.random.set_rng_state(tensors["rng.cpu"])
        cuda_states = [
            tensors[f"rng.cuda.{index}"]
            for index in range(torch.cuda.device_count())
            if f"rng.cuda.{index}" in tensors
        ]
        if cuda_states:
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA RNG device count differs from runtime")
            torch.cuda.set_rng_state_all(cuda_states)
    return _decode_metadata(metadata)


def read_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    return _decode_metadata(_raw_metadata(path))


def _raw_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} artifact; legacy checkpoints are archived only"
        )
    required = {"experiment", "update", "config", "metrics", "optimizers", "learner"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"checkpoint metadata missing {sorted(missing)!r}")
    return metadata


def _decode_metadata(metadata: Mapping[str, str]) -> CheckpointMetadata:
    return CheckpointMetadata(
        experiment=metadata["experiment"],
        update=int(metadata["update"]),
        config=json.loads(metadata["config"]),
        metrics={key: float(value) for key, value in json.loads(metadata["metrics"]).items()},
        learner_state=json.loads(metadata["learner"]),
    )


def _cpu_contiguous(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class TrainingState:
    """Optimizer/RNG state needed to resume training from a checkpoint."""

    update: int
    optimizer_state: dict[str, Any]
    cpu_rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor | None = None


CHECKPOINT_FORMAT = "mouse-run-run-checkpoint-v1"
ROLLOUT_FORMAT = "mouse-run-run-rollout-v1"


def save_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    chaser_state: Mapping[str, torch.Tensor],
    explorer_state: Mapping[str, torch.Tensor],
    training_state: TrainingState | None = None,
) -> None:
    tensors = {
        **_prefix_state("chaser", chaser_state),
        **_prefix_state("explorer", explorer_state),
    }
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "config": _to_json(config),
        "metrics": _to_json(metrics),
    }
    if training_state is not None:
        state_tensors, state_metadata = _encode_training_state(training_state)
        tensors.update(state_tensors)
        metadata["training_state"] = _to_json(state_metadata)
    save_file(tensors, str(path), metadata=metadata)


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    metadata = read_metadata(path)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}: {metadata.get('format')}")

    tensors = load_file(str(path), device="cpu")
    return (
        _from_json(metadata["config"]),
        _from_json(metadata["metrics"]),
        _unprefix_state("chaser", tensors),
        _unprefix_state("explorer", tensors),
    )


def save_rollout(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    rollout: Mapping[str, torch.Tensor],
) -> None:
    tensors = {
        f"rollout.{key}": _prepare_tensor(value)
        for key, value in rollout.items()
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "format": ROLLOUT_FORMAT,
            "metadata": _to_json(metadata),
        },
    )


def read_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
    return dict(metadata or {})


def read_checkpoint_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (config, metrics) from checkpoint metadata without tensor IO."""
    metadata = read_metadata(path)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}: {metadata.get('format')}")
    return _from_json(metadata["config"]), _from_json(metadata["metrics"])


def load_training_state(path: Path) -> TrainingState | None:
    """Return the resume state stored in a checkpoint, or None if absent."""
    metadata = read_metadata(path)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}: {metadata.get('format')}")
    raw_state = metadata.get("training_state")
    if raw_state is None:
        return None
    state_metadata = _from_json(raw_state)
    tensors = load_file(str(path), device="cpu")

    optimizer_entries: dict[int, dict[str, Any]] = {}
    for index_key, keys in state_metadata["optimizer_state_keys"].items():
        entry: dict[str, Any] = {}
        for key in keys:
            entry[key] = tensors[f"optimizer.state.{index_key}.{key}"]
        optimizer_entries[int(index_key)] = entry
    optimizer_state = {
        "state": optimizer_entries,
        "param_groups": state_metadata["optimizer_param_groups"],
    }
    cuda_key = "rng.cuda"
    return TrainingState(
        update=int(state_metadata["update"]),
        optimizer_state=optimizer_state,
        cpu_rng_state=tensors["rng.cpu"],
        cuda_rng_state=tensors.get(cuda_key),
    )


def _encode_training_state(
    state: TrainingState,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {"rng.cpu": _prepare_tensor(state.cpu_rng_state)}
    if state.cuda_rng_state is not None:
        tensors["rng.cuda"] = _prepare_tensor(state.cuda_rng_state)

    state_keys: dict[str, list[str]] = {}
    for index, entry in state.optimizer_state.get("state", {}).items():
        keys: list[str] = []
        for key, value in entry.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"optimizer state entry {index}.{key} is not a tensor: {type(value)!r}"
                )
            tensors[f"optimizer.state.{index}.{key}"] = _prepare_tensor(value)
            keys.append(key)
        state_keys[str(index)] = keys

    metadata = {
        "update": state.update,
        "optimizer_param_groups": state.optimizer_state.get("param_groups", []),
        "optimizer_state_keys": state_keys,
    }
    return tensors, metadata


def _prefix_state(
    prefix: str,
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{key}": _prepare_tensor(value)
        for key, value in state.items()
    }


def _unprefix_state(
    prefix: str,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    marker = f"{prefix}."
    return {
        key.removeprefix(marker): value
        for key, value in tensors.items()
        if key.startswith(marker)
    }


def _prepare_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def _to_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _from_json(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("Expected JSON object metadata.")
    return decoded

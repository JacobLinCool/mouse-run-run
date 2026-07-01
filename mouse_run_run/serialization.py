import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


CHECKPOINT_FORMAT = "mouse-run-run-checkpoint-v1"
ROLLOUT_FORMAT = "mouse-run-run-rollout-v1"


def save_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    chaser_state: Mapping[str, torch.Tensor],
    explorer_state: Mapping[str, torch.Tensor],
) -> None:
    tensors = {
        **_prefix_state("chaser", chaser_state),
        **_prefix_state("explorer", explorer_state),
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "format": CHECKPOINT_FORMAT,
            "config": _to_json(config),
            "metrics": _to_json(metrics),
        },
    )


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

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from mouse_run_run.serialization import CHECKPOINT_FORMAT, read_metadata


class NonFiniteTrainingError(RuntimeError):
    def __init__(self, *, location: str, name: str, detail: str) -> None:
        self.location = location
        self.name = name
        self.detail = detail
        super().__init__(f"non-finite value at {location}: {name}: {detail}")


@dataclass(frozen=True)
class CheckpointHealth:
    path: str
    ok: bool
    format_ok: bool
    tensor_count: int
    nonfinite_tensor_count: int
    max_abs: float
    failures: tuple[dict[str, Any], ...]


def assert_finite_scalar(value: float, *, location: str, name: str) -> None:
    if not math.isfinite(float(value)):
        raise NonFiniteTrainingError(
            location=location,
            name=name,
            detail=f"value={value}",
        )


def assert_finite_tensor(tensor: torch.Tensor, *, location: str, name: str) -> None:
    if not tensor.dtype.is_floating_point:
        return
    if torch.isfinite(tensor).all().item():
        return
    finite = torch.isfinite(tensor)
    nonfinite_count = int((~finite).sum().item())
    total = tensor.numel()
    raise NonFiniteTrainingError(
        location=location,
        name=name,
        detail=f"nonfinite={nonfinite_count}/{total}",
    )


def assert_finite_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    location: str,
) -> None:
    for name, tensor in tensors.items():
        assert_finite_tensor(tensor, location=location, name=name)


def assert_finite_module(module: torch.nn.Module, *, location: str, prefix: str) -> None:
    for name, tensor in module.state_dict().items():
        assert_finite_tensor(tensor, location=location, name=f"{prefix}.{name}")


def assert_finite_metrics(metrics: dict[str, float], *, location: str) -> None:
    for name, value in metrics.items():
        assert_finite_scalar(float(value), location=location, name=name)


def module_max_abs(*modules: torch.nn.Module) -> float:
    values = []
    for module in modules:
        for tensor in module.state_dict().values():
            if tensor.dtype.is_floating_point and tensor.numel():
                values.append(tensor.detach().abs().max())
    if not values:
        return 0.0
    return float(torch.stack(values).max().item())


def checkpoint_health(path: Path) -> CheckpointHealth:
    failures: list[dict[str, Any]] = []
    try:
        metadata = read_metadata(path)
    except Exception as exc:
        return CheckpointHealth(
            path=str(path),
            ok=False,
            format_ok=False,
            tensor_count=0,
            nonfinite_tensor_count=0,
            max_abs=0.0,
            failures=({"name": "__metadata__", "error": repr(exc)},),
        )

    format_ok = metadata.get("format") == CHECKPOINT_FORMAT
    if not format_ok:
        failures.append(
            {
                "name": "__format__",
                "error": f"expected {CHECKPOINT_FORMAT}, got {metadata.get('format')}",
            }
        )

    tensor_count = 0
    nonfinite_tensor_count = 0
    max_abs = 0.0
    if format_ok:
        try:
            tensors = load_file(str(path), device="cpu")
        except Exception as exc:
            failures.append({"name": "__load__", "error": repr(exc)})
            tensors = {}
        tensor_count = len(tensors)
        for name, tensor in tensors.items():
            if not tensor.dtype.is_floating_point:
                continue
            finite = torch.isfinite(tensor)
            if not finite.all().item():
                nonfinite_count = int((~finite).sum().item())
                nonfinite_tensor_count += 1
                failures.append(
                    {
                        "name": name,
                        "nonfinite_count": nonfinite_count,
                        "element_count": tensor.numel(),
                    }
                )
                continue
            if tensor.numel():
                max_abs = max(max_abs, float(tensor.abs().max().item()))

    ok = format_ok and not failures and nonfinite_tensor_count == 0
    return CheckpointHealth(
        path=str(path),
        ok=ok,
        format_ok=format_ok,
        tensor_count=tensor_count,
        nonfinite_tensor_count=nonfinite_tensor_count,
        max_abs=max_abs,
        failures=tuple(failures),
    )


def require_healthy_checkpoint(path: Path) -> CheckpointHealth:
    health = checkpoint_health(path)
    if not health.ok:
        raise ValueError(f"invalid checkpoint {path}: {health.failures}")
    return health

"""Small tensor-only value types shared across the execution core."""

from __future__ import annotations

from typing import NewType, TypeAlias

import torch


AgentId = NewType("AgentId", str)
TensorMap: TypeAlias = dict[str, torch.Tensor]
TensorTree: TypeAlias = dict[str, torch.Tensor]


def require_agent_keys(
    values: dict[AgentId, object],
    agent_ids: tuple[AgentId, ...],
    *,
    label: str,
) -> None:
    """Require an exact, ordered experiment agent set."""
    if tuple(values) != agent_ids:
        raise ValueError(
            f"{label} keys {tuple(values)!r} do not match experiment agents {agent_ids!r}"
        )


def require_tensor_tree(tree: TensorTree, *, label: str) -> None:
    """Reject opaque/pickled state before it reaches artifacts or simulation."""
    for key, value in tree.items():
        if not key or "/" in key:
            raise ValueError(f"{label} has invalid tensor key {key!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}.{key} is not a tensor: {type(value)!r}")

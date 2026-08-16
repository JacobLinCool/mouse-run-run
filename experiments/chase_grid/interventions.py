"""Factories for shared/unique chase-grid causal interventions."""

from __future__ import annotations

from typing import Literal

from experiments.chase_grid.environment import CHASER
from mouse_run_run.analyses.plsc import FittedPLSC
from mouse_run_run.core.intervention import InterventionTarget
from mouse_run_run.interventions import SubspaceIntervention


Condition = Literal[
    "shared_removed",
    "shared_only",
    "top_unique_removed",
    "random_unique_removed",
]


def build_intervention(
    fitted: FittedPLSC,
    *,
    condition: Condition,
    target: InterventionTarget,
) -> SubspaceIntervention:
    if condition == "shared_removed":
        basis_name, operation = "shared", "remove"
    elif condition == "shared_only":
        basis_name, operation = "shared", "keep"
    elif condition == "top_unique_removed":
        basis_name, operation = "top_unique", "remove"
    elif condition == "random_unique_removed":
        basis_name, operation = "random_unique", "remove"
    else:
        raise ValueError(f"unsupported chase-grid intervention {condition!r}")
    return SubspaceIntervention(
        name=condition,
        subspace=fitted.subspace(CHASER, basis_name),
        target=target,
        operation=operation,
    )

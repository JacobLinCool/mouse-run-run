"""Convenient analysis recipes for chase-grid rollouts."""

from __future__ import annotations

from experiments.chase_grid.environment import CHASER, EXPLORER
from mouse_run_run.analyses.plsc import PLSCConfig, PermutationThreshold


def plsc_config(
    *,
    permutations: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    control_rank: int | None = 10,
) -> PLSCConfig:
    return PLSCConfig(
        agent_a=CHASER,
        agent_b=EXPLORER,
        site="hidden",
        selection=PermutationThreshold(
            alpha=alpha,
            permutations=permutations,
            null_model="episode_shuffle",
            seed=seed,
        ),
        control_rank=control_rank,
        control_seed=seed,
    )

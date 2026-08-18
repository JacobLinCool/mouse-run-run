"""Seeded group statistics shared by figure sources.

The house convention for inference in this repository is a permutation null
computed in Torch, as already used by the PLSC and decoder analyses. Group
comparisons for figures follow the same convention instead of pulling in a
separate statistics dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Sequence

import torch


@dataclass(frozen=True)
class GroupComparison:
    left_n: int
    right_n: int
    left_mean: float
    right_mean: float
    difference: float
    p_value: float
    permutations: int
    exact: bool

    @property
    def stars(self) -> str:
        return significance_stars(self.p_value)


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error of the mean; sem is 0.0 for a single sample."""

    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("mean_sem requires at least one value")
    if tensor.numel() == 1:
        return float(tensor[0]), 0.0
    return float(tensor.mean()), float(tensor.std(unbiased=True) / tensor.numel() ** 0.5)


def permutation_comparison(
    left: Sequence[float],
    right: Sequence[float],
    *,
    permutations: int = 10_000,
    seed: int = 0,
) -> GroupComparison:
    """Two-sided permutation test on the difference of group means.

    Every distinct split is enumerated when the group sizes make that cheaper
    than sampling, which keeps small-seed studies reproducible rather than
    dependent on the sampling stream.
    """

    left_tensor = torch.tensor(list(left), dtype=torch.float64)
    right_tensor = torch.tensor(list(right), dtype=torch.float64)
    if left_tensor.numel() == 0 or right_tensor.numel() == 0:
        raise ValueError("permutation_comparison requires two non-empty groups")
    pooled = torch.cat([left_tensor, right_tensor])
    total = pooled.numel()
    left_n = left_tensor.numel()
    observed = float(left_tensor.mean() - right_tensor.mean())
    exact = comb(total, left_n) <= permutations
    if exact:
        assignments = torch.tensor(
            list(_combinations(total, left_n)), dtype=torch.long
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        assignments = torch.stack(
            [torch.randperm(total, generator=generator)[:left_n] for _ in range(permutations)]
        )
    means = pooled[assignments].mean(dim=1)
    null = means * total / (total - left_n) - pooled.sum() / (total - left_n)
    draws = int(null.numel())
    extreme = int((null.abs() >= abs(observed) - 1e-12).sum())
    # Full enumeration already contains the observed split; a sampled null does
    # not, so it takes the usual add-one correction.
    p_value = extreme / draws if exact else (1 + extreme) / (draws + 1)
    return GroupComparison(
        left_n=int(left_n),
        right_n=int(right_tensor.numel()),
        left_mean=float(left_tensor.mean()),
        right_mean=float(right_tensor.mean()),
        difference=observed,
        p_value=min(p_value, 1.0),
        permutations=draws,
        exact=exact,
    )


def significance_stars(p_value: float) -> str:
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def _combinations(total: int, size: int) -> list[tuple[int, ...]]:
    return list(combinations(range(total), size))

"""Positive/negative controls for the merged PLSC estimator.

Coupled pairs (explorer a linear function of the chaser plus small noise) must
produce significant shared dimensions under every null model; independent or
permuted pairs must produce none. A known covariance structure (one exactly
shared unit) pins the expected leading singular value and latent correlation.
All data is seeded so the assertions are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_run_run.analysis import plsc_shared_dimensions
from mouse_run_run.plsc import compute_plsc


def _pooled_coupled_pair(
    n: int, units: int, *, seed: int, noise: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Generator]:
    generator = torch.Generator().manual_seed(seed)
    chaser = torch.randn(n, units, generator=generator)
    mixing = torch.randn(units, units, generator=generator)
    explorer = chaser @ mixing + noise * torch.randn(n, units, generator=generator)
    return chaser, explorer, generator


def _episode_pair(
    episodes: int, steps: int, units: int, *, coupled: bool, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((units, units))
    chaser_episodes, explorer_episodes = [], []
    for _ in range(episodes):
        chaser = rng.standard_normal((steps, units))
        if coupled:
            explorer = chaser @ mixing + 0.1 * rng.standard_normal((steps, units))
        else:
            explorer = rng.standard_normal((steps, units))
        chaser_episodes.append(chaser)
        explorer_episodes.append(explorer)
    return chaser_episodes, explorer_episodes


def test_coupled_pair_significant_under_temporal_permutation() -> None:
    chaser, explorer, _ = _pooled_coupled_pair(400, 6, seed=0)

    result = compute_plsc(chaser, explorer, permutations=100, alpha=0.05, seed=0)

    assert result.significant_dims > 0
    assert result.leading_significant_dims > 0
    assert float(result.latent_correlations[0]) > 0.9


def test_permuted_pair_not_significant_under_temporal_permutation() -> None:
    # Coupled data with the pairing destroyed by a row permutation: nothing
    # moment-to-moment survives, so no dimension may beat the null.
    chaser, explorer, generator = _pooled_coupled_pair(400, 6, seed=3)
    permutation = torch.randperm(400, generator=generator)

    result = compute_plsc(chaser, explorer[permutation], permutations=100, alpha=0.05, seed=0)

    assert result.significant_dims == 0


def test_coupled_episodes_significant_under_each_episode_null() -> None:
    chaser_episodes, explorer_episodes = _episode_pair(10, 60, 6, coupled=True, seed=2)

    result = plsc_shared_dimensions(
        chaser_episodes, explorer_episodes, permutations=100, seed=0
    )

    for null_model in ("episode_shuffle", "circular_shift"):
        assert result.nulls[null_model].n_significant > 0, null_model
    assert result.n_significant == result.nulls[result.headline_null].n_significant
    assert result.top_dim_correlation > 0.9


def test_independent_episodes_not_significant_under_each_episode_null() -> None:
    chaser_episodes, explorer_episodes = _episode_pair(10, 60, 6, coupled=False, seed=1)

    result = plsc_shared_dimensions(
        chaser_episodes, explorer_episodes, permutations=100, seed=0
    )

    for null_model in ("episode_shuffle", "circular_shift"):
        assert result.nulls[null_model].n_significant == 0, null_model
    assert result.n_significant == 0


def test_known_shared_unit_yields_unit_leading_correlation() -> None:
    # Explorer unit 0 copies chaser unit 0 exactly; all other units are
    # independent noise. After z-scoring, the (0, 0) cross-covariance entry is
    # exactly 1, so the leading singular value is ~1 and the leading latent
    # pair is (numerically) perfectly correlated.
    generator = torch.Generator().manual_seed(4)
    steps, episodes, units = 60, 15, 4
    chaser = torch.randn(steps * episodes, units, generator=generator)
    explorer = torch.cat(
        [chaser[:, :1], torch.randn(steps * episodes, units - 1, generator=generator)],
        dim=1,
    )

    pooled = compute_plsc(chaser, explorer, permutations=16, alpha=0.05, seed=0)
    assert float(pooled.singular_values[0]) == pytest.approx(1.0, abs=0.05)
    assert float(pooled.latent_correlations[0]) > 0.99

    chaser_np = chaser.double().numpy()
    explorer_np = explorer.double().numpy()
    per_episode = plsc_shared_dimensions(
        [chaser_np[i * steps : (i + 1) * steps] for i in range(episodes)],
        [explorer_np[i * steps : (i + 1) * steps] for i in range(episodes)],
        permutations=16,
        seed=0,
    )
    assert per_episode.singular_values[0] == pytest.approx(1.0, abs=0.05)
    assert abs(per_episode.top_dim_correlation) > 0.99

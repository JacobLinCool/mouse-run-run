"""Partial Least Squares Correlation (PLSC) of paired agent hidden states.

This is the cross-agent shared-dimension method from Zhang et al. (2025),
Fig. 5/6 (claim C3). Given two agents' time-aligned RNN hidden states, it
z-scores each unit, forms the cross-covariance matrix
``R = X_chaser^T X_explorer / (n - 1)``, and takes its SVD. The singular values
measure shared-dimension strength; the left/right singular vectors are the two
agents' shared bases. Significance comes from a permutation null — one of
``temporal_permutation``, ``episode_shuffle``, or ``circular_shift`` (see
``null_singular_values``) — that builds a per-rank singular-value distribution
from re-paired data.

The shared subspace is spanned by the significant singular vectors; the unique
subspace is its orthogonal complement. Because the bases are orthonormal, an
agent's variance (or a single hidden state's squared norm) splits exactly into
shared and unique parts.

This module is the single home of the estimator core (``zscore_columns``,
``cross_covariance``, ``cross_covariance_svd``, ``null_singular_values``).
``compute_plsc`` below is the pooled-sample entry point with the
temporal-permutation null, used by ``scripts/analysis/analyze_shared_neural.py``
and the interactive viewer; ``mouse_run_run.analysis.plsc_shared_dimensions``
is the per-episode entry point with the episode-based nulls, built on the same
core.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

TOP_K = 20

NULL_MODELS = ("temporal_permutation", "episode_shuffle", "circular_shift")


def zscore_columns(
    matrix: torch.Tensor,
    *,
    correction: int = 1,
    epsilon: float = 1e-8,
    degenerate_to_one: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Column-wise z-score of ``matrix`` (n, H); returns (zscored, mean, std).

    ``correction`` is the std's Bessel correction (1 = sample std, the pooled
    ``compute_plsc`` convention; 0 = population std, the per-episode
    convention). Near-constant columns are handled by clamping the std to
    ``epsilon`` by default, or — with ``degenerate_to_one`` — by replacing
    sub-``epsilon`` stds with 1.0 so those columns map to exactly zero.
    """
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, correction=correction)
    if degenerate_to_one:
        std = torch.where(std < epsilon, torch.ones_like(std), std)
    else:
        std = std.clamp_min(epsilon)
    return (matrix - mean) / std, mean, std


def cross_covariance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``R = x^T y / (n - 1)`` for paired z-scored samples (n, Hx) and (n, Hy)."""
    return (x.T @ y) / (x.shape[0] - 1)


def cross_covariance_svd(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD of the cross-covariance; returns ``(R, U, singular_values, V)``.

    ``U`` (Hx, rank) and ``V`` (Hy, rank) hold the two agents' shared bases as
    columns; ``rank = min(Hx, Hy)``.
    """
    cross = cross_covariance(x, y)
    u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
    return cross, u, singular, vh.T


def null_singular_values(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    null_model: str,
    permutations: int,
    seed: int,
    episode_lengths: Sequence[int] | None = None,
    min_shift: int = 10,
) -> torch.Tensor:
    """Permutation-null singular values, shape ``(permutations, rank)``.

    - ``temporal_permutation``: shuffle ``y``'s pooled timepoints so the
      moment-to-moment pairing is destroyed while each agent's marginals are
      preserved. A fixed torch generator keeps the null reproducible.
    - ``episode_shuffle``: re-pair whole episodes of ``y`` (requires
      ``episode_lengths``). Both sides keep their within-episode temporal
      structure aligned from t=0, so shared time-in-episode dynamics stay in
      the null and only episode-specific interaction counts as shared.
    - ``circular_shift``: roll each episode of ``y`` by a random offset of at
      least ``min_shift`` steps, breaking all temporal correspondence
      including time-locking.

    The temporal null draws from ``torch.Generator`` and the episode nulls
    from ``numpy.random.RandomState``; each permutation stream is unchanged
    from the pre-merge per-caller implementations so existing analyses
    reproduce exactly.
    """
    if null_model not in NULL_MODELS:
        raise ValueError(f"unknown null model: {null_model}")
    n = x.shape[0]
    rank = min(x.shape[1], y.shape[1])
    null = torch.empty(permutations, rank, dtype=x.dtype)
    if null_model == "temporal_permutation":
        generator = torch.Generator().manual_seed(seed)
        for index in range(permutations):
            perm = torch.randperm(n, generator=generator)
            null[index] = torch.linalg.svdvals(cross_covariance(x, y[perm]))
        return null
    if episode_lengths is None:
        raise ValueError(f"{null_model} null requires episode_lengths")
    lengths = list(episode_lengths)
    offsets = np.cumsum([0, *lengths[:-1]])
    blocks = [y[start : start + length] for start, length in zip(offsets, lengths, strict=True)]
    rng = np.random.RandomState(seed)
    for p in range(permutations):
        if null_model == "episode_shuffle":
            order = rng.permutation(len(blocks))
            shuffled = torch.cat([blocks[index] for index in order])
        else:
            shuffled = torch.cat(
                [
                    torch.roll(
                        block,
                        int(rng.randint(min_shift, max(len(block) - min_shift, min_shift + 1))),
                        dims=0,
                    )
                    for block in blocks
                ]
            )
        null[p] = torch.linalg.svdvals(cross_covariance(x, shuffled))
    return null


@dataclass(frozen=True)
class PLSCResult:
    singular_values: torch.Tensor  # (rank,)
    null: torch.Tensor  # (permutations, rank)
    null_percentile: torch.Tensor  # (rank,) the (1 - alpha) null quantile per rank
    covariance_explained: torch.Tensor  # (rank,) singular_value^2 normalized
    latent_correlations: torch.Tensor  # (<=TOP_K,) paired latent Pearson r
    p_values: torch.Tensor  # (rank,)
    significant: torch.Tensor  # (rank,) bool
    significant_dims: int
    leading_significant_dims: int
    cross_covariance: torch.Tensor  # (H, H) R
    chaser_basis: torch.Tensor  # (H, rank) U
    explorer_basis: torch.Tensor  # (H, rank) V
    chaser_mean: torch.Tensor  # (H,)
    chaser_std: torch.Tensor  # (H,)
    explorer_mean: torch.Tensor  # (H,)
    explorer_std: torch.Tensor  # (H,)
    latent_c0: torch.Tensor  # (n,) chaser latent variable 1
    latent_e0: torch.Tensor  # (n,) explorer latent variable 1
    alpha: float


def compute_plsc(
    chaser: torch.Tensor,
    explorer: torch.Tensor,
    *,
    permutations: int,
    alpha: float,
    seed: int,
    correlation_k: int = TOP_K,
) -> PLSCResult:
    """Run PLSC on pooled paired samples ``chaser`` and ``explorer`` (n, H)."""
    chaser = chaser.float()
    explorer = explorer.float()
    xc, chaser_mean, chaser_std = zscore_columns(chaser)
    xe, explorer_mean, explorer_std = zscore_columns(explorer)

    cross, u, singular, v = cross_covariance_svd(xc, xe)
    latent_c = xc @ u
    latent_e = xe @ v
    rank = singular.shape[0]

    correlations = torch.tensor(
        [pearson(latent_c[:, i], latent_e[:, i]) for i in range(min(correlation_k, rank))]
    )
    covariance_explained = singular.square() / singular.square().sum().clamp_min(1e-12)

    null = null_singular_values(
        xc,
        xe,
        null_model="temporal_permutation",
        permutations=permutations,
        seed=seed,
    )

    null_percentile = torch.quantile(null, 1.0 - alpha, dim=0)
    significant = singular > null_percentile
    p_values = (1.0 + (null >= singular.unsqueeze(0)).sum(dim=0)) / (permutations + 1)
    leading = 0
    for flag in significant.tolist():
        if not flag:
            break
        leading += 1

    return PLSCResult(
        singular_values=singular,
        null=null,
        null_percentile=null_percentile,
        covariance_explained=covariance_explained,
        latent_correlations=correlations,
        p_values=p_values,
        significant=significant,
        significant_dims=int(significant.sum()),
        leading_significant_dims=leading,
        cross_covariance=cross,
        chaser_basis=u,
        explorer_basis=v,
        chaser_mean=chaser_mean,
        chaser_std=chaser_std,
        explorer_mean=explorer_mean,
        explorer_std=explorer_std,
        latent_c0=latent_c[:, 0],
        latent_e0=latent_e[:, 0],
        alpha=alpha,
    )


def shared_variance_fraction(
    zscored: torch.Tensor,
    basis: torch.Tensor,
    dims: int,
) -> float:
    """Fraction of an agent's total (z-scored) variance in the shared subspace.

    The basis is orthonormal, so projecting onto its first ``dims`` columns and
    summing the projected variances gives the shared variance; the total is the
    sum over all columns (equivalently the z-scored variance, i.e. the unit
    count). Returns 0 when the shared subspace is empty.
    """
    total = zscored.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    if dims <= 0:
        return 0.0
    coords = zscored @ basis[:, :dims]
    shared = coords.var(dim=0, unbiased=False).sum()
    return float((shared / total).clamp(0.0, 1.0))


def project_norms(
    zscored_episode: torch.Tensor,
    basis: torch.Tensor,
    dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step shared and unique norms of one z-scored episode (T, H).

    ``||h||^2 = ||shared||^2 + ||unique||^2`` exactly, because the shared basis
    is orthonormal and the unique subspace is its orthogonal complement.
    """
    total_norm = zscored_episode.norm(dim=1)
    if dims <= 0:
        return torch.zeros_like(total_norm), total_norm
    coords = zscored_episode @ basis[:, :dims]
    shared_norm = coords.norm(dim=1)
    unique_norm = (total_norm.square() - shared_norm.square()).clamp_min(0.0).sqrt()
    return shared_norm, unique_norm


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denominator)

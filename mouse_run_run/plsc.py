"""Partial Least Squares Correlation (PLSC) of paired agent hidden states.

This is the cross-agent shared-dimension method from Zhang et al. (2025),
Fig. 5/6 (claim C3). Given two agents' time-aligned RNN hidden states, it
z-scores each unit, forms the cross-covariance matrix
``R = X_chaser^T X_explorer / (n - 1)``, and takes its SVD. The singular values
measure shared-dimension strength; the left/right singular vectors are the two
agents' shared bases. Significance comes from a temporal permutation (one
agent's timepoints shuffled) that builds a per-rank null distribution.

The shared subspace is spanned by the significant singular vectors; the unique
subspace is its orthogonal complement. Because the bases are orthonormal, an
agent's variance (or a single hidden state's squared norm) splits exactly into
shared and unique parts.

Both ``scripts/analysis/analyze_shared_neural.py`` and the interactive viewer
import from here so the method has a single definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TOP_K = 20


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
    chaser_mean = chaser.mean(dim=0)
    chaser_std = chaser.std(dim=0).clamp_min(1e-8)
    explorer_mean = explorer.mean(dim=0)
    explorer_std = explorer.std(dim=0).clamp_min(1e-8)
    xc = (chaser - chaser_mean) / chaser_std
    xe = (explorer - explorer_mean) / explorer_std
    n = xc.shape[0]

    cross = (xc.T @ xe) / (n - 1)
    u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
    v = vh.T
    latent_c = xc @ u
    latent_e = xe @ v
    rank = singular.shape[0]

    correlations = torch.tensor(
        [pearson(latent_c[:, i], latent_e[:, i]) for i in range(min(correlation_k, rank))]
    )
    covariance_explained = singular.square() / singular.square().sum().clamp_min(1e-12)

    # Temporal permutation: shuffle the explorer's timepoints so the moment-to-
    # moment pairing is destroyed while each agent's marginals are preserved. A
    # fixed generator keeps the null reproducible.
    generator = torch.Generator().manual_seed(seed)
    null = torch.empty(permutations, rank)
    for index in range(permutations):
        perm = torch.randperm(n, generator=generator)
        cross_perm = (xc.T @ xe[perm]) / (n - 1)
        null[index] = torch.linalg.svdvals(cross_perm)

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


def zscore(matrix: torch.Tensor) -> torch.Tensor:
    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (matrix - mean) / std


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denominator)

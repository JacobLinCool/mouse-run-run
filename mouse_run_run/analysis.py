"""Neural-representation analyses for recorded rollouts.

Implements the paper's shared-dimension and event-decoding analyses plus a
cross-model representation comparison:

- PLSC: partial least squares correlation between the two agents' hidden
  states, with circular-shift permutation significance.
- Event decoding: linear-classifier balanced accuracy for collision and
  approach/escape events from single-agent hidden states.
- Replay + linear CKA: teacher-force one rollout's observations through any
  checkpoint (same or different architecture) and compare representations.

All analyses exclude paper-degenerate episodes and operate on the rollout
schema v3 alignment: hidden[t] produced action[t] from state s_t, event flags
at t describe the transition t -> t+1, visible[t] is visibility in s_t.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from mouse_run_run.plsc import cross_covariance_svd, null_singular_values, zscore_columns
from mouse_run_run.policy import build_policy
from mouse_run_run.serialization import ROLLOUT_FORMAT, load_checkpoint, read_metadata


def load_rollout(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return (metadata, tensors) with the ``rollout.`` prefix stripped."""
    metadata = read_metadata(path)
    if metadata.get("format") != ROLLOUT_FORMAT:
        raise ValueError(f"Unsupported rollout format in {path}: {metadata.get('format')}")
    tensors = {
        key.removeprefix("rollout."): value
        for key, value in load_file(str(path)).items()
        if key.startswith("rollout.")
    }
    return json.loads(metadata["metadata"]), tensors


def valid_episode_indices(tensors: dict[str, torch.Tensor]) -> list[int]:
    degenerate = tensors["episode_degenerate"].bool()
    return [index for index in range(degenerate.shape[0]) if not degenerate[index]]


def episode_hidden_pairs(
    tensors: dict[str, torch.Tensor],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-episode (steps, hidden) float64 arrays, degenerate episodes excluded."""
    chaser = tensors["chaser_hidden"].double().numpy()
    explorer = tensors["explorer_hidden"].double().numpy()
    indices = valid_episode_indices(tensors)
    return (
        [chaser[:, index] for index in indices],
        [explorer[:, index] for index in indices],
    )


# ---------------------------------------------------------------------------
# PLSC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLSCNull:
    thresholds: list[float]
    significant: list[bool]
    n_significant: int


@dataclass(frozen=True)
class SharedDimensionsResult:
    singular_values: list[float]
    nulls: dict[str, PLSCNull]
    n_significant: int
    top_dim_correlation: float
    n_samples: int
    n_episodes: int
    permutations: int
    headline_null: str


def _zscore(matrix: np.ndarray) -> np.ndarray:
    std = matrix.std(axis=0)
    std[std < 1e-12] = 1.0
    return (matrix - matrix.mean(axis=0)) / std


def plsc_shared_dimensions(
    chaser_episodes: list[np.ndarray],
    explorer_episodes: list[np.ndarray],
    *,
    permutations: int = 200,
    min_shift: int = 10,
    percentile: float = 95.0,
    seed: int = 0,
    null_models: tuple[str, ...] = ("episode_shuffle", "circular_shift"),
    headline_null: str = "episode_shuffle",
    subtract_time_mean: bool = False,
) -> SharedDimensionsResult:
    """Shared cross-agent dimensions via SVD of the cross-covariance.

    The estimator core (z-scoring, cross-covariance, SVD, permutation nulls)
    lives in ``mouse_run_run.plsc``; this entry point pools per-episode hidden
    states and reports the episode-based nulls. Two permutation nulls,
    significant when the rank-k singular value exceeds the given percentile of
    the permuted rank-k values:

    - episode_shuffle (headline): re-pairs the explorer's episodes with other
      episodes' chaser data. Both sides keep their within-episode temporal
      structure aligned from t=0, so shared time-in-episode dynamics (e.g. a
      hidden-state "clock" growing from the zero init) stay in the null and
      only episode-specific interaction counts as shared. This is the correct
      null for the inter-agent coupling claim.
    - circular_shift: shifts each explorer episode by a random offset,
      breaking all temporal correspondence including time-locking. Detects
      any time-locked structure and is reported as the weaker control; on
      its own it flags trivially time-locked dimensions as shared.
    """
    if len(chaser_episodes) != len(explorer_episodes) or not chaser_episodes:
        raise ValueError("need matching, non-empty episode lists")
    if headline_null not in null_models:
        raise ValueError("headline_null must be one of null_models")
    lengths = [len(episode) for episode in explorer_episodes]
    equal_length = len(set(lengths)) == 1
    if "episode_shuffle" in null_models and not equal_length:
        raise ValueError("episode_shuffle null requires equal-length episodes")
    if subtract_time_mean and not equal_length:
        raise ValueError("subtract_time_mean requires equal-length episodes")
    chaser_stack = list(chaser_episodes)
    explorer_stack = list(explorer_episodes)
    if subtract_time_mean:
        # Remove the across-episode mean at each timestep (the time-locked
        # "evoked" response: positional embeddings, hidden-state clocks, any
        # structure identical across episodes at a given t). PLSC then runs on
        # the episode-specific residual, where genuine interaction lives, so
        # the effect size is comparable across architectures with different
        # deterministic temporal scaffolds.
        chaser_mean = np.mean(chaser_stack, axis=0)
        explorer_mean = np.mean(explorer_stack, axis=0)
        chaser_stack = [episode - chaser_mean for episode in chaser_stack]
        explorer_stack = [episode - explorer_mean for episode in explorer_stack]
    X, _, _ = zscore_columns(
        torch.from_numpy(np.concatenate(chaser_stack)),
        correction=0,
        epsilon=1e-12,
        degenerate_to_one=True,
    )
    Y, _, _ = zscore_columns(
        torch.from_numpy(np.concatenate(explorer_stack)),
        correction=0,
        epsilon=1e-12,
        degenerate_to_one=True,
    )
    n = X.shape[0]

    _, U, S_torch, V = cross_covariance_svd(X, Y)
    S = S_torch.numpy()

    # Permutation singular values only feed percentile thresholds; float32
    # halves the matmul cost without affecting the real SVD above.
    X32 = X.float()
    Y32 = Y.float()
    nulls: dict[str, PLSCNull] = {}
    for null_model in null_models:
        permuted = null_singular_values(
            X32,
            Y32,
            null_model=null_model,
            permutations=permutations,
            seed=seed,
            episode_lengths=lengths,
            min_shift=min_shift,
        )
        thresholds = np.percentile(permuted.numpy().astype(np.float64), percentile, axis=0)
        significant = S > thresholds
        nulls[null_model] = PLSCNull(
            thresholds=[float(v) for v in thresholds],
            significant=[bool(v) for v in significant],
            n_significant=int(significant.sum()),
        )

    x_scores = (X @ U[:, 0]).numpy()
    y_scores = (Y @ V[:, 0]).numpy()
    top_correlation = float(np.corrcoef(x_scores, y_scores)[0, 1])
    return SharedDimensionsResult(
        singular_values=[float(v) for v in S],
        nulls=nulls,
        n_significant=nulls[headline_null].n_significant,
        top_dim_correlation=top_correlation,
        n_samples=n,
        n_episodes=len(chaser_episodes),
        permutations=permutations,
        headline_null=headline_null,
    )


# ---------------------------------------------------------------------------
# Event decoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodingResult:
    balanced_accuracy: float | None
    shuffled_accuracy: float | None
    n_samples: int
    n_positive: int
    skipped_reason: str | None = None


def decode_balanced_accuracy(
    hidden: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
    max_samples: int = 20_000,
    min_positive: int = 50,
    folds: int = 5,
) -> DecodingResult:
    """Cross-validated linear-classifier balanced accuracy, with a
    label-shuffled control run through the identical pipeline."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    labels = labels.astype(np.int64)
    n_positive = int(labels.sum())
    if n_positive < min_positive or n_positive > len(labels) - min_positive:
        return DecodingResult(
            balanced_accuracy=None,
            shuffled_accuracy=None,
            n_samples=len(labels),
            n_positive=n_positive,
            skipped_reason=f"class imbalance beyond limits (positive={n_positive})",
        )

    rng = np.random.RandomState(seed)
    if len(labels) > max_samples:
        chosen = rng.choice(len(labels), size=max_samples, replace=False)
        hidden = hidden[chosen]
        labels = labels[chosen]
        n_positive = int(labels.sum())
        if n_positive < min_positive:
            return DecodingResult(
                balanced_accuracy=None,
                shuffled_accuracy=None,
                n_samples=len(labels),
                n_positive=n_positive,
                skipped_reason="positive class lost in subsampling",
            )

    def run(y: np.ndarray) -> float:
        pipeline = make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", dual=False, max_iter=5000, random_state=seed),
        )
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, hidden, y, cv=splitter, scoring="balanced_accuracy", n_jobs=-1
        )
        return float(scores.mean())

    accuracy = run(labels)
    shuffled = run(rng.permutation(labels))
    return DecodingResult(
        balanced_accuracy=accuracy,
        shuffled_accuracy=shuffled,
        n_samples=len(labels),
        n_positive=n_positive,
    )


def decoding_targets(
    tensors: dict[str, torch.Tensor],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(hidden, labels) per paper decoding target, pooled over valid episodes.

    Chaser hidden decodes collision and partner escape; explorer hidden
    decodes collision and partner approach. Approach/escape targets use only
    timesteps where the partner is visible in the pre-action state s_t.
    """
    indices = valid_episode_indices(tensors)
    steps = tensors["chaser_hidden"].shape[0]

    chaser_hidden = tensors["chaser_hidden"][:, indices].float().numpy().reshape(-1, tensors["chaser_hidden"].shape[-1])
    explorer_hidden = tensors["explorer_hidden"][:, indices].float().numpy().reshape(-1, tensors["explorer_hidden"].shape[-1])
    collision = tensors["collision"][:, indices].numpy().reshape(-1)
    escape = tensors["explorer_escape"][:, indices].numpy().reshape(-1)
    approach = tensors["chaser_approach"][:, indices].numpy().reshape(-1)
    # visible[t] over the pre-action states s_0..s_{T-1}.
    chaser_visible = tensors["chaser_partner_visible"][:steps, indices].numpy().reshape(-1)
    explorer_visible = tensors["explorer_partner_visible"][:steps, indices].numpy().reshape(-1)

    return {
        "chaser_collision": (chaser_hidden, collision),
        "chaser_partner_escape": (chaser_hidden[chaser_visible], escape[chaser_visible]),
        "explorer_collision": (explorer_hidden, collision),
        "explorer_partner_approach": (
            explorer_hidden[explorer_visible],
            approach[explorer_visible],
        ),
    }


# ---------------------------------------------------------------------------
# Replay + CKA
# ---------------------------------------------------------------------------


def replay_hidden(checkpoint: Path, observations: torch.Tensor, *, agent: str) -> torch.Tensor:
    """Teacher-force recorded observations through a checkpoint's network.

    observations: (steps, episodes, obs). Returns (steps, episodes, hidden)
    from the requested agent's network ("chaser" or "explorer").
    """
    config, _, chaser_state, explorer_state = load_checkpoint(checkpoint)
    policy = build_policy(
        config.get("architecture", "rnn"),
        observations.shape[-1],
        hidden_size=config["hidden_size"],
    )
    policy.load_state_dict(chaser_state if agent == "chaser" else explorer_state)
    policy.eval()
    with torch.no_grad():
        return policy.hidden_sequence(observations.float())


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Biased linear CKA between (n, d1) and (n, d2) feature matrices."""
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    normalizer = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if normalizer < 1e-12:
        return 0.0
    return float(cross / normalizer)


def replay_cka(
    rollout_tensors: dict[str, torch.Tensor],
    checkpoint: Path,
    *,
    agent: str = "chaser",
) -> float:
    """CKA between a rollout's recorded hidden states and the states another
    checkpoint's network produces on the identical observation stream."""
    indices = valid_episode_indices(rollout_tensors)
    observations = rollout_tensors[f"{agent}_observation"][:, indices]
    recorded = rollout_tensors[f"{agent}_hidden"][:, indices]
    replayed = replay_hidden(checkpoint, observations, agent=agent)
    recorded_flat = recorded.reshape(-1, recorded.shape[-1]).double().numpy()
    replayed_flat = replayed.reshape(-1, replayed.shape[-1]).double().numpy()
    return linear_cka(recorded_flat, replayed_flat)


# ---------------------------------------------------------------------------
# C4: partner representation in the neural action subspace
# ---------------------------------------------------------------------------


def neural_action_subspace(hidden: np.ndarray, action_weight: np.ndarray) -> np.ndarray:
    """z_t = W_action^T W_action h_t : the action-relevant projection of the
    hidden state (rank <= number of actions). ``hidden`` is (n, H),
    ``action_weight`` is (A, H)."""
    gram = action_weight.T @ action_weight
    return hidden @ gram.T


def _behavior_predictors(
    tensors: dict[str, torch.Tensor],
    indices: list[int],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix and a boolean mask of the partner-behaviour columns.

    Focal agent is the chaser; the partner is the explorer. Self columns:
    collision, self x/y, self action (one-hot), self approach/new-field.
    Partner columns: partner x/y, partner action (one-hot), partner
    escape/new-field. All at the pre-action state s_t.
    """
    steps = tensors["chaser_hidden"].shape[0]

    def per_step(key: str) -> np.ndarray:
        return tensors[key][:steps, indices].numpy().reshape(-1)

    def onehot(actions: np.ndarray) -> np.ndarray:
        oh = np.zeros((actions.shape[0], 4), dtype=np.float64)
        valid = actions >= 0
        oh[np.arange(actions.shape[0])[valid], actions[valid]] = 1.0
        return oh

    chaser_pos = tensors["chaser_position"][:steps, indices].numpy().reshape(-1, 2) / (grid_size - 1)
    explorer_pos = tensors["explorer_position"][:steps, indices].numpy().reshape(-1, 2) / (grid_size - 1)
    chaser_act = onehot(tensors["chaser_action"][:, indices].numpy().reshape(-1))
    explorer_act = onehot(tensors["explorer_action"][:, indices].numpy().reshape(-1))

    self_cols = [
        per_step("collision").astype(np.float64)[:, None],
        chaser_pos,
        chaser_act,
        per_step("chaser_approach").astype(np.float64)[:, None],
        per_step("chaser_new_field").astype(np.float64)[:, None],
    ]
    partner_cols = [
        explorer_pos,
        explorer_act,
        per_step("explorer_escape").astype(np.float64)[:, None],
        per_step("explorer_new_field").astype(np.float64)[:, None],
    ]
    self_block = np.concatenate(self_cols, axis=1)
    partner_block = np.concatenate(partner_cols, axis=1)
    design = np.concatenate([self_block, partner_block], axis=1)
    is_partner = np.zeros(design.shape[1], dtype=bool)
    is_partner[self_block.shape[1]:] = True
    return design, is_partner


def _multivariate_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Total variance of y explained by a least-squares linear fit on x
    (x includes no intercept column; one is added)."""
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


@dataclass(frozen=True)
class PartnerRepresentation:
    r2_full: float
    r2_without_partner: float
    partner_unique: float
    n_samples: int
    n_valid_episodes: int


def partner_representation(
    tensors: dict[str, torch.Tensor],
    action_weight: np.ndarray,
    *,
    grid_size: int = 10,
    seed: int = 0,
) -> PartnerRepresentation:
    """Non-redundant variance of the chaser's neural action subspace explained
    by the partner's behaviour (paper C4).

    Fits z ~ [self + partner] and z ~ [self + permuted-partner]; the drop in
    total R^2 is the unique partner contribution.
    """
    indices = valid_episode_indices(tensors)
    if len(indices) < 2:
        return PartnerRepresentation(0.0, 0.0, 0.0, 0, len(indices))
    hidden = tensors["chaser_hidden"][:, indices].double().numpy().reshape(-1, tensors["chaser_hidden"].shape[-1])
    z = neural_action_subspace(hidden, action_weight)
    z = z - z.mean(axis=0)
    design, is_partner = _behavior_predictors(tensors, indices, grid_size)

    r2_full = _multivariate_r2(design, z)
    rng = np.random.RandomState(seed)
    permuted = design.copy()
    order = rng.permutation(permuted.shape[0])
    permuted[:, is_partner] = permuted[order][:, is_partner]
    r2_reduced = _multivariate_r2(permuted, z)
    return PartnerRepresentation(
        r2_full=r2_full,
        r2_without_partner=r2_reduced,
        partner_unique=max(r2_full - r2_reduced, 0.0),
        n_samples=z.shape[0],
        n_valid_episodes=len(indices),
    )


def chaser_action_weight(checkpoint: Path) -> np.ndarray:
    """The chaser's action-layer weight (A, H) from a checkpoint."""
    _, _, chaser_state, _ = load_checkpoint(checkpoint)
    return chaser_state["action_layer.weight"].double().numpy()


# ---------------------------------------------------------------------------
# C5: top-k shared-dimension null-space perturbation basis
# ---------------------------------------------------------------------------


def shared_dimension_basis(
    tensors: dict[str, torch.Tensor],
    *,
    top_k: int = 10,
) -> np.ndarray:
    """Chaser-side top-k PLSC directions (H, k), orthonormalized, for the
    null-space perturbation. Built from the cross-covariance of the two
    agents' z-scored hidden states over non-degenerate episodes."""
    chaser_episodes, explorer_episodes = episode_hidden_pairs(tensors)
    X = _zscore(np.concatenate(chaser_episodes))
    Y = _zscore(np.concatenate(explorer_episodes))
    cross = X.T @ Y / (X.shape[0] - 1)
    U, _, _ = np.linalg.svd(cross)
    basis = U[:, :top_k]
    q, _ = np.linalg.qr(basis)
    return q


def random_variance_basis(
    tensors: dict[str, torch.Tensor],
    *,
    top_k: int = 25,
    seed: int = 0,
) -> np.ndarray:
    """Control basis (H, k): top-k PCs of temporally permuted chaser activity,
    matching a comparable amount of removed variance without the shared
    structure (paper C5 control)."""
    chaser_episodes, _ = episode_hidden_pairs(tensors)
    X = _zscore(np.concatenate(chaser_episodes))
    rng = np.random.RandomState(seed)
    permuted = X[rng.permutation(X.shape[0])]
    cov = permuted.T @ permuted / (permuted.shape[0] - 1)
    values, vectors = np.linalg.eigh(cov)
    basis = vectors[:, ::-1][:, :top_k]
    q, _ = np.linalg.qr(basis)
    return q

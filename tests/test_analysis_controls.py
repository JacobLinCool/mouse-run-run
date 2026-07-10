"""Positive/negative controls for the C3/C4/C5 analysis stack in analysis.py.

Covers the pieces not exercised by test_plsc_controls.py: the
subtract_time_mean option of plsc_shared_dimensions, event decoding
(decode_balanced_accuracy / decoding_targets), linear CKA, the C4 partner
representation (including the permuted-partner unique-R^2 control), and the
C5 perturbation bases. Every dataset is synthetic with a known ground truth
and every random draw is seeded, so the assertions are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_run_run.analysis import (
    decode_balanced_accuracy,
    decoding_targets,
    linear_cka,
    partner_representation,
    plsc_shared_dimensions,
    random_variance_basis,
    shared_dimension_basis,
)

# ---------------------------------------------------------------------------
# subtract_time_mean (PLSC on the episode-specific residual)
# ---------------------------------------------------------------------------


def test_subtract_time_mean_removes_cross_episode_mean_exactly() -> None:
    # Episodes are (time-locked mean M) + (residual R_i) with the residuals
    # constructed to have exactly zero cross-episode mean. Subtracting the
    # time mean must therefore recover the residuals exactly, so the result
    # must match PLSC run directly on the residuals.
    rng = np.random.default_rng(11)
    episodes, steps, units = 8, 50, 5
    chaser_mean = rng.standard_normal((steps, units))
    explorer_mean = rng.standard_normal((steps, units))
    chaser_resid = rng.standard_normal((episodes, steps, units))
    chaser_resid -= chaser_resid.mean(axis=0)
    mixing = rng.standard_normal((units, units))
    explorer_resid = chaser_resid @ mixing + 0.1 * rng.standard_normal(
        (episodes, steps, units)
    )
    explorer_resid -= explorer_resid.mean(axis=0)

    detrended = plsc_shared_dimensions(
        [chaser_mean + chaser_resid[i] for i in range(episodes)],
        [explorer_mean + explorer_resid[i] for i in range(episodes)],
        permutations=50,
        seed=0,
        subtract_time_mean=True,
    )
    residual_only = plsc_shared_dimensions(
        [chaser_resid[i] for i in range(episodes)],
        [explorer_resid[i] for i in range(episodes)],
        permutations=50,
        seed=0,
    )

    assert np.allclose(
        detrended.singular_values, residual_only.singular_values, atol=1e-8
    )
    assert detrended.top_dim_correlation == pytest.approx(
        residual_only.top_dim_correlation, abs=1e-8
    )


def test_subtract_time_mean_removes_time_locked_pseudo_sharing() -> None:
    # Both agents replay the same deterministic time-locked pattern in every
    # episode; the episode-specific residuals are independent. circular_shift
    # (the weak null) flags the time-locked structure as shared, while
    # subtract_time_mean removes it and nothing survives either null.
    rng = np.random.default_rng(0)
    episodes, steps, units = 10, 60, 5
    chaser_mean = rng.standard_normal((steps, units))
    explorer_mean = chaser_mean @ rng.standard_normal((units, units))
    chaser_episodes = [
        chaser_mean + 0.3 * rng.standard_normal((steps, units)) for _ in range(episodes)
    ]
    explorer_episodes = [
        explorer_mean + 0.3 * rng.standard_normal((steps, units))
        for _ in range(episodes)
    ]

    baseline = plsc_shared_dimensions(
        chaser_episodes, explorer_episodes, permutations=100, seed=0
    )
    assert baseline.nulls["circular_shift"].n_significant >= 4
    assert baseline.nulls["episode_shuffle"].n_significant == 0

    detrended = plsc_shared_dimensions(
        chaser_episodes,
        explorer_episodes,
        permutations=100,
        seed=0,
        subtract_time_mean=True,
    )
    assert detrended.nulls["circular_shift"].n_significant == 0
    assert detrended.n_significant == 0
    # Effect size, not just the thresholded count: the leading singular value
    # collapses once the time-locked mean is removed.
    assert detrended.singular_values[0] < 0.25 * baseline.singular_values[0]


def test_subtract_time_mean_requires_equal_length_episodes() -> None:
    rng = np.random.default_rng(0)
    chaser = [rng.standard_normal((10, 3)), rng.standard_normal((12, 3))]
    explorer = [rng.standard_normal((10, 3)), rng.standard_normal((12, 3))]
    with pytest.raises(ValueError, match="subtract_time_mean"):
        plsc_shared_dimensions(
            chaser,
            explorer,
            null_models=("circular_shift",),
            headline_null="circular_shift",
            subtract_time_mean=True,
        )


# ---------------------------------------------------------------------------
# Event decoding
# ---------------------------------------------------------------------------


def test_decode_separable_labels_near_one_and_shuffled_near_chance() -> None:
    rng = np.random.RandomState(0)
    n, dims = 800, 8
    labels = np.zeros(n, dtype=np.int64)
    labels[: n // 2] = 1
    rng.shuffle(labels)
    hidden = rng.randn(n, dims)
    hidden[:, 0] += 4.0 * labels

    result = decode_balanced_accuracy(hidden, labels, seed=0)
    assert result.skipped_reason is None
    assert result.balanced_accuracy is not None
    assert result.balanced_accuracy > 0.97
    # The built-in label-shuffled control runs through the same pipeline and
    # must sit at chance.
    assert 0.4 < result.shuffled_accuracy < 0.6

    shuffled = decode_balanced_accuracy(hidden, rng.permutation(labels), seed=0)
    assert 0.4 < shuffled.balanced_accuracy < 0.6


def test_decode_balanced_accuracy_is_balanced_under_class_imbalance() -> None:
    # 10% positives: a majority-class classifier scores 0.9 raw accuracy but
    # only 0.5 balanced accuracy. Uninformative features must land at chance,
    # not at the majority-class rate; separable features must still reach ~1.
    rng = np.random.RandomState(1)
    n, dims, positives = 1000, 8, 100
    labels = np.zeros(n, dtype=np.int64)
    labels[:positives] = 1
    rng.shuffle(labels)

    noise = decode_balanced_accuracy(rng.randn(n, dims), labels, seed=0)
    assert noise.n_positive == positives
    assert 0.35 < noise.balanced_accuracy < 0.65

    hidden = rng.randn(n, dims)
    hidden[:, 0] += 4.0 * labels
    separable = decode_balanced_accuracy(hidden, labels, seed=0)
    assert separable.balanced_accuracy > 0.95


def test_decode_skips_when_either_class_is_too_small() -> None:
    rng = np.random.RandomState(2)
    hidden = rng.randn(500, 4)

    labels = np.zeros(500, dtype=np.int64)
    labels[:10] = 1
    rare_positive = decode_balanced_accuracy(hidden, labels, min_positive=50)
    assert rare_positive.balanced_accuracy is None
    assert rare_positive.shuffled_accuracy is None
    assert rare_positive.skipped_reason is not None
    assert rare_positive.n_positive == 10

    rare_negative = decode_balanced_accuracy(hidden, 1 - labels, min_positive=50)
    assert rare_negative.balanced_accuracy is None
    assert rare_negative.skipped_reason is not None


def test_decoding_targets_pools_valid_episodes_and_masks_visibility() -> None:
    steps, episodes = 4, 3  # episode 1 is degenerate -> only episodes 0 and 2
    valid = [0, 2]
    chaser_hidden = torch.zeros(steps, episodes, 2)
    explorer_hidden = torch.zeros(steps, episodes, 2)
    for t in range(steps):
        for e in range(episodes):
            chaser_hidden[t, e] = torch.tensor([float(t), 100.0 + e])
            explorer_hidden[t, e] = torch.tensor([1000.0 + t, 2000.0 + e])
    collision = [[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]
    escape = [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]]
    approach = [[0, 1, 1], [1, 0, 0], [0, 0, 0], [1, 0, 1]]
    # steps+1 rows: the final row is visibility in the post-action state and
    # must be ignored (all-ones there would otherwise change the masks).
    chaser_visible = [[1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]]
    explorer_visible = [[0, 1, 1], [1, 0, 1], [0, 0, 0], [1, 0, 1], [1, 1, 1]]
    tensors = {
        "chaser_hidden": chaser_hidden,
        "explorer_hidden": explorer_hidden,
        "collision": torch.tensor(collision, dtype=torch.bool),
        "explorer_escape": torch.tensor(escape, dtype=torch.bool),
        "chaser_approach": torch.tensor(approach, dtype=torch.bool),
        "chaser_partner_visible": torch.tensor(chaser_visible, dtype=torch.bool),
        "explorer_partner_visible": torch.tensor(explorer_visible, dtype=torch.bool),
        "episode_degenerate": torch.tensor([False, True, False]),
    }

    targets = decoding_targets(tensors)
    assert set(targets) == {
        "chaser_collision",
        "chaser_partner_escape",
        "explorer_collision",
        "explorer_partner_approach",
    }

    # Pooled targets: degenerate episode dropped, time-major ordering.
    hidden, labels = targets["chaser_collision"]
    expected_hidden = np.array(
        [[t, 100 + e] for t in range(steps) for e in valid], dtype=np.float64
    )
    expected_labels = np.array(
        [collision[t][e] for t in range(steps) for e in valid], dtype=bool
    )
    assert hidden.shape == (steps * len(valid), 2)
    assert np.array_equal(hidden, expected_hidden)
    assert np.array_equal(labels, expected_labels)

    # Masked targets: only rows where the partner is visible in s_t survive,
    # derived here by an independent loop over (t, episode).
    hidden, labels = targets["chaser_partner_escape"]
    expected_rows = [
        ([t, 100 + e], escape[t][e])
        for t in range(steps)
        for e in valid
        if chaser_visible[t][e]
    ]
    assert np.array_equal(hidden, np.array([row for row, _ in expected_rows]))
    assert np.array_equal(labels, np.array([label for _, label in expected_rows], dtype=bool))

    hidden, labels = targets["explorer_partner_approach"]
    expected_rows = [
        ([1000 + t, 2000 + e], approach[t][e])
        for t in range(steps)
        for e in valid
        if explorer_visible[t][e]
    ]
    assert np.array_equal(hidden, np.array([row for row, _ in expected_rows]))
    assert np.array_equal(labels, np.array([label for _, label in expected_rows], dtype=bool))


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------


def test_linear_cka_self_is_one() -> None:
    rng = np.random.RandomState(0)
    x = rng.randn(200, 6)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-9)


def test_linear_cka_invariant_to_orthogonal_rotation() -> None:
    rng = np.random.RandomState(1)
    x = rng.randn(300, 6)
    y = x @ rng.randn(6, 6) + 0.5 * rng.randn(300, 6)
    q, _ = np.linalg.qr(rng.randn(6, 6))
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-9)
    assert linear_cka(x, y @ q) == pytest.approx(linear_cka(x, y), abs=1e-9)


def test_linear_cka_near_zero_for_independent_matrices() -> None:
    rng = np.random.RandomState(2)
    x = rng.randn(2000, 8)
    y = rng.randn(2000, 8)
    assert linear_cka(x, y) < 0.05


def test_linear_cka_degenerate_input_returns_zero() -> None:
    zeros = np.zeros((50, 4))
    rng = np.random.RandomState(3)
    assert linear_cka(zeros, rng.randn(50, 4)) == 0.0


# ---------------------------------------------------------------------------
# C4: partner representation
# ---------------------------------------------------------------------------

_GRID = 10


def _behavior_tensors(
    rng: np.random.Generator,
    hidden: np.ndarray,
    chaser_position: np.ndarray,
    explorer_position: np.ndarray,
) -> dict[str, torch.Tensor]:
    steps, episodes = hidden.shape[0], hidden.shape[1]

    def flags() -> torch.Tensor:
        return torch.from_numpy(rng.random((steps, episodes)) < 0.2)

    def actions() -> torch.Tensor:
        return torch.from_numpy(rng.integers(0, 4, size=(steps, episodes)))

    return {
        "chaser_hidden": torch.from_numpy(hidden),
        "episode_degenerate": torch.zeros(episodes, dtype=torch.bool),
        "collision": flags(),
        "chaser_position": torch.from_numpy(chaser_position),
        "explorer_position": torch.from_numpy(explorer_position),
        "chaser_action": actions(),
        "explorer_action": actions(),
        "chaser_approach": flags(),
        "chaser_new_field": flags(),
        "explorer_escape": flags(),
        "explorer_new_field": flags(),
    }


def _partner_setup(
    *, encode: str, seed: int
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Hidden states whose action subspace linearly encodes the position of
    ``encode`` ("partner" = explorer, "self" = chaser); everything else is
    independent noise."""
    rng = np.random.default_rng(seed)
    steps, episodes, units = 60, 8, 6
    chaser_position = rng.integers(0, _GRID, size=(steps + 1, episodes, 2))
    explorer_position = rng.integers(0, _GRID, size=(steps + 1, episodes, 2))
    encoded = explorer_position if encode == "partner" else chaser_position
    hidden = np.zeros((steps, episodes, units))
    hidden[..., :2] = encoded[:steps] / (_GRID - 1) + 0.01 * rng.standard_normal(
        (steps, episodes, 2)
    )
    hidden[..., 2:4] = 0.05 * rng.standard_normal((steps, episodes, 2))
    hidden[..., 4:] = rng.standard_normal((steps, episodes, 2))
    # Action layer reads hidden units 0-3, so the neural action subspace keeps
    # the position code and drops the pure-noise units 4-5.
    action_weight = np.zeros((4, units))
    action_weight[:, :4] = np.eye(4)
    tensors = _behavior_tensors(rng, hidden, chaser_position, explorer_position)
    return tensors, action_weight


def test_partner_representation_high_unique_r2_for_partner_coding() -> None:
    tensors, action_weight = _partner_setup(encode="partner", seed=0)
    result = partner_representation(tensors, action_weight, grid_size=_GRID, seed=0)
    assert result.n_valid_episodes == 8
    assert result.n_samples == 60 * 8
    assert result.r2_full > 0.9
    # The permuted-partner control destroys the partner columns' alignment,
    # so nearly all of the explained variance is uniquely the partner's.
    assert result.r2_without_partner < 0.2
    assert result.partner_unique > 0.8


def test_partner_representation_self_only_coding_has_no_unique_partner() -> None:
    tensors, action_weight = _partner_setup(encode="self", seed=1)
    result = partner_representation(tensors, action_weight, grid_size=_GRID, seed=0)
    # Self columns alone explain the subspace; permuting the partner columns
    # changes (essentially) nothing.
    assert result.r2_full > 0.9
    assert result.r2_without_partner > 0.85
    assert result.partner_unique < 0.05


def test_partner_representation_needs_two_valid_episodes() -> None:
    tensors, action_weight = _partner_setup(encode="partner", seed=2)
    tensors["episode_degenerate"] = torch.tensor(
        [False] + [True] * 7, dtype=torch.bool
    )
    result = partner_representation(tensors, action_weight, grid_size=_GRID, seed=0)
    assert result == type(result)(0.0, 0.0, 0.0, 0, 1)


# ---------------------------------------------------------------------------
# C5: perturbation bases
# ---------------------------------------------------------------------------


def _basis_tensors(seed: int) -> tuple[dict[str, torch.Tensor], int, int]:
    """Hidden pairs whose shared structure lives exactly in chaser units 0-1.

    The degenerate episode is filled with NaN: if either basis builder failed
    to exclude it, the linear algebra downstream would blow up loudly.
    """
    rng = np.random.default_rng(seed)
    steps, episodes, units = 80, 8, 6
    chaser = rng.standard_normal((steps, episodes, units))
    explorer = rng.standard_normal((steps, episodes, units))
    for e in range(episodes - 1):
        shared = rng.standard_normal((steps, 2))
        chaser[:, e, :2] = shared + 0.05 * rng.standard_normal((steps, 2))
        explorer[:, e, :2] = shared + 0.05 * rng.standard_normal((steps, 2))
    chaser[:, -1] = np.nan
    explorer[:, -1] = np.nan
    degenerate = torch.zeros(episodes, dtype=torch.bool)
    degenerate[-1] = True
    tensors = {
        "chaser_hidden": torch.from_numpy(chaser),
        "explorer_hidden": torch.from_numpy(explorer),
        "episode_degenerate": degenerate,
    }
    return tensors, steps * (episodes - 1), units


def _span_projection(basis: np.ndarray, unit: int) -> float:
    return float(np.linalg.norm(basis.T[:, unit]))


def test_shared_dimension_basis_orthonormal_and_recovers_shared_units() -> None:
    tensors, _, units = _basis_tensors(seed=3)
    basis = shared_dimension_basis(tensors, top_k=2)
    assert basis.shape == (units, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-10)
    # The top-2 span must be the shared subspace (units 0-1), not the noise
    # units and not the NaN-poisoned degenerate episode.
    assert _span_projection(basis, 0) > 0.95
    assert _span_projection(basis, 1) > 0.95
    assert _span_projection(basis, 5) < 0.3


def test_random_variance_basis_orthonormal_seeded_and_variance_matched() -> None:
    tensors, n_valid_samples, units = _basis_tensors(seed=4)
    basis = random_variance_basis(tensors, top_k=3, seed=7)
    assert basis.shape == (units, 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-10)
    assert np.array_equal(basis, random_variance_basis(tensors, top_k=3, seed=7))

    # Variance-matched control: projecting the (z-scored, pooled) chaser
    # activity onto the control basis removes at least as much variance as
    # the shared basis of the same rank.
    chaser = tensors["chaser_hidden"].double().numpy()[:, :-1].reshape(-1, units)
    assert chaser.shape[0] == n_valid_samples
    z = (chaser - chaser.mean(axis=0)) / chaser.std(axis=0)
    cov = z.T @ z / (z.shape[0] - 1)

    def captured(b: np.ndarray) -> float:
        return float(np.trace(b.T @ cov @ b))

    shared = shared_dimension_basis(tensors, top_k=3)
    assert captured(basis) >= captured(shared) - 1e-9

from __future__ import annotations

import pytest
import torch

from mouse_run_run.analyses import linear_svm
from mouse_run_run.analyses.decoder import _batched_scores, _sklearn_scores
from mouse_run_run.analyses.paper_plsc import PaperPLSCConfig, _circular_shift_null
from mouse_run_run.core.progress import ProgressSink, ProgressTracker


class Silent(ProgressSink):
    def update(self, update: object) -> None:
        return


def _tracker(total: int) -> ProgressTracker:
    return ProgressTracker("test", total, Silent())


def _paired_series(
    time: int = 40,
    episodes: int = 5,
    width: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(time, episodes, width, generator=generator, dtype=torch.float64)
    right = 0.5 * left + torch.randn(
        time, episodes, width, generator=generator, dtype=torch.float64
    )
    return left, right


def test_plsc_null_assembly_paths_agree() -> None:
    left, right = _paired_series()
    sample_mask = torch.ones(left.shape[0], left.shape[1], dtype=torch.bool)
    episode_mask = torch.ones(left.shape[1], dtype=torch.bool)
    arguments = dict(agent_a="a", agent_b="b", permutations=32, min_shift=4, seed=5)
    covariances = {}
    correlations = {}
    for algorithm in ("auto", "fft", "direct"):
        config = PaperPLSCConfig(**arguments, null_algorithm=algorithm)
        covariances[algorithm], correlations[algorithm] = _circular_shift_null(
            left, right, sample_mask, episode_mask, config, _tracker(32)
        )
    for algorithm in ("fft", "direct"):
        assert torch.allclose(covariances["auto"], covariances[algorithm], atol=1e-12)
        assert torch.allclose(correlations["auto"], correlations[algorithm], atol=1e-10)


def test_plsc_null_falls_back_when_samples_are_dropped_inside_an_episode() -> None:
    left, right = _paired_series()
    sample_mask = torch.ones(left.shape[0], left.shape[1], dtype=torch.bool)
    episode_mask = torch.ones(left.shape[1], dtype=torch.bool)
    partial = sample_mask.clone()
    partial[3, 1] = False
    config = PaperPLSCConfig(agent_a="a", agent_b="b", permutations=8, min_shift=4, seed=1)
    whole_covariance, _ = _circular_shift_null(
        left, right, sample_mask, episode_mask, config, _tracker(8)
    )
    partial_covariance, partial_pcc = _circular_shift_null(
        left, right, partial, episode_mask, config, _tracker(8)
    )
    # The fallback runs a different analysis, not a different implementation of
    # the same one, so it only has to stay finite and correctly shaped.
    assert partial_covariance.shape == whole_covariance.shape
    assert torch.isfinite(partial_covariance).all()
    assert torch.isfinite(partial_pcc).all()
    assert not torch.allclose(partial_covariance, whole_covariance)


def test_plsc_null_is_reproducible_from_the_seed() -> None:
    left, right = _paired_series()
    sample_mask = torch.ones(left.shape[0], left.shape[1], dtype=torch.bool)
    episode_mask = torch.ones(left.shape[1], dtype=torch.bool)
    config = PaperPLSCConfig(agent_a="a", agent_b="b", permutations=16, min_shift=4, seed=9)
    first, _ = _circular_shift_null(
        left, right, sample_mask, episode_mask, config, _tracker(16)
    )
    second, _ = _circular_shift_null(
        left, right, sample_mask, episode_mask, config, _tracker(16)
    )
    assert torch.equal(first, second)


def test_batched_decoder_matches_the_per_fit_solver() -> None:
    generator = torch.Generator().manual_seed(4)
    samples, width = 240, 6
    labels = (torch.arange(samples) % 2).long()
    values = torch.randn(samples, width, generator=generator)
    values[labels == 1] += 0.9
    label_sets = torch.stack(
        [labels]
        + [labels[torch.randperm(samples, generator=generator)] for _ in range(3)]
    )
    folds = [
        (
            torch.arange(samples)[torch.arange(samples) % 4 != index],
            torch.arange(samples)[torch.arange(samples) % 4 == index],
        )
        for index in range(4)
    ]
    batched = _batched_scores(values, label_sets, folds)
    reference = _sklearn_scores(values, label_sets, folds)
    assert batched[0] > 0.7
    assert torch.allclose(batched, reference, atol=0.02)


def test_balanced_weights_equalize_class_totals() -> None:
    labels = torch.tensor([[0, 0, 0, 1]])
    weights = linear_svm.balanced_weights(labels, 1.0)
    assert weights[0, :3].sum() == pytest.approx(float(weights[0, 3]))


def test_balanced_accuracy_averages_class_recall() -> None:
    labels = torch.tensor([[0, 0, 1, 1]])
    predictions = torch.tensor([[0, 0, 1, 0]])
    assert linear_svm.balanced_accuracy(labels, predictions).tolist() == pytest.approx([0.75])


def test_squared_hinge_separates_a_linearly_separable_batch() -> None:
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float64)
    labels = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    weights = linear_svm.balanced_weights(labels, 1.0)
    coefficients = linear_svm.fit_squared_hinge(
        linear_svm.with_intercept(features), labels, weights
    )
    predictions = linear_svm.predict(linear_svm.with_intercept(features), coefficients)
    assert torch.equal(predictions, labels)

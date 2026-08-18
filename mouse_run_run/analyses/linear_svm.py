"""Batched linear SVM sharing one design matrix across many label sets.

Shuffled-control decoding refits the same features against hundreds of relabelled
targets. Fitting them one at a time repeats identical work, so this module solves
the whole family at once: the objective is the one scikit-learn's ``LinearSVC``
minimises by default — squared hinge loss with an L2 penalty and a regularised
intercept — expressed so that every label set advances in the same optimiser step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SquaredHingeConfig:
    regularization: float = 1.0
    """``C`` in the scikit-learn parameterisation."""

    max_iterations: int = 300
    tolerance: float = 1e-10

    def validate(self) -> None:
        if self.regularization <= 0:
            raise ValueError("SVM regularization must be positive")
        if self.max_iterations < 1:
            raise ValueError("SVM iterations must be positive")


def balanced_weights(labels: torch.Tensor, regularization: float) -> torch.Tensor:
    """Per-sample weights matching ``class_weight="balanced"``.

    Each class receives the same total weight, so the penalty on a sample is
    ``C * n / (2 * count(class))``.
    """

    counts = torch.stack([(labels == value).sum(dim=1) for value in (0, 1)], dim=1)
    scale = labels.shape[1] / (2.0 * counts.clamp_min(1).to(labels.dtype))
    return regularization * scale.gather(1, labels.long())


def fit_squared_hinge(
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    config: SquaredHingeConfig = SquaredHingeConfig(),
) -> torch.Tensor:
    """Minimise ``0.5||w||^2 + sum_i C_i max(0, 1 - y_i x_i w)^2`` for every label set.

    ``features`` is shared and carries its own intercept column; ``labels`` holds
    one row per problem in ``{0, 1}``. The returned coefficients have one row per
    problem.
    """

    config.validate()
    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError("features must be (samples, width) and labels (problems, samples)")
    if features.shape[0] != labels.shape[1]:
        raise ValueError("features and labels disagree on the sample count")
    signs = labels.to(features.dtype) * 2.0 - 1.0
    coefficients = torch.zeros(
        labels.shape[0], features.shape[1], dtype=features.dtype, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [coefficients],
        max_iter=config.max_iterations,
        tolerance_grad=config.tolerance,
        tolerance_change=config.tolerance,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        margins = 1.0 - signs * (coefficients @ features.T)
        loss = (
            0.5 * coefficients.square().sum()
            + (weights * margins.clamp_min(0.0).square()).sum()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return coefficients.detach()


def predict(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Decision-rule labels in ``{0, 1}``, one row per problem."""

    return ((coefficients @ features.T) > 0).long()


def balanced_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Mean per-class recall, one value per problem."""

    recalls = []
    for value in (0, 1):
        actual = labels == value
        hits = (actual & (predictions == value)).sum(dim=1)
        recalls.append(hits / actual.sum(dim=1).clamp_min(1))
    present = torch.stack([(labels == value).any(dim=1) for value in (0, 1)], dim=0)
    stacked = torch.stack(recalls, dim=0)
    return (stacked * present).sum(dim=0) / present.sum(dim=0).clamp_min(1)


def with_intercept(features: torch.Tensor) -> torch.Tensor:
    """Append the constant column liblinear uses for a regularised intercept."""

    ones = torch.ones(features.shape[0], 1, dtype=features.dtype, device=features.device)
    return torch.cat([features, ones], dim=1)


def standardize(train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Centre and scale by the training fold's statistics, as the pipeline does."""

    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
    return tuple((values - mean) / scale for values in (train, *others))

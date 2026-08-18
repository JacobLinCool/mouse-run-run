from __future__ import annotations

import pytest
import torch

from mouse_run_run.analyses.partner_variance import (
    PartnerVarianceConfig,
    neural_action_space,
    non_redundant_variance,
    variance_explained,
)
from mouse_run_run.analyses.shared_structure import loading_comparison


def _config() -> PartnerVarianceConfig:
    return PartnerVarianceConfig(group_permutations=2, chance_shuffles=2, min_shift=20)


def test_variance_explained_recovers_a_linear_map() -> None:
    generator = torch.Generator().manual_seed(0)
    predictors = torch.randn(400, 3, generator=generator, dtype=torch.float64)
    response = predictors @ torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64)
    assert variance_explained(predictors, response) == pytest.approx(1.0, abs=1e-9)
    noise = torch.randn(400, 1, generator=generator, dtype=torch.float64)
    assert 0.0 < variance_explained(predictors, noise) < 0.1


def test_only_the_group_that_drives_the_response_is_credited() -> None:
    generator = torch.Generator().manual_seed(3)
    samples = 600
    partner = torch.randn(samples, 2, generator=generator, dtype=torch.float64)
    unrelated = torch.randn(samples, 2, generator=generator, dtype=torch.float64)
    response = partner @ torch.tensor([[1.0, 0.5], [-1.0, 2.0]], dtype=torch.float64)
    contributions = non_redundant_variance(
        {"self": unrelated, "partner": partner}, response, config=_config()
    )
    assert contributions["partner"] > 0.5
    assert contributions["self"] < 0.05
    assert contributions["full_model"] > 0.5


def test_a_response_nobody_drives_credits_nobody() -> None:
    generator = torch.Generator().manual_seed(5)
    samples = 600
    groups = {
        "self": torch.randn(samples, 2, generator=generator, dtype=torch.float64),
        "partner": torch.randn(samples, 2, generator=generator, dtype=torch.float64),
    }
    response = torch.randn(samples, 4, generator=generator, dtype=torch.float64)
    contributions = non_redundant_variance(groups, response, config=_config())
    assert abs(contributions["partner"]) < 0.05
    assert abs(contributions["self"]) < 0.05


def test_neural_action_space_keeps_only_what_the_readout_reads() -> None:
    hidden = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    readout = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    projected = neural_action_space(hidden, readout)
    assert projected.tolist() == [[1.0, 0.0, 0.0]]


def test_loading_comparison_flags_units_above_a_random_direction() -> None:
    width = 64
    first = torch.zeros(width, dtype=torch.float64)
    first[0] = 0.9
    second = torch.zeros(width, dtype=torch.float64)
    second[1] = 0.9
    comparison = loading_comparison(first, second)
    assert bool(comparison["first_significant"][0])
    assert bool(comparison["second_significant"][1])
    assert not bool(comparison["first_significant"][1])
    assert comparison["difference"][0] == pytest.approx(0.9)

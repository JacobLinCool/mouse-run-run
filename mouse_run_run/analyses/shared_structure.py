"""Loading structure and intra-brain disruption controls for shared dimensions.

Two questions the shared subspace raises once it exists: which units carry it,
and whether the cross-agent covariance survives destroying structure inside each
agent. Neither analysis is in the released code, so the operationalisations here
are stated rather than inherited.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mouse_run_run.analyses.paper_plsc import PaperPLSCResult, _standardize
from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.types import AgentId


@dataclass(frozen=True)
class DisruptionConfig:
    shuffles: int = 100
    min_shift: int = 60
    seed: int = 0

    def validate(self) -> None:
        if self.shuffles < 1:
            raise ValueError("disruption shuffles must be positive")


def unique_dimension_basis(
    result: PaperPLSCResult,
    rollout: RolloutArtifact,
    *,
    agent_id: AgentId,
    shared_rank: int,
    rank: int = 1,
) -> torch.Tensor:
    """Leading principal directions of what the shared subspace leaves behind."""

    activity = rollout.trajectory.agents[agent_id].activations[result.config.site]
    standardized = (activity.double() - result.means[agent_id]) / result.scales[agent_id]
    samples = standardized[result.sample_mask]
    shared = result.bases[agent_id][:, :shared_rank]
    residual = samples - (samples @ shared) @ shared.T
    covariance = residual.T @ residual / (residual.shape[0] - 1)
    _, vectors = torch.linalg.eigh(covariance)
    return vectors.flip(dims=(1,))[:, :rank]


def loading_comparison(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    significance: float = 0.975,
) -> dict[str, torch.Tensor]:
    """Per-unit loading magnitudes on two dimensions, and which units carry them.

    A unit counts as carrying a dimension when its absolute loading exceeds what
    a direction drawn uniformly at random would give at the stated level; for a
    unit vector in ``d`` dimensions that threshold is available in closed form.
    """

    width = first.numel()
    threshold = _random_direction_threshold(width, significance)
    return {
        "first": first.abs(),
        "second": second.abs(),
        "difference": first.abs() - second.abs(),
        "first_significant": first.abs() > threshold,
        "second_significant": second.abs() > threshold,
        "threshold": torch.tensor(threshold, dtype=first.dtype),
    }


def disrupted_cross_covariance(
    result: PaperPLSCResult,
    rollout: RolloutArtifact,
    *,
    config: DisruptionConfig = DisruptionConfig(),
) -> dict[str, float]:
    """Top shared covariance under intact and intra-brain-disrupted activity.

    ``intraset_correlation`` permutes each unit's samples independently, which
    keeps every unit's marginal and destroys unit-to-unit correlation inside an
    agent. ``intraset_coupling`` circularly shifts each unit independently, which
    keeps each unit's autocorrelation and destroys its alignment with the other
    units of the same agent. Both are reported against their own shifted null,
    matching the observed-minus-shuffle quantity.
    """

    config.validate()
    left, right = _standardized_pair(result, rollout)
    generator = torch.Generator().manual_seed(config.seed)
    outcomes = {"original": _top_covariance(left, right)}
    for name, disrupt in (
        ("intraset_correlation", _permute_units),
        ("intraset_coupling", _shift_units),
    ):
        outcomes[name] = _top_covariance(
            disrupt(left, generator, config), disrupt(right, generator, config)
        )
    null = _shifted_null(left, right, config, generator)
    return {name: value - null for name, value in outcomes.items()}


def _standardized_pair(
    result: PaperPLSCResult, rollout: RolloutArtifact
) -> tuple[torch.Tensor, torch.Tensor]:
    pair = []
    for agent_id in (result.config.agent_a, result.config.agent_b):
        activity = rollout.trajectory.agents[agent_id].activations[result.config.site]
        pair.append(_standardize(activity.double()[result.sample_mask])[0])
    return pair[0], pair[1]


def _top_covariance(left: torch.Tensor, right: torch.Tensor) -> float:
    cross = left.T @ right / (left.shape[0] - 1)
    return float(torch.linalg.svdvals(cross)[0])


def _permute_units(
    values: torch.Tensor, generator: torch.Generator, config: DisruptionConfig
) -> torch.Tensor:
    samples = values.shape[0]
    order = torch.stack(
        [torch.randperm(samples, generator=generator) for _ in range(values.shape[1])],
        dim=1,
    )
    return values.gather(0, order)


def _shift_units(
    values: torch.Tensor, generator: torch.Generator, config: DisruptionConfig
) -> torch.Tensor:
    samples, width = values.shape
    shifts = torch.randint(
        config.min_shift, samples - config.min_shift + 1, (width,), generator=generator
    )
    index = (torch.arange(samples).unsqueeze(1) - shifts.unsqueeze(0)) % samples
    return values.gather(0, index)


def _shifted_null(
    left: torch.Tensor,
    right: torch.Tensor,
    config: DisruptionConfig,
    generator: torch.Generator,
) -> float:
    samples = left.shape[0]
    values = []
    for _ in range(config.shuffles):
        shift = int(
            torch.randint(
                config.min_shift, samples - config.min_shift + 1, (), generator=generator
            )
        )
        values.append(_top_covariance(left, torch.roll(right, shift, dims=0)))
    return sum(values) / len(values)


def _random_direction_threshold(width: int, level: float) -> float:
    """Absolute component of a uniform random unit vector at the given level."""

    samples = 20_000
    generator = torch.Generator().manual_seed(0)
    draws = torch.randn(samples, width, generator=generator, dtype=torch.float64)
    components = (draws[:, 0] / draws.norm(dim=1)).abs()
    return float(torch.quantile(components, level))

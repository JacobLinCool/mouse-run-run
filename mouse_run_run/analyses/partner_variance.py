"""Non-redundant neural variance explained by a partner's behaviour.

A port of the released `computeNonRedundantVar.m`: fit partial least squares
regression from grouped behavioural predictors to neural activity, then destroy
one group's temporal alignment and measure how much explained variance is lost.
Every model is compared against its own temporally shifted chance level, so the
reported contribution is what a group explains beyond what the other groups
already account for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.types import AgentId


CHASER_BEHAVIOURS = ("chaser_new_field", "chaser_approach")
EXPLORER_BEHAVIOURS = (
    "explorer_new_field",
    "explorer_escape",
    "explorer_escape_close",
    "explorer_escape_near",
    "explorer_escape_far",
)


@dataclass(frozen=True)
class PartnerVarianceConfig:
    group_permutations: int = 10
    """Repeats of the group-destroying shift, averaged over."""

    chance_shuffles: int = 10
    """Shifts of a whole design matrix used as that model's chance level."""

    min_shift: int = 60
    seed: int = 0
    def validate(self) -> None:
        if self.group_permutations < 1 or self.chance_shuffles < 1:
            raise ValueError("partner variance repeat counts must be positive")
        if self.min_shift < 1:
            raise ValueError("partner variance min_shift must be positive")


def variance_explained(predictors: torch.Tensor, response: torch.Tensor) -> float:
    """Fraction of response variance the predictors reproduce.

    The released code fits PLS with as many components as the design matrix has
    rank, which spans the same column space as the predictors and therefore
    gives the least-squares fit. Solving that directly is exact where iterative
    deflation degenerates on the collinear one-hot blocks these designs contain.
    """

    features = predictors.double()
    targets = response.double()
    centred = features - features.mean(dim=0, keepdim=True)
    responses = targets - targets.mean(dim=0, keepdim=True)
    coefficients = torch.linalg.lstsq(centred, responses).solution
    residual = (responses - centred @ coefficients).square().sum()
    total = responses.square().sum()
    return float(1.0 - residual / total.clamp_min(1e-12))


def non_redundant_variance(
    groups: Mapping[str, torch.Tensor],
    response: torch.Tensor,
    *,
    config: PartnerVarianceConfig = PartnerVarianceConfig(),
) -> dict[str, float]:
    """Chance-corrected variance each group explains beyond the others."""

    config.validate()
    if len(groups) < 2:
        raise ValueError("non-redundant variance needs at least two predictor groups")
    names = list(groups)
    generator = torch.Generator().manual_seed(config.seed)
    full = torch.cat([groups[name] for name in names], dim=1)
    full_score = variance_explained(full, response)
    full_chance = _chance_level(full, response, config, generator)
    contributions: dict[str, float] = {}
    for name in names:
        scores = []
        chances = []
        for _ in range(config.group_permutations):
            damaged = dict(groups)
            damaged[name] = _shift_block(groups[name], config.min_shift, generator)
            design = torch.cat([damaged[key] for key in names], dim=1)
            scores.append(variance_explained(design, response))
            chances.append(_chance_level(design, response, config, generator))
        without = sum(scores) / len(scores) - sum(chances) / len(chances)
        contributions[name] = (full_score - full_chance) - without
    contributions["full_model"] = full_score - full_chance
    return contributions


def behaviour_groups(
    rollout: RolloutArtifact,
    *,
    agent_id: AgentId,
    partner_id: AgentId,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Self and partner predictor blocks: positions, actions, and behaviours."""

    trajectory = rollout.trajectory
    selected = trajectory.active.cpu() if mask is None else mask.cpu()
    own, partner = str(agent_id), str(partner_id)
    blocks = {
        "self": _stack_columns(
            [
                _positions(trajectory, f"{own}_position", selected),
                _actions(trajectory, agent_id, selected),
                _events(trajectory, _behaviour_names(own) + ("collision",), selected),
            ]
        ),
        "partner": _stack_columns(
            [
                _positions(trajectory, f"{partner}_position", selected),
                _actions(trajectory, partner_id, selected),
                _events(trajectory, _behaviour_names(partner), selected),
            ]
        ),
    }
    return blocks


def neural_action_space(hidden: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
    """Activity projected into the subspace the action readout reads from.

    The paper notes this need not be an orthogonal projection because the
    readout is linear; it is ``W_action^T W_action h`` exactly as written.
    """

    weights = readout.double()
    return hidden.double() @ (weights.T @ weights)


def _behaviour_names(agent: str) -> tuple[str, ...]:
    return CHASER_BEHAVIOURS if agent == "chaser" else EXPLORER_BEHAVIOURS


def _positions(trajectory: object, key: str, mask: torch.Tensor) -> torch.Tensor:
    world = getattr(trajectory, "world")
    if key not in world:
        raise ValueError(f"rollout world lacks {key!r}")
    steps = mask.shape[0]
    return world[key][:steps].double()[mask]


def _actions(trajectory: object, agent_id: AgentId, mask: torch.Tensor) -> torch.Tensor:
    agents = getattr(trajectory, "agents")
    actions = agents[agent_id].actions.cpu()[mask]
    width = int(actions.max()) + 1 if actions.numel() else 1
    return torch.nn.functional.one_hot(actions.long(), num_classes=max(width, 4)).double()

def _events(trajectory: object, names: Sequence[str], mask: torch.Tensor) -> torch.Tensor:
    events = getattr(trajectory, "events")
    present = [name for name in names if name in events]
    if not present:
        return torch.zeros(int(mask.sum()), 0, dtype=torch.float64)
    return torch.stack([events[name][mask].double() for name in present], dim=1)


def _stack_columns(blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    usable = [block for block in blocks if block.numel() and block.shape[1] > 0]
    return torch.cat(usable, dim=1)


def _shift_block(values: torch.Tensor, min_shift: int, generator: torch.Generator) -> torch.Tensor:
    """One circular shift applied to the whole block, as `tempShift` does."""

    samples = values.shape[0]
    span = samples - min_shift
    if span < 1:
        raise ValueError("partner variance shift exceeds the sample count")
    shift = int(torch.randint(1, span + 1, (), generator=generator)) + min_shift // 2
    return torch.roll(values, shift % samples, dims=0)


def _chance_level(
    design: torch.Tensor,
    response: torch.Tensor,
    config: PartnerVarianceConfig,
    generator: torch.Generator,
) -> float:
    scores = [
        variance_explained(_shift_block(design, config.min_shift, generator), response)
        for _ in range(config.chance_shuffles)
    ]
    return sum(scores) / len(scores)

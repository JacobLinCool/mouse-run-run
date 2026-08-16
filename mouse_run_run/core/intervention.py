"""Composable, explicit interventions over policy activation sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from mouse_run_run.core.policy import ActivationSite, PolicyModule
from mouse_run_run.core.types import AgentId


InterventionTarget = Literal["readout", "recurrent"]


@dataclass(frozen=True)
class InterventionContext:
    agent_id: AgentId
    site: ActivationSite
    target: InterventionTarget
    step: int
    episode_indices: torch.Tensor


class Intervention(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def agent_ids(self) -> frozenset[AgentId]: ...

    @property
    def site(self) -> str: ...

    @property
    def target(self) -> InterventionTarget: ...

    def apply(self, value: torch.Tensor, context: InterventionContext) -> torch.Tensor: ...


class InterventionPipeline:
    """Ordered intervention composition; no matching is treated as identity."""

    def __init__(self, interventions: tuple[Intervention, ...] = ()) -> None:
        names = [intervention.name for intervention in interventions]
        if len(names) != len(set(names)):
            raise ValueError("intervention names must be unique")
        self.interventions = interventions

    def validate(self, policies: dict[AgentId, PolicyModule]) -> None:
        for intervention in self.interventions:
            if intervention.target not in ("readout", "recurrent"):
                raise ValueError(
                    f"intervention {intervention.name!r} has invalid target "
                    f"{intervention.target!r}"
                )
            if not intervention.agent_ids:
                raise ValueError(f"intervention {intervention.name!r} selects no agents")
            unknown = intervention.agent_ids - policies.keys()
            if unknown:
                raise ValueError(
                    f"intervention {intervention.name!r} selects unknown agents {sorted(unknown)!r}"
                )
            for agent_id in intervention.agent_ids:
                sites = policies[agent_id].activation_sites
                if intervention.site not in sites:
                    raise ValueError(
                        f"intervention {intervention.name!r} selects unknown site "
                        f"{agent_id}.{intervention.site}"
                    )
                site = sites[intervention.site]
                allowed = site.readout if intervention.target == "readout" else site.recurrent
                if not allowed:
                    raise ValueError(
                        f"site {agent_id}.{site.name} does not support {intervention.target} interventions"
                    )

    def matches(
        self,
        *,
        agent_id: AgentId,
        site: str,
        target: InterventionTarget,
    ) -> bool:
        return any(
            agent_id in intervention.agent_ids
            and intervention.site == site
            and intervention.target == target
            for intervention in self.interventions
        )

    def apply(
        self,
        *,
        agent_id: AgentId,
        site: ActivationSite,
        target: InterventionTarget,
        step: int,
        episode_indices: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        result = value
        context = InterventionContext(
            agent_id=agent_id,
            site=site,
            target=target,
            step=step,
            episode_indices=episode_indices,
        )
        for intervention in self.interventions:
            if (
                agent_id not in intervention.agent_ids
                or intervention.site != site.name
                or intervention.target != target
            ):
                continue
            transformed = intervention.apply(result, context)
            if transformed.shape != result.shape:
                raise ValueError(
                    f"intervention {intervention.name!r} changed shape "
                    f"{tuple(result.shape)} to {tuple(transformed.shape)}"
                )
            if transformed.device != result.device or transformed.dtype != result.dtype:
                raise ValueError(
                    f"intervention {intervention.name!r} changed device or dtype"
                )
            if not torch.isfinite(transformed).all():
                raise ValueError(f"intervention {intervention.name!r} produced non-finite values")
            result = transformed
        return result

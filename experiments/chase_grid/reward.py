"""Reward rules for the chase-grid experiment.

This is the only file a researcher needs to edit to change the task objective.
The environment supplies mutually exclusive transition events; this module
assigns their values and precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mouse_run_run.core.types import AgentId


TaskName = Literal["social", "non_social"]


@dataclass(frozen=True)
class RewardConfig:
    task: TaskName = "social"
    chaser_step: float = -0.1
    explorer_step: float = -0.5
    chaser_new_field: float = 0.1
    explorer_new_field: float = 1.0
    social_chaser_collision: float = 1.0
    social_explorer_collision: float = -1.0
    non_social_chaser_collision: float = -0.1
    non_social_explorer_collision: float = -0.5

    def validate(self) -> None:
        if self.task not in ("social", "non_social"):
            raise ValueError(f"unsupported reward task: {self.task!r}")


class GridReward:
    def __init__(
        self,
        config: RewardConfig,
        *,
        chaser_id: AgentId,
        explorer_id: AgentId,
    ) -> None:
        config.validate()
        self.config = config
        self.chaser_id = chaser_id
        self.explorer_id = explorer_id

    def compute(
        self,
        *,
        active: torch.Tensor,
        collision: torch.Tensor,
        chaser_new_field: torch.Tensor,
        explorer_new_field: torch.Tensor,
    ) -> dict[AgentId, torch.Tensor]:
        if self.config.task == "social":
            chaser_collision = self.config.social_chaser_collision
            explorer_collision = self.config.social_explorer_collision
        else:
            chaser_collision = self.config.non_social_chaser_collision
            explorer_collision = self.config.non_social_explorer_collision

        chaser = torch.full_like(active, self.config.chaser_step, dtype=torch.float32)
        explorer = torch.full_like(active, self.config.explorer_step, dtype=torch.float32)
        chaser = torch.where(chaser_new_field, self.config.chaser_new_field, chaser)
        explorer = torch.where(explorer_new_field, self.config.explorer_new_field, explorer)
        chaser = torch.where(collision, chaser_collision, chaser)
        explorer = torch.where(collision, explorer_collision, explorer)
        mask = active.to(torch.float32)
        return {
            self.chaser_id: chaser * mask,
            self.explorer_id: explorer * mask,
        }

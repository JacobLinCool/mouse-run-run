"""Editable chase-grid environment semantics and replay renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from experiments.chase_grid.reward import GridReward, RewardConfig
from mouse_run_run.core.environment import EnvReset, EnvTransition
from mouse_run_run.core.experiment import RuntimeConfig
from mouse_run_run.core.types import AgentId, TensorMap, require_agent_keys


CHASER = AgentId("chaser")
EXPLORER = AgentId("explorer")
AGENT_IDS = (CHASER, EXPLORER)

PartnerVisibility = Literal["partial", "none", "full"]
SpawnMode = Literal["full_grid", "exclude_last"]

ACTION_DELTAS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)


@dataclass(frozen=True)
class ChaseGridConfig:
    grid_size: int = 10
    vision_radius: int = 3
    max_steps: int = 100
    partner_visibility: PartnerVisibility = "partial"
    spawn_mode: SpawnMode = "full_grid"
    reward: RewardConfig = field(default_factory=RewardConfig)

    def validate(self) -> None:
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two")
        if not 0 <= self.vision_radius < self.grid_size:
            raise ValueError("vision_radius must be in [0, grid_size)")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.partner_visibility not in ("partial", "none", "full"):
            raise ValueError(f"invalid partner_visibility: {self.partner_visibility!r}")
        if self.spawn_mode not in ("full_grid", "exclude_last"):
            raise ValueError(f"invalid spawn_mode: {self.spawn_mode!r}")
        self.reward.validate()


class ChaseGridEnvironment:
    """Batched, sequential-movement chaser/explorer environment."""

    def __init__(
        self,
        config: ChaseGridConfig,
        runtime: RuntimeConfig,
        device: torch.device,
    ) -> None:
        config.validate()
        runtime.validate()
        self.config = config
        self.runtime = runtime
        self._device = device
        self._batch_size = runtime.batch_size
        self._batch = torch.arange(runtime.batch_size, device=device)
        self._deltas = torch.tensor(ACTION_DELTAS, dtype=torch.long, device=device)
        self._reward = GridReward(config.reward, chaser_id=CHASER, explorer_id=EXPLORER)
        shape = (runtime.batch_size, 2)
        self.chaser_position = torch.empty(shape, dtype=torch.long, device=device)
        self.explorer_position = torch.empty_like(self.chaser_position)
        visited_shape = (runtime.batch_size, config.grid_size, config.grid_size)
        self.chaser_visited = torch.empty(visited_shape, dtype=torch.bool, device=device)
        self.explorer_visited = torch.empty_like(self.chaser_visited)
        self.step_count = torch.empty(runtime.batch_size, dtype=torch.long, device=device)
        self.done = torch.empty(runtime.batch_size, dtype=torch.bool, device=device)

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        return AGENT_IDS

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self, *, generator: torch.Generator) -> EnvReset:
        spawn_high = (
            self.config.grid_size - 1
            if self.config.spawn_mode == "exclude_last"
            else self.config.grid_size
        )
        self.chaser_position = torch.randint(
            spawn_high,
            (self.batch_size, 2),
            generator=generator,
            device=generator.device,
        ).to(self.device)
        self.explorer_position = torch.randint(
            spawn_high,
            (self.batch_size, 2),
            generator=generator,
            device=generator.device,
        ).to(self.device)
        same = self._positions_equal(self.chaser_position, self.explorer_position)
        while bool(same.any()):
            count = int(same.sum())
            self.explorer_position[same] = torch.randint(
                spawn_high,
                (count, 2),
                generator=generator,
                device=generator.device,
            ).to(self.device)
            same = self._positions_equal(self.chaser_position, self.explorer_position)

        self.chaser_visited.zero_()
        self.explorer_visited.zero_()
        self.chaser_visited[
            self._batch, self.chaser_position[:, 0], self.chaser_position[:, 1]
        ] = True
        self.explorer_visited[
            self._batch, self.explorer_position[:, 0], self.explorer_position[:, 1]
        ] = True
        self.step_count.zero_()
        self.done.zero_()
        return EnvReset(observations=self._observations(), world=self._world())

    def step(self, actions: dict[AgentId, torch.Tensor]) -> EnvTransition:
        require_agent_keys(actions, AGENT_IDS, label="actions")
        for agent_id, action in actions.items():
            if action.shape != (self.batch_size,) or action.dtype != torch.long:
                raise ValueError(
                    f"{agent_id} actions must be int64 shape ({self.batch_size},)"
                )
            if bool(((action < 0) | (action >= len(ACTION_DELTAS))).any()):
                raise ValueError(f"{agent_id} action outside valid range")
        if self.runtime.environment_backend == "triton":
            return self._step_triton(actions)
        return self._step_torch(actions)

    def _step_torch(self, actions: dict[AgentId, torch.Tensor]) -> EnvTransition:
        active = ~self.done
        old_chaser = self.chaser_position.clone()
        old_explorer = self.explorer_position.clone()

        chaser_candidate = self._candidate(old_chaser, actions[CHASER])
        chaser_collision = active & self._positions_equal(chaser_candidate, old_explorer)
        chaser_next = torch.where(chaser_collision[:, None], old_chaser, chaser_candidate)
        self.chaser_position = torch.where(active[:, None], chaser_next, old_chaser)

        explorer_candidate = self._candidate(old_explorer, actions[EXPLORER])
        explorer_collision = active & self._positions_equal(
            explorer_candidate, self.chaser_position
        )
        explorer_next = torch.where(
            explorer_collision[:, None], old_explorer, explorer_candidate
        )
        self.explorer_position = torch.where(active[:, None], explorer_next, old_explorer)

        chaser_new = active & ~chaser_collision & ~self.chaser_visited[
            self._batch, self.chaser_position[:, 0], self.chaser_position[:, 1]
        ]
        explorer_new = active & ~explorer_collision & ~self.explorer_visited[
            self._batch, self.explorer_position[:, 0], self.explorer_position[:, 1]
        ]
        self.chaser_visited[
            self._batch, self.chaser_position[:, 0], self.chaser_position[:, 1]
        ] |= active & ~chaser_collision
        self.explorer_visited[
            self._batch, self.explorer_position[:, 0], self.explorer_position[:, 1]
        ] |= active & ~explorer_collision

        old_distance = _distance(old_chaser, old_explorer)
        chaser_distance = _distance(self.chaser_position, old_explorer)
        explorer_distance = _distance(self.explorer_position, old_chaser)
        chaser_approach = (
            active & ~chaser_collision & ~chaser_new & (chaser_distance < old_distance)
        )
        explorer_escape = (
            active
            & ~explorer_collision
            & ~explorer_new
            & (explorer_distance > old_distance)
        )
        collision = chaser_collision | explorer_collision
        rewards = self._reward.compute(
            active=active,
            collision=collision,
            chaser_new_field=chaser_new,
            explorer_new_field=explorer_new,
        )
        self.step_count += active.long()
        self.done |= self.step_count >= self.config.max_steps
        chaser_visible, explorer_visible = self._physical_visibilities()
        events = {
            "collision": collision,
            "chaser_collision": chaser_collision,
            "explorer_collision": explorer_collision,
            "chaser_new_field": chaser_new,
            "explorer_new_field": explorer_new,
            "chaser_approach": chaser_approach,
            "explorer_escape": explorer_escape,
            "explorer_escape_close": explorer_escape & (explorer_distance < 3.0),
            "explorer_escape_near": (
                explorer_escape & (explorer_distance >= 3.0) & (explorer_distance < 5.0)
            ),
            "explorer_escape_far": explorer_escape & (explorer_distance >= 5.0),
            "chaser_partner_visible": chaser_visible,
            "explorer_partner_visible": explorer_visible,
        }
        return EnvTransition(
            observations=self._observations(),
            rewards=rewards,
            terminated=torch.zeros_like(self.done),
            truncated=self.done.clone(),
            active=active,
            world=self._world(),
            events=events,
        )

    def _step_triton(self, actions: dict[AgentId, torch.Tensor]) -> EnvTransition:
        if self.device.type != "cuda":
            raise RuntimeError("the Triton chase-grid backend requires CUDA")
        from experiments.chase_grid.triton_backend import triton_step

        return triton_step(self, actions)

    def _observations(self) -> dict[AgentId, torch.Tensor]:
        return {
            CHASER: self._observation(self.chaser_position, self.explorer_position),
            EXPLORER: self._observation(self.explorer_position, self.chaser_position),
        }

    def _observation(self, own: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        size = self.config.grid_size
        observation = torch.zeros(
            self.batch_size, 2, size, size, dtype=torch.float32, device=self.device
        )
        observation[self._batch, 0, own[:, 0], own[:, 1]] = 1.0
        visible = self._observed_partner_visible(own, other)
        observation[self._batch, 1, other[:, 0], other[:, 1]] = visible.float()
        return observation

    def _world(self) -> TensorMap:
        chaser_visible, explorer_visible = self._physical_visibilities()
        return {
            "chaser_position": self.chaser_position.clone(),
            "explorer_position": self.explorer_position.clone(),
            "distance": _distance(self.chaser_position, self.explorer_position),
            "chaser_partner_visible": chaser_visible,
            "explorer_partner_visible": explorer_visible,
            "step": self.step_count.clone(),
        }

    def _physical_visibilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._in_fov(self.chaser_position, self.explorer_position),
            self._in_fov(self.explorer_position, self.chaser_position),
        )

    def _observed_partner_visible(
        self, own: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        if self.config.partner_visibility == "none":
            return torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        if self.config.partner_visibility == "full":
            return torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        return self._in_fov(own, other)

    def _in_fov(self, own: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return (own - other).abs().amax(dim=1) <= self.config.vision_radius

    def _candidate(self, position: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return (position + self._deltas[action]).clamp(0, self.config.grid_size - 1)

    @staticmethod
    def _positions_equal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left == right).all(dim=1)


class GridRenderer:
    def __init__(self, config: ChaseGridConfig) -> None:
        self.config = config

    def frame(self, world: TensorMap, *, batch_index: int, step: int) -> dict[str, object]:
        return {
            "grid_size": self.config.grid_size,
            "step": step,
            "chaser": world["chaser_position"][step, batch_index].tolist(),
            "explorer": world["explorer_position"][step, batch_index].tolist(),
            "distance": float(world["distance"][step, batch_index]),
            "chaser_partner_visible": bool(
                world["chaser_partner_visible"][step, batch_index]
            ),
            "explorer_partner_visible": bool(
                world["explorer_partner_visible"][step, batch_index]
            ),
        }


def make_environment(
    config: ChaseGridConfig,
    runtime: RuntimeConfig,
    device: torch.device,
) -> ChaseGridEnvironment:
    return ChaseGridEnvironment(config, runtime, device)


def _distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().sum(dim=1).sqrt()

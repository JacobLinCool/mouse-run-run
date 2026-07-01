from dataclasses import dataclass
from typing import Literal

import torch


TaskName = Literal["social", "non_social"]
PartnerVisibility = Literal["partial", "none", "full"]


ACTION_DELTAS = torch.tensor(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ],
    dtype=torch.long,
)


@dataclass(frozen=True)
class GridWorldConfig:
    grid_size: int = 10
    vision_radius: int = 3
    max_steps: int = 100
    task: TaskName = "social"
    partner_visibility: PartnerVisibility = "partial"

    @property
    def observation_size(self) -> int:
        return 2 * self.grid_size * self.grid_size


@dataclass(frozen=True)
class StepResult:
    chaser_observation: torch.Tensor
    explorer_observation: torch.Tensor
    chaser_reward: torch.Tensor
    explorer_reward: torch.Tensor
    done: torch.Tensor
    active: torch.Tensor
    collision: torch.Tensor
    chaser_new_field: torch.Tensor
    explorer_new_field: torch.Tensor
    chaser_partner_visible: torch.Tensor
    explorer_partner_visible: torch.Tensor
    chaser_approach: torch.Tensor
    explorer_escape: torch.Tensor
    explorer_escape_close: torch.Tensor
    explorer_escape_near: torch.Tensor
    explorer_escape_far: torch.Tensor
    distance: torch.Tensor
    chaser_position: torch.Tensor
    explorer_position: torch.Tensor


@dataclass(frozen=True)
class TrainingStepResult:
    chaser_reward: torch.Tensor
    explorer_reward: torch.Tensor
    done: torch.Tensor
    active: torch.Tensor
    collision: torch.Tensor
    chaser_new_field: torch.Tensor
    explorer_new_field: torch.Tensor
    chaser_partner_visible: torch.Tensor
    explorer_partner_visible: torch.Tensor
    distance: torch.Tensor


@dataclass(frozen=True)
class _TransitionResult:
    chaser_reward: torch.Tensor
    explorer_reward: torch.Tensor
    done: torch.Tensor
    active: torch.Tensor
    collision: torch.Tensor
    chaser_new_field: torch.Tensor
    explorer_new_field: torch.Tensor
    chaser_partner_visible: torch.Tensor
    explorer_partner_visible: torch.Tensor
    chaser_approach: torch.Tensor
    explorer_escape: torch.Tensor
    explorer_escape_close: torch.Tensor
    explorer_escape_near: torch.Tensor
    explorer_escape_far: torch.Tensor
    distance: torch.Tensor


class BatchedChaseEnv:
    def __init__(
        self,
        config: GridWorldConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.batch_index = torch.arange(batch_size, device=device)
        self.deltas = ACTION_DELTAS.to(device)
        self.chaser_position = torch.empty(batch_size, 2, dtype=torch.long, device=device)
        self.explorer_position = torch.empty(batch_size, 2, dtype=torch.long, device=device)
        self.chaser_visited = torch.empty(
            batch_size,
            config.grid_size,
            config.grid_size,
            dtype=torch.bool,
            device=device,
        )
        self.explorer_visited = torch.empty_like(self.chaser_visited)
        self.done = torch.empty(batch_size, dtype=torch.bool, device=device)
        self.step_count = torch.empty(batch_size, dtype=torch.long, device=device)

    def reset_state(self) -> None:
        grid_size = self.config.grid_size
        self.chaser_position = torch.randint(
            grid_size,
            (self.batch_size, 2),
            dtype=torch.long,
            device=self.device,
        )
        self.explorer_position = torch.randint(
            grid_size,
            (self.batch_size, 2),
            dtype=torch.long,
            device=self.device,
        )

        same_position = self._same_position()
        while same_position.any():
            count = int(same_position.sum().item())
            self.explorer_position[same_position] = torch.randint(
                grid_size,
                (count, 2),
                dtype=torch.long,
                device=self.device,
            )
            same_position = self._same_position()

        self.chaser_visited.zero_()
        self.explorer_visited.zero_()
        self.chaser_visited[
            self.batch_index,
            self.chaser_position[:, 0],
            self.chaser_position[:, 1],
        ] = True
        self.explorer_visited[
            self.batch_index,
            self.explorer_position[:, 0],
            self.explorer_position[:, 1],
        ] = True
        self.done.zero_()
        self.step_count.zero_()

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.reset_state()
        return self.observations()

    def observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._observation(self.chaser_position, self.explorer_position),
            self._observation(self.explorer_position, self.chaser_position),
        )

    def step(
        self,
        chaser_action: torch.Tensor,
        explorer_action: torch.Tensor,
    ) -> StepResult:
        transition = self._advance(chaser_action, explorer_action)
        chaser_observation, explorer_observation = self.observations()
        return StepResult(
            chaser_observation=chaser_observation,
            explorer_observation=explorer_observation,
            chaser_reward=transition.chaser_reward,
            explorer_reward=transition.explorer_reward,
            done=transition.done,
            active=transition.active,
            collision=transition.collision,
            chaser_new_field=transition.chaser_new_field,
            explorer_new_field=transition.explorer_new_field,
            chaser_partner_visible=transition.chaser_partner_visible,
            explorer_partner_visible=transition.explorer_partner_visible,
            chaser_approach=transition.chaser_approach,
            explorer_escape=transition.explorer_escape,
            explorer_escape_close=transition.explorer_escape_close,
            explorer_escape_near=transition.explorer_escape_near,
            explorer_escape_far=transition.explorer_escape_far,
            distance=transition.distance,
            chaser_position=self.chaser_position.clone(),
            explorer_position=self.explorer_position.clone(),
        )

    def step_training(
        self,
        chaser_action: torch.Tensor,
        explorer_action: torch.Tensor,
    ) -> TrainingStepResult:
        transition = self._advance(chaser_action, explorer_action)
        return TrainingStepResult(
            chaser_reward=transition.chaser_reward,
            explorer_reward=transition.explorer_reward,
            done=transition.done,
            active=transition.active,
            collision=transition.collision,
            chaser_new_field=transition.chaser_new_field,
            explorer_new_field=transition.explorer_new_field,
            chaser_partner_visible=transition.chaser_partner_visible,
            explorer_partner_visible=transition.explorer_partner_visible,
            distance=transition.distance,
        )

    def step_training_fused(
        self,
        chaser_action: torch.Tensor,
        explorer_action: torch.Tensor,
    ) -> TrainingStepResult:
        from mouse_run_run.triton_env import triton_step_training

        return triton_step_training(self, chaser_action, explorer_action)

    def partner_visible(self) -> torch.Tensor:
        return self._partner_visible(self.chaser_position, self.explorer_position)

    def _advance(
        self,
        chaser_action: torch.Tensor,
        explorer_action: torch.Tensor,
    ) -> _TransitionResult:
        active = ~self.done
        old_chaser_position = self.chaser_position.clone()
        old_explorer_position = self.explorer_position.clone()

        chaser_candidate = self._candidate_position(old_chaser_position, chaser_action)
        chaser_collision = active & self._positions_equal(
            chaser_candidate,
            old_explorer_position,
        )
        chaser_next = torch.where(
            chaser_collision[:, None],
            old_chaser_position,
            chaser_candidate,
        )
        self.chaser_position = torch.where(active[:, None], chaser_next, old_chaser_position)

        explorer_candidate = self._candidate_position(old_explorer_position, explorer_action)
        explorer_collision = active & self._positions_equal(
            explorer_candidate,
            self.chaser_position,
        )
        explorer_next = torch.where(
            explorer_collision[:, None],
            old_explorer_position,
            explorer_candidate,
        )
        self.explorer_position = torch.where(active[:, None], explorer_next, old_explorer_position)

        chaser_new_field = active & ~chaser_collision & ~self.chaser_visited[
            self.batch_index,
            self.chaser_position[:, 0],
            self.chaser_position[:, 1],
        ]
        explorer_new_field = active & ~explorer_collision & ~self.explorer_visited[
            self.batch_index,
            self.explorer_position[:, 0],
            self.explorer_position[:, 1],
        ]
        self.chaser_visited[
            self.batch_index,
            self.chaser_position[:, 0],
            self.chaser_position[:, 1],
        ] |= active & ~chaser_collision
        self.explorer_visited[
            self.batch_index,
            self.explorer_position[:, 0],
            self.explorer_position[:, 1],
        ] |= active & ~explorer_collision

        old_distance = _euclidean_distance(old_chaser_position, old_explorer_position)
        chaser_partner_distance = _euclidean_distance(self.chaser_position, old_explorer_position)
        explorer_partner_distance = _euclidean_distance(self.explorer_position, old_chaser_position)
        chaser_approach = active & ~chaser_collision & (chaser_partner_distance < old_distance)
        explorer_escape = active & ~explorer_collision & (explorer_partner_distance > old_distance)
        explorer_escape_far = explorer_escape & (explorer_partner_distance >= 5.0)
        explorer_escape_near = (
            explorer_escape
            & (explorer_partner_distance >= 3.0)
            & (explorer_partner_distance < 5.0)
        )
        explorer_escape_close = explorer_escape & (explorer_partner_distance < 3.0)

        collision = chaser_collision | explorer_collision
        chaser_reward, explorer_reward = self._rewards(
            active=active,
            collision=collision,
            chaser_new_field=chaser_new_field,
            explorer_new_field=explorer_new_field,
        )

        self.step_count = self.step_count + active.long()
        self.done = self.done | (self.step_count >= self.config.max_steps)

        chaser_visible = self._partner_visible(self.chaser_position, self.explorer_position)
        explorer_visible = self._partner_visible(self.explorer_position, self.chaser_position)
        return _TransitionResult(
            chaser_reward=chaser_reward,
            explorer_reward=explorer_reward,
            done=self.done.clone(),
            active=active,
            collision=collision,
            chaser_new_field=chaser_new_field,
            explorer_new_field=explorer_new_field,
            chaser_partner_visible=chaser_visible,
            explorer_partner_visible=explorer_visible,
            chaser_approach=chaser_approach,
            explorer_escape=explorer_escape,
            explorer_escape_close=explorer_escape_close,
            explorer_escape_near=explorer_escape_near,
            explorer_escape_far=explorer_escape_far,
            distance=self.distance(),
        )

    def distance(self) -> torch.Tensor:
        return _euclidean_distance(self.chaser_position, self.explorer_position)

    def _candidate_position(self, position: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        next_position = position + self.deltas[action]
        return next_position.clamp(0, self.config.grid_size - 1)

    def _observation(self, own_position: torch.Tensor, other_position: torch.Tensor) -> torch.Tensor:
        grid_size = self.config.grid_size
        observation = torch.zeros(
            self.batch_size,
            2,
            grid_size,
            grid_size,
            device=self.device,
        )
        observation[
            self.batch_index,
            0,
            own_position[:, 0],
            own_position[:, 1],
        ] = 1.0

        visible = self._partner_visible(own_position, other_position)
        if visible.any():
            visible_index = self.batch_index[visible]
            observation[
                visible_index,
                1,
                other_position[visible, 0],
                other_position[visible, 1],
            ] = 1.0

        return observation.flatten(start_dim=1)

    def _partner_visible(
        self,
        own_position: torch.Tensor,
        other_position: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.partner_visibility == "none":
            return torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        if self.config.partner_visibility == "full":
            return torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        offset = (own_position - other_position).abs()
        return offset.max(dim=1).values <= self.config.vision_radius

    def _rewards(
        self,
        *,
        active: torch.Tensor,
        collision: torch.Tensor,
        chaser_new_field: torch.Tensor,
        explorer_new_field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.task == "social":
            chaser_collision_reward = 1.0
            explorer_collision_reward = -1.0
        elif self.config.task == "non_social":
            chaser_collision_reward = -0.1
            explorer_collision_reward = -0.5
        else:
            raise ValueError(f"Unsupported task: {self.config.task}")

        chaser_reward = torch.full((self.batch_size,), -0.1, device=self.device)
        explorer_reward = torch.full((self.batch_size,), -0.5, device=self.device)
        chaser_reward = torch.where(chaser_new_field, 0.1, chaser_reward)
        explorer_reward = torch.where(explorer_new_field, 1.0, explorer_reward)
        chaser_reward = torch.where(collision, chaser_collision_reward, chaser_reward)
        explorer_reward = torch.where(collision, explorer_collision_reward, explorer_reward)
        return chaser_reward * active.float(), explorer_reward * active.float()

    def _same_position(self) -> torch.Tensor:
        return self._positions_equal(self.chaser_position, self.explorer_position)

    @staticmethod
    def _positions_equal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left == right).all(dim=1)


def _euclidean_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().sum(dim=1).sqrt()

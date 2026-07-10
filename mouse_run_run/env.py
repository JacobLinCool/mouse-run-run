from dataclasses import dataclass
from typing import Literal

import torch


TaskName = Literal["social", "non_social"]
PartnerVisibility = Literal["partial", "none", "full"]
SpawnMode = Literal["full_grid", "official_exclude_last"]


# [OFFICIAL-ARENAS] maps actions as 0=up, 1=right, 2=down, 3=left.
# Source IDs resolve in experiments/paper_marl_official_dynamics_2026/SPEC.md,
# "Implementation Source Registry".
# Single source of truth for the action table: the CPU path uses the tensor
# below and the Triton kernel receives these values as constexpr arguments.
ACTION_DELTA_VALUES: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)

ACTION_DELTAS = torch.tensor(ACTION_DELTA_VALUES, dtype=torch.long)

ACTION_COUNT = len(ACTION_DELTA_VALUES)

# [PAPER-METHODS] Supplementary Table 3; [OFFICIAL-ARENAS] lines that
# implement r1=explorer and r2=chaser. The names below use paper roles.
# Single source of truth for the reward table: the CPU path (_rewards) uses
# these values directly and the Triton kernel receives them as constexpr
# arguments at launch.
CHASER_STEP_REWARD = -0.1
EXPLORER_STEP_REWARD = -0.5
CHASER_NEW_FIELD_REWARD = 0.1
EXPLORER_NEW_FIELD_REWARD = 1.0
SOCIAL_CHASER_COLLISION_REWARD = 1.0
SOCIAL_EXPLORER_COLLISION_REWARD = -1.0
NON_SOCIAL_CHASER_COLLISION_REWARD = -0.1
NON_SOCIAL_EXPLORER_COLLISION_REWARD = -0.5


@dataclass(frozen=True)
class GridWorldConfig:
    # [PAPER-METHODS], [OFFICIAL-TRAIN], [OFFICIAL-ARENAS]: 10x10 grid,
    # 7x7 field of view (radius 3), and 100-step training episodes.
    grid_size: int = 10
    vision_radius: int = 3
    max_steps: int = 100
    task: TaskName = "social"
    partner_visibility: PartnerVisibility = "partial"
    # full_grid is the paper-text/modern default. Official-dynamics configs
    # explicitly select official_exclude_last; see [OFFICIAL-ARENAS].
    spawn_mode: SpawnMode = "full_grid"

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
    chaser_collision: torch.Tensor
    explorer_collision: torch.Tensor
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
    chaser_collision: torch.Tensor
    explorer_collision: torch.Tensor
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
        self._never_visible = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._always_visible = torch.ones(batch_size, dtype=torch.bool, device=device)

    def reset_state(self) -> None:
        grid_size = self.config.grid_size
        # [OFFICIAL-ARENAS] calls np.random.randint(height - 1), whose exclusive
        # high bound yields coordinates 0..8 on the released 10x10 arena.
        spawn_high = (
            grid_size - 1
            if self.config.spawn_mode == "official_exclude_last"
            else grid_size
        )
        if spawn_high < 1:
            raise ValueError(
                "official_exclude_last requires a grid_size of at least 2"
            )
        self.chaser_position = torch.randint(
            spawn_high,
            (self.batch_size, 2),
            dtype=torch.long,
            device=self.device,
        )
        self.explorer_position = torch.randint(
            spawn_high,
            (self.batch_size, 2),
            dtype=torch.long,
            device=self.device,
        )

        same_position = self._same_position()
        while same_position.any():
            count = int(same_position.sum().item())
            self.explorer_position[same_position] = torch.randint(
                spawn_high,
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
            chaser_collision=transition.chaser_collision,
            explorer_collision=transition.explorer_collision,
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

    def partner_visibilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Physical partner-in-FOV for (chaser view, explorer view)."""
        return (
            self._partner_in_fov(self.chaser_position, self.explorer_position),
            self._partner_in_fov(self.explorer_position, self.chaser_position),
        )

    def _advance(
        self,
        chaser_action: torch.Tensor,
        explorer_action: torch.Tensor,
    ) -> _TransitionResult:
        active = ~self.done
        old_chaser_position = self.chaser_position.clone()
        old_explorer_position = self.explorer_position.clone()

        # [OFFICIAL-ARENAS] agent2=chaser moves first; agent1=explorer then
        # resolves against the chaser's updated position.
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
        # [OFFICIAL-ARENAS] event precedence: collision > own new field >
        # approach/escape; a new-field step never also emits approach/escape.
        chaser_approach = (
            active
            & ~chaser_collision
            & ~chaser_new_field
            & (chaser_partner_distance < old_distance)
        )
        explorer_escape = (
            active
            & ~explorer_collision
            & ~explorer_new_field
            & (explorer_partner_distance > old_distance)
        )
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

        # This is a behavioral event, not an observation-channel flag. The
        # official non-social analysis still measures physical partner-in-FOV
        # even though the partner channel is hidden from the policy.
        chaser_visible = self._partner_in_fov(self.chaser_position, self.explorer_position)
        explorer_visible = self._partner_in_fov(self.explorer_position, self.chaser_position)
        return _TransitionResult(
            chaser_reward=chaser_reward,
            explorer_reward=explorer_reward,
            done=self.done.clone(),
            active=active,
            collision=collision,
            chaser_collision=chaser_collision,
            explorer_collision=explorer_collision,
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

        # Branch-free: writing 0.0 into an already-zero cell is a no-op, and
        # avoiding the data-dependent branch keeps this free of device syncs.
        visible = self._partner_visible(own_position, other_position)
        observation[
            self.batch_index,
            1,
            other_position[:, 0],
            other_position[:, 1],
        ] = visible.to(observation.dtype)

        return observation.flatten(start_dim=1)

    def _partner_visible(
        self,
        own_position: torch.Tensor,
        other_position: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.partner_visibility == "none":
            return self._never_visible
        if self.config.partner_visibility == "full":
            return self._always_visible
        offset = (own_position - other_position).abs()
        return offset.max(dim=1).values <= self.config.vision_radius

    def _partner_in_fov(
        self,
        own_position: torch.Tensor,
        other_position: torch.Tensor,
    ) -> torch.Tensor:
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
        # [PAPER-METHODS] Supplementary Table 3; [OFFICIAL-ARENAS] lines that
        # implement r1=explorer and r2=chaser. The names below use paper roles.
        if self.config.task == "social":
            chaser_collision_reward = SOCIAL_CHASER_COLLISION_REWARD
            explorer_collision_reward = SOCIAL_EXPLORER_COLLISION_REWARD
        elif self.config.task == "non_social":
            chaser_collision_reward = NON_SOCIAL_CHASER_COLLISION_REWARD
            explorer_collision_reward = NON_SOCIAL_EXPLORER_COLLISION_REWARD
        else:
            raise ValueError(f"Unsupported task: {self.config.task}")

        chaser_reward = torch.full((self.batch_size,), CHASER_STEP_REWARD, device=self.device)
        explorer_reward = torch.full((self.batch_size,), EXPLORER_STEP_REWARD, device=self.device)
        chaser_reward = torch.where(chaser_new_field, CHASER_NEW_FIELD_REWARD, chaser_reward)
        explorer_reward = torch.where(explorer_new_field, EXPLORER_NEW_FIELD_REWARD, explorer_reward)
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

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mouse_run_run.env import (
    ACTION_DELTA_VALUES,
    CHASER_NEW_FIELD_REWARD,
    CHASER_STEP_REWARD,
    EXPLORER_NEW_FIELD_REWARD,
    EXPLORER_STEP_REWARD,
    NON_SOCIAL_CHASER_COLLISION_REWARD,
    NON_SOCIAL_EXPLORER_COLLISION_REWARD,
    SOCIAL_CHASER_COLLISION_REWARD,
    SOCIAL_EXPLORER_COLLISION_REWARD,
    BatchedChaseEnv,
    TrainingStepResult,
)


# [LOCAL-CALIBRATION] Triton is a local acceleration path, not part of the
# paper or released implementation. Its semantics are gated against env.py;
# upstream source IDs resolve in the official-dynamics SPEC source registry.
def triton_step_training(
    env: BatchedChaseEnv,
    chaser_action: torch.Tensor,
    explorer_action: torch.Tensor,
) -> TrainingStepResult:
    if env.device.type != "cuda":
        raise RuntimeError("Triton environment stepping requires CUDA.")

    batch_size = env.batch_size
    block_size = triton.next_power_of_2(batch_size)
    chaser_reward = torch.empty(batch_size, dtype=torch.float32, device=env.device)
    explorer_reward = torch.empty_like(chaser_reward)
    done = torch.empty_like(env.done)
    active = torch.empty_like(env.done)
    collision = torch.empty_like(env.done)
    chaser_new_field = torch.empty_like(env.done)
    explorer_new_field = torch.empty_like(env.done)
    chaser_partner_visible = torch.empty_like(env.done)
    explorer_partner_visible = torch.empty_like(env.done)
    distance = torch.empty(batch_size, dtype=torch.float32, device=env.device)

    # Task-conditional collision rewards resolve here so the kernel receives
    # the shared env.py constants; constexpr arguments specialize the compiled
    # kernel per task exactly as the previous TASK_ID branch did.
    if env.config.task == "social":
        chaser_collision_reward = SOCIAL_CHASER_COLLISION_REWARD
        explorer_collision_reward = SOCIAL_EXPLORER_COLLISION_REWARD
    else:
        chaser_collision_reward = NON_SOCIAL_CHASER_COLLISION_REWARD
        explorer_collision_reward = NON_SOCIAL_EXPLORER_COLLISION_REWARD
    (up_row, up_col), (right_row, right_col), (down_row, down_col), (left_row, left_col) = (
        ACTION_DELTA_VALUES
    )

    _step_training_kernel[(1,)](
        env.chaser_position,
        env.explorer_position,
        env.chaser_visited,
        env.explorer_visited,
        env.done,
        env.step_count,
        chaser_action,
        explorer_action,
        chaser_reward,
        explorer_reward,
        done,
        active,
        collision,
        chaser_new_field,
        explorer_new_field,
        chaser_partner_visible,
        explorer_partner_visible,
        distance,
        batch_size,
        BLOCK_SIZE=block_size,
        GRID_SIZE=env.config.grid_size,
        MAX_STEPS=env.config.max_steps,
        VISION_RADIUS=env.config.vision_radius,
        UP_ROW_DELTA=up_row,
        UP_COL_DELTA=up_col,
        RIGHT_ROW_DELTA=right_row,
        RIGHT_COL_DELTA=right_col,
        DOWN_ROW_DELTA=down_row,
        DOWN_COL_DELTA=down_col,
        LEFT_ROW_DELTA=left_row,
        LEFT_COL_DELTA=left_col,
        CHASER_STEP_REWARD_VALUE=CHASER_STEP_REWARD,
        EXPLORER_STEP_REWARD_VALUE=EXPLORER_STEP_REWARD,
        CHASER_NEW_FIELD_REWARD_VALUE=CHASER_NEW_FIELD_REWARD,
        EXPLORER_NEW_FIELD_REWARD_VALUE=EXPLORER_NEW_FIELD_REWARD,
        CHASER_COLLISION_REWARD_VALUE=chaser_collision_reward,
        EXPLORER_COLLISION_REWARD_VALUE=explorer_collision_reward,
    )
    return TrainingStepResult(
        chaser_reward=chaser_reward,
        explorer_reward=explorer_reward,
        done=done,
        active=active,
        collision=collision,
        chaser_new_field=chaser_new_field,
        explorer_new_field=explorer_new_field,
        chaser_partner_visible=chaser_partner_visible,
        explorer_partner_visible=explorer_partner_visible,
        distance=distance,
    )


@triton.jit
def _action_delta(
    action: tl.tensor,
    UP_ROW_DELTA: tl.constexpr,
    UP_COL_DELTA: tl.constexpr,
    RIGHT_ROW_DELTA: tl.constexpr,
    RIGHT_COL_DELTA: tl.constexpr,
    DOWN_ROW_DELTA: tl.constexpr,
    DOWN_COL_DELTA: tl.constexpr,
    LEFT_ROW_DELTA: tl.constexpr,
    LEFT_COL_DELTA: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    # [OFFICIAL-ARENAS]: 0=up, 1=right, 2=down, 3=left. The deltas come from
    # env.ACTION_DELTA_VALUES via the kernel's constexpr arguments.
    row_delta = tl.where(
        action == 0,
        UP_ROW_DELTA,
        tl.where(action == 1, RIGHT_ROW_DELTA, tl.where(action == 2, DOWN_ROW_DELTA, LEFT_ROW_DELTA)),
    )
    col_delta = tl.where(
        action == 0,
        UP_COL_DELTA,
        tl.where(action == 1, RIGHT_COL_DELTA, tl.where(action == 2, DOWN_COL_DELTA, LEFT_COL_DELTA)),
    )
    return row_delta, col_delta


@triton.jit
def _step_training_kernel(
    chaser_position,
    explorer_position,
    chaser_visited,
    explorer_visited,
    done_state,
    step_count,
    chaser_action,
    explorer_action,
    chaser_reward_out,
    explorer_reward_out,
    done_out,
    active_out,
    collision_out,
    chaser_new_field_out,
    explorer_new_field_out,
    chaser_partner_visible_out,
    explorer_partner_visible_out,
    distance_out,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    VISION_RADIUS: tl.constexpr,
    UP_ROW_DELTA: tl.constexpr,
    UP_COL_DELTA: tl.constexpr,
    RIGHT_ROW_DELTA: tl.constexpr,
    RIGHT_COL_DELTA: tl.constexpr,
    DOWN_ROW_DELTA: tl.constexpr,
    DOWN_COL_DELTA: tl.constexpr,
    LEFT_ROW_DELTA: tl.constexpr,
    LEFT_COL_DELTA: tl.constexpr,
    CHASER_STEP_REWARD_VALUE: tl.constexpr,
    EXPLORER_STEP_REWARD_VALUE: tl.constexpr,
    CHASER_NEW_FIELD_REWARD_VALUE: tl.constexpr,
    EXPLORER_NEW_FIELD_REWARD_VALUE: tl.constexpr,
    CHASER_COLLISION_REWARD_VALUE: tl.constexpr,
    EXPLORER_COLLISION_REWARD_VALUE: tl.constexpr,
) -> None:
    batch = tl.arange(0, BLOCK_SIZE)
    mask = batch < batch_size
    position_base = batch * 2
    visited_base = batch * GRID_SIZE * GRID_SIZE

    old_chaser_row = tl.load(chaser_position + position_base, mask=mask, other=0)
    old_chaser_col = tl.load(chaser_position + position_base + 1, mask=mask, other=0)
    old_explorer_row = tl.load(explorer_position + position_base, mask=mask, other=0)
    old_explorer_col = tl.load(explorer_position + position_base + 1, mask=mask, other=0)
    old_done = tl.load(done_state + batch, mask=mask, other=1).to(tl.int1)
    active = ~old_done

    # [OFFICIAL-ARENAS]: agent2=chaser moves before agent1=explorer.
    chaser_action_value = tl.load(chaser_action + batch, mask=mask, other=0)
    chaser_row_delta, chaser_col_delta = _action_delta(
        chaser_action_value,
        UP_ROW_DELTA,
        UP_COL_DELTA,
        RIGHT_ROW_DELTA,
        RIGHT_COL_DELTA,
        DOWN_ROW_DELTA,
        DOWN_COL_DELTA,
        LEFT_ROW_DELTA,
        LEFT_COL_DELTA,
    )
    chaser_candidate_row = tl.minimum(
        tl.maximum(old_chaser_row + chaser_row_delta, 0),
        GRID_SIZE - 1,
    )
    chaser_candidate_col = tl.minimum(
        tl.maximum(old_chaser_col + chaser_col_delta, 0),
        GRID_SIZE - 1,
    )
    chaser_collision = (
        active
        & (chaser_candidate_row == old_explorer_row)
        & (chaser_candidate_col == old_explorer_col)
    )
    chaser_next_row = tl.where(chaser_collision, old_chaser_row, chaser_candidate_row)
    chaser_next_col = tl.where(chaser_collision, old_chaser_col, chaser_candidate_col)
    chaser_next_row = tl.where(active, chaser_next_row, old_chaser_row)
    chaser_next_col = tl.where(active, chaser_next_col, old_chaser_col)

    explorer_action_value = tl.load(explorer_action + batch, mask=mask, other=0)
    explorer_row_delta, explorer_col_delta = _action_delta(
        explorer_action_value,
        UP_ROW_DELTA,
        UP_COL_DELTA,
        RIGHT_ROW_DELTA,
        RIGHT_COL_DELTA,
        DOWN_ROW_DELTA,
        DOWN_COL_DELTA,
        LEFT_ROW_DELTA,
        LEFT_COL_DELTA,
    )
    explorer_candidate_row = tl.minimum(
        tl.maximum(old_explorer_row + explorer_row_delta, 0),
        GRID_SIZE - 1,
    )
    explorer_candidate_col = tl.minimum(
        tl.maximum(old_explorer_col + explorer_col_delta, 0),
        GRID_SIZE - 1,
    )
    explorer_collision = (
        active
        & (explorer_candidate_row == chaser_next_row)
        & (explorer_candidate_col == chaser_next_col)
    )
    explorer_next_row = tl.where(explorer_collision, old_explorer_row, explorer_candidate_row)
    explorer_next_col = tl.where(explorer_collision, old_explorer_col, explorer_candidate_col)
    explorer_next_row = tl.where(active, explorer_next_row, old_explorer_row)
    explorer_next_col = tl.where(active, explorer_next_col, old_explorer_col)

    tl.store(chaser_position + position_base, chaser_next_row, mask=mask)
    tl.store(chaser_position + position_base + 1, chaser_next_col, mask=mask)
    tl.store(explorer_position + position_base, explorer_next_row, mask=mask)
    tl.store(explorer_position + position_base + 1, explorer_next_col, mask=mask)

    chaser_visit_offset = visited_base + chaser_next_row * GRID_SIZE + chaser_next_col
    explorer_visit_offset = visited_base + explorer_next_row * GRID_SIZE + explorer_next_col
    chaser_seen = tl.load(chaser_visited + chaser_visit_offset, mask=mask, other=1).to(tl.int1)
    explorer_seen = tl.load(explorer_visited + explorer_visit_offset, mask=mask, other=1).to(tl.int1)
    chaser_new_field = active & (~chaser_collision) & (~chaser_seen)
    explorer_new_field = active & (~explorer_collision) & (~explorer_seen)
    tl.store(chaser_visited + chaser_visit_offset, True, mask=mask & active & (~chaser_collision))
    tl.store(
        explorer_visited + explorer_visit_offset,
        True,
        mask=mask & active & (~explorer_collision),
    )

    collision = chaser_collision | explorer_collision

    old_step = tl.load(step_count + batch, mask=mask, other=0)
    new_step = old_step + active.to(tl.int64)
    new_done = old_done | (new_step >= MAX_STEPS)
    tl.store(step_count + batch, new_step, mask=mask)
    tl.store(done_state + batch, new_done, mask=mask)

    row_delta = (chaser_next_row - explorer_next_row).to(tl.float32)
    col_delta = (chaser_next_col - explorer_next_col).to(tl.float32)
    distance = tl.sqrt(row_delta * row_delta + col_delta * col_delta)

    abs_row = tl.abs(chaser_next_row - explorer_next_row)
    abs_col = tl.abs(chaser_next_col - explorer_next_col)
    visible = tl.maximum(abs_row, abs_col) <= VISION_RADIUS

    # [PAPER-METHODS] Supplementary Table 3 and [OFFICIAL-ARENAS]; names are
    # translated from released agent2/agent1 to chaser/explorer. The values
    # come from the shared env.py reward constants via constexpr arguments;
    # the task-conditional collision rewards resolve at launch.
    chaser_reward = tl.full((BLOCK_SIZE,), CHASER_STEP_REWARD_VALUE, tl.float32)
    explorer_reward = tl.full((BLOCK_SIZE,), EXPLORER_STEP_REWARD_VALUE, tl.float32)
    chaser_reward = tl.where(chaser_new_field, CHASER_NEW_FIELD_REWARD_VALUE, chaser_reward)
    explorer_reward = tl.where(explorer_new_field, EXPLORER_NEW_FIELD_REWARD_VALUE, explorer_reward)
    chaser_reward = tl.where(collision, CHASER_COLLISION_REWARD_VALUE, chaser_reward)
    explorer_reward = tl.where(collision, EXPLORER_COLLISION_REWARD_VALUE, explorer_reward)
    active_float = active.to(tl.float32)

    tl.store(chaser_reward_out + batch, chaser_reward * active_float, mask=mask)
    tl.store(explorer_reward_out + batch, explorer_reward * active_float, mask=mask)
    tl.store(done_out + batch, new_done, mask=mask)
    tl.store(active_out + batch, active, mask=mask)
    tl.store(collision_out + batch, collision, mask=mask)
    tl.store(chaser_new_field_out + batch, chaser_new_field, mask=mask)
    tl.store(explorer_new_field_out + batch, explorer_new_field, mask=mask)
    tl.store(chaser_partner_visible_out + batch, visible, mask=mask)
    tl.store(explorer_partner_visible_out + batch, visible, mask=mask)
    tl.store(distance_out + batch, distance, mask=mask)

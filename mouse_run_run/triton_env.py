from __future__ import annotations

import torch
import triton
import triton.language as tl

from mouse_run_run.env import BatchedChaseEnv, TrainingStepResult


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
        TASK_ID=0 if env.config.task == "social" else 1,
        VISIBILITY_ID=_visibility_id(env.config.partner_visibility),
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


def _visibility_id(value: str) -> int:
    if value == "partial":
        return 0
    if value == "none":
        return 1
    if value == "full":
        return 2
    raise ValueError(f"Unsupported partner visibility: {value}")


@triton.jit
def _action_delta(action: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    row_delta = tl.where(action == 0, -1, tl.where(action == 2, 1, 0))
    col_delta = tl.where(action == 1, 1, tl.where(action == 3, -1, 0))
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
    TASK_ID: tl.constexpr,
    VISIBILITY_ID: tl.constexpr,
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

    chaser_action_value = tl.load(chaser_action + batch, mask=mask, other=0)
    chaser_row_delta, chaser_col_delta = _action_delta(chaser_action_value)
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
    explorer_row_delta, explorer_col_delta = _action_delta(explorer_action_value)
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
    if VISIBILITY_ID == 1:
        visible = active & False
    elif VISIBILITY_ID == 2:
        visible = active | True
    else:
        visible = tl.maximum(abs_row, abs_col) <= VISION_RADIUS

    chaser_reward = tl.full((BLOCK_SIZE,), -0.1, tl.float32)
    explorer_reward = tl.full((BLOCK_SIZE,), -0.5, tl.float32)
    chaser_reward = tl.where(chaser_new_field, 0.1, chaser_reward)
    explorer_reward = tl.where(explorer_new_field, 1.0, explorer_reward)
    if TASK_ID == 0:
        chaser_reward = tl.where(collision, 1.0, chaser_reward)
        explorer_reward = tl.where(collision, -1.0, explorer_reward)
    else:
        chaser_reward = tl.where(collision, -0.1, chaser_reward)
        explorer_reward = tl.where(collision, -0.5, explorer_reward)
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

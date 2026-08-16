"""Verified CUDA/Triton implementation of the chase-grid transition kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from experiments.chase_grid.environment import (
    ACTION_DELTAS,
    CHASER,
    EXPLORER,
    ChaseGridEnvironment,
)
from mouse_run_run.core.environment import EnvTransition
from mouse_run_run.core.types import AgentId


def triton_step(
    environment: ChaseGridEnvironment,
    actions: dict[AgentId, torch.Tensor],
) -> EnvTransition:
    """Advance one transition with no Torch fallback."""
    if environment.device.type != "cuda":
        raise RuntimeError("the Triton chase-grid backend requires CUDA")
    batch_size = environment.batch_size
    outputs = _outputs(batch_size, environment.device)
    reward = environment.config.reward
    if reward.task == "social":
        chaser_collision_reward = reward.social_chaser_collision
        explorer_collision_reward = reward.social_explorer_collision
    else:
        chaser_collision_reward = reward.non_social_chaser_collision
        explorer_collision_reward = reward.non_social_explorer_collision
    deltas = tuple(value for pair in ACTION_DELTAS for value in pair)
    _step_kernel[(1,)](
        environment.chaser_position,
        environment.explorer_position,
        environment.chaser_visited,
        environment.explorer_visited,
        environment.done,
        environment.step_count,
        actions[CHASER],
        actions[EXPLORER],
        outputs["chaser_reward"],
        outputs["explorer_reward"],
        outputs["truncated"],
        outputs["active"],
        outputs["collision"],
        outputs["chaser_collision"],
        outputs["explorer_collision"],
        outputs["chaser_new_field"],
        outputs["explorer_new_field"],
        outputs["chaser_approach"],
        outputs["explorer_escape"],
        outputs["explorer_escape_close"],
        outputs["explorer_escape_near"],
        outputs["explorer_escape_far"],
        outputs["chaser_partner_visible"],
        outputs["explorer_partner_visible"],
        outputs["distance"],
        batch_size,
        BLOCK_SIZE=triton.next_power_of_2(batch_size),
        GRID_SIZE=environment.config.grid_size,
        MAX_STEPS=environment.config.max_steps,
        VISION_RADIUS=environment.config.vision_radius,
        UP_ROW=deltas[0],
        UP_COL=deltas[1],
        RIGHT_ROW=deltas[2],
        RIGHT_COL=deltas[3],
        DOWN_ROW=deltas[4],
        DOWN_COL=deltas[5],
        LEFT_ROW=deltas[6],
        LEFT_COL=deltas[7],
        CHASER_STEP_REWARD=reward.chaser_step,
        EXPLORER_STEP_REWARD=reward.explorer_step,
        CHASER_NEW_REWARD=reward.chaser_new_field,
        EXPLORER_NEW_REWARD=reward.explorer_new_field,
        CHASER_COLLISION_REWARD=chaser_collision_reward,
        EXPLORER_COLLISION_REWARD=explorer_collision_reward,
    )
    events = {
        key: outputs[key]
        for key in (
            "collision",
            "chaser_collision",
            "explorer_collision",
            "chaser_new_field",
            "explorer_new_field",
            "chaser_approach",
            "explorer_escape",
            "explorer_escape_close",
            "explorer_escape_near",
            "explorer_escape_far",
            "chaser_partner_visible",
            "explorer_partner_visible",
        )
    }
    return EnvTransition(
        observations=environment._observations(),
        rewards={CHASER: outputs["chaser_reward"], EXPLORER: outputs["explorer_reward"]},
        terminated=torch.zeros_like(environment.done),
        truncated=outputs["truncated"],
        active=outputs["active"],
        world=environment._world(),
        events=events,
    )


def _outputs(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    boolean = (
        "truncated",
        "active",
        "collision",
        "chaser_collision",
        "explorer_collision",
        "chaser_new_field",
        "explorer_new_field",
        "chaser_approach",
        "explorer_escape",
        "explorer_escape_close",
        "explorer_escape_near",
        "explorer_escape_far",
        "chaser_partner_visible",
        "explorer_partner_visible",
    )
    outputs = {
        key: torch.empty(batch_size, dtype=torch.bool, device=device) for key in boolean
    }
    for key in ("chaser_reward", "explorer_reward", "distance"):
        outputs[key] = torch.empty(batch_size, dtype=torch.float32, device=device)
    return outputs


@triton.jit
def _action_delta(
    action,
    UP_ROW: tl.constexpr,
    UP_COL: tl.constexpr,
    RIGHT_ROW: tl.constexpr,
    RIGHT_COL: tl.constexpr,
    DOWN_ROW: tl.constexpr,
    DOWN_COL: tl.constexpr,
    LEFT_ROW: tl.constexpr,
    LEFT_COL: tl.constexpr,
):
    row = tl.where(
        action == 0,
        UP_ROW,
        tl.where(action == 1, RIGHT_ROW, tl.where(action == 2, DOWN_ROW, LEFT_ROW)),
    )
    col = tl.where(
        action == 0,
        UP_COL,
        tl.where(action == 1, RIGHT_COL, tl.where(action == 2, DOWN_COL, LEFT_COL)),
    )
    return row, col


@triton.jit
def _step_kernel(
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
    truncated_out,
    active_out,
    collision_out,
    chaser_collision_out,
    explorer_collision_out,
    chaser_new_out,
    explorer_new_out,
    chaser_approach_out,
    explorer_escape_out,
    explorer_escape_close_out,
    explorer_escape_near_out,
    explorer_escape_far_out,
    chaser_visible_out,
    explorer_visible_out,
    distance_out,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    VISION_RADIUS: tl.constexpr,
    UP_ROW: tl.constexpr,
    UP_COL: tl.constexpr,
    RIGHT_ROW: tl.constexpr,
    RIGHT_COL: tl.constexpr,
    DOWN_ROW: tl.constexpr,
    DOWN_COL: tl.constexpr,
    LEFT_ROW: tl.constexpr,
    LEFT_COL: tl.constexpr,
    CHASER_STEP_REWARD: tl.constexpr,
    EXPLORER_STEP_REWARD: tl.constexpr,
    CHASER_NEW_REWARD: tl.constexpr,
    EXPLORER_NEW_REWARD: tl.constexpr,
    CHASER_COLLISION_REWARD: tl.constexpr,
    EXPLORER_COLLISION_REWARD: tl.constexpr,
):
    batch = tl.arange(0, BLOCK_SIZE)
    mask = batch < batch_size
    position_offset = batch * 2
    visited_offset = batch * GRID_SIZE * GRID_SIZE
    old_cr = tl.load(chaser_position + position_offset, mask=mask, other=0)
    old_cc = tl.load(chaser_position + position_offset + 1, mask=mask, other=0)
    old_er = tl.load(explorer_position + position_offset, mask=mask, other=0)
    old_ec = tl.load(explorer_position + position_offset + 1, mask=mask, other=0)
    old_done = tl.load(done_state + batch, mask=mask, other=1).to(tl.int1)
    active = ~old_done

    chaser_action_value = tl.load(chaser_action + batch, mask=mask, other=0)
    cdr, cdc = _action_delta(
        chaser_action_value,
        UP_ROW,
        UP_COL,
        RIGHT_ROW,
        RIGHT_COL,
        DOWN_ROW,
        DOWN_COL,
        LEFT_ROW,
        LEFT_COL,
    )
    candidate_cr = tl.minimum(tl.maximum(old_cr + cdr, 0), GRID_SIZE - 1)
    candidate_cc = tl.minimum(tl.maximum(old_cc + cdc, 0), GRID_SIZE - 1)
    chaser_collision = active & (candidate_cr == old_er) & (candidate_cc == old_ec)
    next_cr = tl.where(active, tl.where(chaser_collision, old_cr, candidate_cr), old_cr)
    next_cc = tl.where(active, tl.where(chaser_collision, old_cc, candidate_cc), old_cc)

    explorer_action_value = tl.load(explorer_action + batch, mask=mask, other=0)
    edr, edc = _action_delta(
        explorer_action_value,
        UP_ROW,
        UP_COL,
        RIGHT_ROW,
        RIGHT_COL,
        DOWN_ROW,
        DOWN_COL,
        LEFT_ROW,
        LEFT_COL,
    )
    candidate_er = tl.minimum(tl.maximum(old_er + edr, 0), GRID_SIZE - 1)
    candidate_ec = tl.minimum(tl.maximum(old_ec + edc, 0), GRID_SIZE - 1)
    explorer_collision = active & (candidate_er == next_cr) & (candidate_ec == next_cc)
    next_er = tl.where(active, tl.where(explorer_collision, old_er, candidate_er), old_er)
    next_ec = tl.where(active, tl.where(explorer_collision, old_ec, candidate_ec), old_ec)
    tl.store(chaser_position + position_offset, next_cr, mask=mask)
    tl.store(chaser_position + position_offset + 1, next_cc, mask=mask)
    tl.store(explorer_position + position_offset, next_er, mask=mask)
    tl.store(explorer_position + position_offset + 1, next_ec, mask=mask)

    chaser_visit = visited_offset + next_cr * GRID_SIZE + next_cc
    explorer_visit = visited_offset + next_er * GRID_SIZE + next_ec
    chaser_seen = tl.load(chaser_visited + chaser_visit, mask=mask, other=1).to(tl.int1)
    explorer_seen = tl.load(explorer_visited + explorer_visit, mask=mask, other=1).to(tl.int1)
    chaser_new = active & ~chaser_collision & ~chaser_seen
    explorer_new = active & ~explorer_collision & ~explorer_seen
    tl.store(chaser_visited + chaser_visit, True, mask=mask & active & ~chaser_collision)
    tl.store(explorer_visited + explorer_visit, True, mask=mask & active & ~explorer_collision)

    old_dr = (old_cr - old_er).to(tl.float32)
    old_dc = (old_cc - old_ec).to(tl.float32)
    old_distance = tl.sqrt(old_dr * old_dr + old_dc * old_dc)
    chaser_dr = (next_cr - old_er).to(tl.float32)
    chaser_dc = (next_cc - old_ec).to(tl.float32)
    chaser_distance = tl.sqrt(chaser_dr * chaser_dr + chaser_dc * chaser_dc)
    explorer_dr = (next_er - old_cr).to(tl.float32)
    explorer_dc = (next_ec - old_cc).to(tl.float32)
    explorer_distance = tl.sqrt(explorer_dr * explorer_dr + explorer_dc * explorer_dc)
    chaser_approach = (
        active & ~chaser_collision & ~chaser_new & (chaser_distance < old_distance)
    )
    explorer_escape = (
        active & ~explorer_collision & ~explorer_new & (explorer_distance > old_distance)
    )
    collision = chaser_collision | explorer_collision

    old_step = tl.load(step_count + batch, mask=mask, other=0)
    new_step = old_step + active.to(tl.int64)
    truncated = old_done | (new_step >= MAX_STEPS)
    tl.store(step_count + batch, new_step, mask=mask)
    tl.store(done_state + batch, truncated, mask=mask)

    final_dr = (next_cr - next_er).to(tl.float32)
    final_dc = (next_cc - next_ec).to(tl.float32)
    distance = tl.sqrt(final_dr * final_dr + final_dc * final_dc)
    visible = tl.maximum(tl.abs(next_cr - next_er), tl.abs(next_cc - next_ec)) <= VISION_RADIUS
    chaser_reward = tl.full((BLOCK_SIZE,), CHASER_STEP_REWARD, tl.float32)
    explorer_reward = tl.full((BLOCK_SIZE,), EXPLORER_STEP_REWARD, tl.float32)
    chaser_reward = tl.where(chaser_new, CHASER_NEW_REWARD, chaser_reward)
    explorer_reward = tl.where(explorer_new, EXPLORER_NEW_REWARD, explorer_reward)
    chaser_reward = tl.where(collision, CHASER_COLLISION_REWARD, chaser_reward)
    explorer_reward = tl.where(collision, EXPLORER_COLLISION_REWARD, explorer_reward)

    tl.store(chaser_reward_out + batch, chaser_reward * active.to(tl.float32), mask=mask)
    tl.store(explorer_reward_out + batch, explorer_reward * active.to(tl.float32), mask=mask)
    tl.store(truncated_out + batch, truncated, mask=mask)
    tl.store(active_out + batch, active, mask=mask)
    tl.store(collision_out + batch, collision, mask=mask)
    tl.store(chaser_collision_out + batch, chaser_collision, mask=mask)
    tl.store(explorer_collision_out + batch, explorer_collision, mask=mask)
    tl.store(chaser_new_out + batch, chaser_new, mask=mask)
    tl.store(explorer_new_out + batch, explorer_new, mask=mask)
    tl.store(chaser_approach_out + batch, chaser_approach, mask=mask)
    tl.store(explorer_escape_out + batch, explorer_escape, mask=mask)
    tl.store(explorer_escape_close_out + batch, explorer_escape & (explorer_distance < 3.0), mask=mask)
    tl.store(
        explorer_escape_near_out + batch,
        explorer_escape & (explorer_distance >= 3.0) & (explorer_distance < 5.0),
        mask=mask,
    )
    tl.store(explorer_escape_far_out + batch, explorer_escape & (explorer_distance >= 5.0), mask=mask)
    tl.store(chaser_visible_out + batch, visible, mask=mask)
    tl.store(explorer_visible_out + batch, visible, mask=mask)
    tl.store(distance_out + batch, distance, mask=mask)

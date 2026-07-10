import pytest
import torch

from mouse_run_run.degenerate import episode_degeneracy
from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig


def _env(batch_size: int, **config_kwargs) -> BatchedChaseEnv:
    torch.manual_seed(0)
    config = GridWorldConfig(grid_size=7, **config_kwargs)
    env = BatchedChaseEnv(config, batch_size=batch_size, device=torch.device("cpu"))
    env.reset_state()
    return env


def _walk_positions(steps: int, hold_steps: set[int]) -> torch.Tensor:
    """Row-0 walk toggling between columns 0 and 1, stationary exactly at hold_steps."""
    cols = [0]
    for t in range(steps):
        cols.append(cols[-1] if t in hold_steps else 1 - cols[-1])
    rows = torch.zeros(steps + 1, dtype=torch.long)
    return torch.stack([rows, torch.tensor(cols, dtype=torch.long)], dim=-1)


def test_approach_and_escape_use_distance_to_old_partner_position() -> None:
    env = _env(3)
    # Mark everything visited so new-field precedence cannot suppress events.
    env.chaser_visited.fill_(True)
    env.explorer_visited.fill_(True)
    # ep0: chaser (3,0)->(3,1) closes on old explorer (3,4): 4 -> 3; the
    #      explorer simultaneously moves (3,4)->(3,5), so measuring against
    #      the NEW explorer would give 4 (not < 4) — this pins the
    #      old-position semantics. Likewise explorer escape measures its new
    #      position against the OLD chaser (3,0): 5 > 4 (new chaser would
    #      give 4, not > 4).
    # ep1: chaser retreats (2 -> 3), explorer closes in (2 -> 1): no events.
    # ep2: both wall-bump in place; unchanged distance is not an approach or
    #      escape (strict inequality).
    env.chaser_position = torch.tensor([[3, 0], [3, 2], [3, 0]])
    env.explorer_position = torch.tensor([[3, 4], [3, 4], [3, 6]])

    result = env.step(
        torch.tensor([1, 3, 3]),  # right, left, left (ep2 clamps at col 0)
        torch.tensor([1, 3, 1]),  # right, left, right (ep2 clamps at col 6)
    )

    assert result.chaser_approach.tolist() == [True, False, False]
    assert result.explorer_escape.tolist() == [True, False, False]
    assert result.chaser_new_field.tolist() == [False, False, False]
    assert result.explorer_new_field.tolist() == [False, False, False]
    assert result.chaser_position.tolist() == [[3, 1], [3, 1], [3, 0]]
    assert result.explorer_position.tolist() == [[3, 5], [3, 3], [3, 6]]
    assert result.distance.tolist() == pytest.approx([4.0, 2.0, 6.0])


def test_escape_distance_band_boundaries_are_exact() -> None:
    env = _env(3)
    env.chaser_visited.fill_(True)
    env.explorer_visited.fill_(True)
    # Chaser stays put (wall bump at col 0); explorer steps right, ending at
    # euclidean distance 2 / 3 / 5 from the old chaser position. Bands are
    # close < 3.0 <= near < 5.0 <= far, so the boundary distances 3.0 and 5.0
    # must land in near and far respectively.
    env.chaser_position = torch.tensor([[0, 0], [0, 0], [0, 0]])
    env.explorer_position = torch.tensor([[0, 1], [0, 2], [0, 4]])

    result = env.step(
        torch.tensor([3, 3, 3]),
        torch.tensor([1, 1, 1]),
    )

    assert result.explorer_escape.tolist() == [True, True, True]
    assert result.explorer_escape_close.tolist() == [True, False, False]
    assert result.explorer_escape_near.tolist() == [False, True, False]
    assert result.explorer_escape_far.tolist() == [False, False, True]
    assert result.chaser_approach.tolist() == [False, False, False]


def test_new_field_takes_precedence_over_approach_and_escape() -> None:
    env = _env(1)
    env.chaser_visited.zero_()
    env.explorer_visited.zero_()
    env.chaser_visited[0, 3, 0] = True
    env.explorer_visited[0, 3, 4] = True
    env.chaser_position = torch.tensor([[3, 0]])
    env.explorer_position = torch.tensor([[3, 4]])

    result = env.step(torch.tensor([1]), torch.tensor([1]))

    # Chaser closed the distance (4 -> 3) and the explorer opened it (4 -> 5),
    # but both stepped onto unvisited tiles: new-field wins.
    assert result.chaser_new_field.tolist() == [True]
    assert result.explorer_new_field.tolist() == [True]
    assert result.chaser_approach.tolist() == [False]
    assert result.explorer_escape.tolist() == [False]
    assert result.explorer_escape_close.tolist() == [False]
    assert result.explorer_escape_near.tolist() == [False]
    assert result.explorer_escape_far.tolist() == [False]
    assert result.chaser_reward.tolist() == pytest.approx([0.1])
    assert result.explorer_reward.tolist() == pytest.approx([1.0])
    assert bool(env.chaser_visited[0, 3, 1])
    assert bool(env.explorer_visited[0, 3, 5])


def test_chaser_moves_first_and_explorer_resolves_against_updated_chaser() -> None:
    env = _env(3, task="social")
    env.chaser_visited.fill_(True)
    env.explorer_visited.fill_(True)
    # ep0: explorer steps into the tile the chaser just vacated: no collision.
    # ep1: explorer steps into the chaser's NEW tile: explorer collision.
    # ep2: chaser is blocked by the explorer's OLD tile even though the
    #      explorer departs on the same step: chaser collision.
    env.chaser_position = torch.tensor([[2, 2], [2, 2], [2, 2]])
    env.explorer_position = torch.tensor([[2, 1], [2, 4], [2, 3]])

    result = env.step(
        torch.tensor([1, 1, 1]),
        torch.tensor([1, 3, 1]),
    )

    assert result.chaser_collision.tolist() == [False, False, True]
    assert result.explorer_collision.tolist() == [False, True, False]
    assert result.collision.tolist() == [False, True, True]
    assert result.chaser_position.tolist() == [[2, 3], [2, 3], [2, 2]]
    assert result.explorer_position.tolist() == [[2, 2], [2, 4], [2, 4]]
    # The combined collision flag drives rewards for both agents, so in ep2
    # the explorer is penalised even though its own move succeeded.
    assert result.chaser_reward.tolist() == pytest.approx([-0.1, 1.0, 1.0])
    assert result.explorer_reward.tolist() == pytest.approx([-0.5, -1.0, -1.0])


def test_done_masks_movement_rewards_and_events() -> None:
    env = _env(1, max_steps=2)
    env.chaser_visited.fill_(True)
    env.explorer_visited.fill_(True)
    env.chaser_position = torch.tensor([[0, 0]])
    env.explorer_position = torch.tensor([[0, 6]])

    chaser_right = torch.tensor([1])
    explorer_left = torch.tensor([3])

    first = env.step(chaser_right, explorer_left)
    assert first.done.tolist() == [False]
    assert first.active.tolist() == [True]
    assert first.chaser_approach.tolist() == [True]

    # The terminating step still reports rewards and events; done flips on
    # the same result that carries them.
    second = env.step(chaser_right, explorer_left)
    assert second.done.tolist() == [True]
    assert second.active.tolist() == [True]
    assert second.chaser_approach.tolist() == [True]
    assert second.chaser_reward.tolist() == pytest.approx([-0.1])
    assert second.explorer_reward.tolist() == pytest.approx([-0.5])
    assert second.chaser_position.tolist() == [[0, 2]]
    assert second.explorer_position.tolist() == [[0, 4]]

    # After termination the same actions must be inert: no movement, zero
    # rewards, no events, and step_count frozen.
    third = env.step(chaser_right, explorer_left)
    assert third.done.tolist() == [True]
    assert third.active.tolist() == [False]
    assert third.chaser_position.tolist() == [[0, 2]]
    assert third.explorer_position.tolist() == [[0, 4]]
    assert third.chaser_reward.tolist() == pytest.approx([0.0])
    assert third.explorer_reward.tolist() == pytest.approx([0.0])
    assert third.chaser_approach.tolist() == [False]
    assert third.explorer_escape.tolist() == [False]
    assert third.collision.tolist() == [False]
    assert third.chaser_new_field.tolist() == [False]
    assert third.explorer_new_field.tolist() == [False]
    assert env.step_count.tolist() == [2]


def test_degeneracy_default_threshold_flags_runs_longer_than_one_step() -> None:
    steps = 100
    chaser = torch.stack(
        [_walk_positions(steps, {10}), _walk_positions(steps, {10, 11})],
        dim=1,
    )
    explorer = torch.stack(
        [_walk_positions(steps, set()), _walk_positions(steps, set())],
        dim=1,
    )

    result = episode_degeneracy(chaser, explorer)

    # threshold = 100 * 0.01 = 1.0 and the comparison is strict, so a
    # one-step stall is allowed but a two-step stall is degenerate.
    assert result["threshold_steps"].tolist() == pytest.approx([1.0, 1.0])
    assert result["chaser_stuck_run_steps"].tolist() == [1, 2]
    assert result["explorer_stuck_run_steps"].tolist() == [0, 0]
    assert result["same_state_run_steps"].tolist() == [0, 0]
    assert result["degenerate"].tolist() == [False, True]


def test_degeneracy_threshold_boundary_is_strict_with_custom_fraction() -> None:
    steps = 10
    chaser = torch.stack(
        [_walk_positions(steps, {4, 5}), _walk_positions(steps, {4, 5, 6})],
        dim=1,
    )
    explorer = torch.stack(
        [_walk_positions(steps, set()), _walk_positions(steps, set())],
        dim=1,
    )

    result = episode_degeneracy(chaser, explorer, threshold_fraction=0.2)

    # threshold = 10 * 0.2 = 2.0: a run of exactly 2 sits on the boundary
    # and must not flag; 3 must.
    assert result["threshold_steps"].tolist() == pytest.approx([2.0, 2.0])
    assert result["chaser_stuck_run_steps"].tolist() == [2, 3]
    assert result["degenerate"].tolist() == [False, True]


def test_collision_steps_are_excluded_from_stuck_runs() -> None:
    steps = 10
    chaser = _walk_positions(steps, {2, 3, 4, 5})[:, None, :]
    explorer = _walk_positions(steps, set())[:, None, :]
    colliding = torch.zeros((steps, 1), dtype=torch.bool)
    colliding[2:6, 0] = True

    without = episode_degeneracy(chaser, explorer, threshold_fraction=0.2)
    assert without["chaser_stuck_run_steps"].tolist() == [4]
    assert without["degenerate"].tolist() == [True]

    excluded = episode_degeneracy(
        chaser,
        explorer,
        chaser_collisions=colliding,
        threshold_fraction=0.2,
    )
    assert excluded["chaser_stuck_run_steps"].tolist() == [0]
    assert excluded["degenerate"].tolist() == [False]


def test_collision_exclusion_is_per_agent_and_splits_runs() -> None:
    steps = 10
    chaser = _walk_positions(steps, {2, 3, 4, 5})[:, None, :]
    explorer = _walk_positions(steps, set())[:, None, :]

    # Collisions attributed to the other agent do not excuse the chaser.
    wrong_agent = torch.zeros((steps, 1), dtype=torch.bool)
    wrong_agent[2:6, 0] = True
    result = episode_degeneracy(
        chaser,
        explorer,
        explorer_collisions=wrong_agent,
        threshold_fraction=0.2,
    )
    assert result["chaser_stuck_run_steps"].tolist() == [4]
    assert result["degenerate"].tolist() == [True]

    # A single colliding step splits the run into 2 + 1, and the longest
    # remaining run (2) sits exactly on the threshold: not degenerate.
    middle_only = torch.zeros((steps, 1), dtype=torch.bool)
    middle_only[4, 0] = True
    split = episode_degeneracy(
        chaser,
        explorer,
        chaser_collisions=middle_only,
        threshold_fraction=0.2,
    )
    assert split["chaser_stuck_run_steps"].tolist() == [2]
    assert split["degenerate"].tolist() == [False]


def test_invalid_actions_do_not_count_as_stuck() -> None:
    steps = 10
    chaser = _walk_positions(steps, {2, 3, 4})[:, None, :]
    explorer = _walk_positions(steps, set())[:, None, :]

    valid = torch.zeros((steps, 1), dtype=torch.long)
    result = episode_degeneracy(chaser, explorer, valid, threshold_fraction=0.2)
    assert result["chaser_stuck_run_steps"].tolist() == [3]
    assert result["degenerate"].tolist() == [True]

    padded = valid.clone()
    padded[2:5, 0] = -1
    masked = episode_degeneracy(chaser, explorer, padded, threshold_fraction=0.2)
    assert masked["chaser_stuck_run_steps"].tolist() == [0]
    assert masked["degenerate"].tolist() == [False]


def test_same_state_run_requires_both_agents_stuck_on_the_same_step() -> None:
    steps = 10
    # 2D single-episode inputs exercise the (steps + 1, 2) expansion path.
    chaser = _walk_positions(steps, {2, 3})
    explorer = _walk_positions(steps, {3, 4})

    result = episode_degeneracy(chaser, explorer, threshold_fraction=0.2)

    assert result["chaser_stuck_run_steps"].tolist() == [2]
    assert result["explorer_stuck_run_steps"].tolist() == [2]
    # Only step 3 has both agents stuck simultaneously.
    assert result["same_state_run_steps"].tolist() == [1]
    assert result["degenerate"].tolist() == [False]

    lower = episode_degeneracy(chaser, explorer, threshold_fraction=0.05)
    assert lower["threshold_steps"].tolist() == pytest.approx([0.5])
    assert lower["degenerate"].tolist() == [True]

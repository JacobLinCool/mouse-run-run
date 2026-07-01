from __future__ import annotations

import torch


def episode_degeneracy(
    chaser_positions: torch.Tensor,
    explorer_positions: torch.Tensor,
    chaser_actions: torch.Tensor | None = None,
    explorer_actions: torch.Tensor | None = None,
    *,
    chaser_collisions: torch.Tensor | None = None,
    explorer_collisions: torch.Tensor | None = None,
    threshold_fraction: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Return per-episode degeneracy masks and stuck-run statistics.

    Positions must be transition-aligned as ``(steps + 1, episodes, 2)`` so
    every action can be compared against the state before and after it.

    Interpretation of the paper rule ("agents stop moving in the same state or
    repeat movements within the same tile for more than 1% of episode
    length"):

    - A step counts as *stuck* for an agent when it issued a valid action but
      stayed in the same tile for a non-social reason (wall bump or explicit
      stationarity). Collision-blocked steps are social behavior — the
      rewarded chase outcome — and are excluded via the per-agent collision
      masks; without this exclusion every successful social episode would be
      flagged degenerate.
    - "For more than 1% of episode length" is read as a sustained duration:
      the longest consecutive stuck run must exceed the threshold, so
      scattered one-off wall bumps do not accumulate into an exclusion.
    """
    if threshold_fraction < 0.0:
        raise ValueError("threshold_fraction must be non-negative")
    chaser_positions = _as_time_batch_positions(chaser_positions, "chaser_positions")
    explorer_positions = _as_time_batch_positions(explorer_positions, "explorer_positions")
    if chaser_positions.shape != explorer_positions.shape:
        raise ValueError("chaser_positions and explorer_positions must have the same shape")
    if chaser_positions.shape[0] < 2:
        raise ValueError("positions must include at least one transition")

    steps = chaser_positions.shape[0] - 1
    episodes = chaser_positions.shape[1]
    device = chaser_positions.device
    chaser_actions = _as_time_batch_actions(chaser_actions, steps, episodes, device, "chaser_actions")
    explorer_actions = _as_time_batch_actions(
        explorer_actions,
        steps,
        episodes,
        device,
        "explorer_actions",
    )
    chaser_collisions = _as_time_batch_mask(
        chaser_collisions,
        steps,
        episodes,
        device,
        "chaser_collisions",
    )
    explorer_collisions = _as_time_batch_mask(
        explorer_collisions,
        steps,
        episodes,
        device,
        "explorer_collisions",
    )

    chaser_stationary = (chaser_positions[1:] == chaser_positions[:-1]).all(dim=-1)
    explorer_stationary = (explorer_positions[1:] == explorer_positions[:-1]).all(dim=-1)
    valid_chaser_action = chaser_actions >= 0
    valid_explorer_action = explorer_actions >= 0
    chaser_stuck = chaser_stationary & valid_chaser_action & ~chaser_collisions
    explorer_stuck = explorer_stationary & valid_explorer_action & ~explorer_collisions
    same_state_stuck = chaser_stuck & explorer_stuck

    same_state_run_steps = _longest_run(same_state_stuck)
    chaser_stuck_run_steps = _longest_run(chaser_stuck)
    explorer_stuck_run_steps = _longest_run(explorer_stuck)
    threshold_steps = torch.full(
        (episodes,),
        float(steps) * threshold_fraction,
        dtype=torch.float32,
        device=device,
    )
    degenerate = (
        (same_state_run_steps.float() > threshold_steps)
        | (chaser_stuck_run_steps.float() > threshold_steps)
        | (explorer_stuck_run_steps.float() > threshold_steps)
    )
    return {
        "degenerate": degenerate,
        "same_state_run_steps": same_state_run_steps,
        "chaser_stuck_run_steps": chaser_stuck_run_steps,
        "explorer_stuck_run_steps": explorer_stuck_run_steps,
        "threshold_steps": threshold_steps,
    }


def _longest_run(mask: torch.Tensor) -> torch.Tensor:
    """Length of the longest consecutive True run along dim 0."""
    counts = mask.long().cumsum(dim=0)
    resets = torch.where(mask, torch.zeros_like(counts), counts)
    running_reset = torch.cummax(resets, dim=0).values
    return (counts - running_reset).max(dim=0).values


def _as_time_batch_positions(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (steps + 1, episodes, 2)")
    return value


def _as_time_batch_actions(
    value: torch.Tensor | None,
    steps: int,
    episodes: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((steps, episodes), dtype=torch.long, device=device)
    if value.ndim == 1:
        value = value[:, None]
    if value.shape != (steps, episodes):
        raise ValueError(f"{name} must have shape ({steps}, {episodes})")
    return value.to(device=device)


def _as_time_batch_mask(
    value: torch.Tensor | None,
    steps: int,
    episodes: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((steps, episodes), dtype=torch.bool, device=device)
    if value.ndim == 1:
        value = value[:, None]
    if value.shape != (steps, episodes):
        raise ValueError(f"{name} must have shape ({steps}, {episodes})")
    return value.to(device=device, dtype=torch.bool)

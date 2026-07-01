from __future__ import annotations

import torch


def episode_degeneracy(
    chaser_positions: torch.Tensor,
    explorer_positions: torch.Tensor,
    chaser_actions: torch.Tensor | None = None,
    explorer_actions: torch.Tensor | None = None,
    *,
    threshold_fraction: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Return per-episode degeneracy masks and counts.

    Positions must be transition-aligned as ``(steps + 1, episodes, 2)`` so
    every action can be compared against the state before and after it.
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

    chaser_stationary = (chaser_positions[1:] == chaser_positions[:-1]).all(dim=-1)
    explorer_stationary = (explorer_positions[1:] == explorer_positions[:-1]).all(dim=-1)
    same_state = chaser_stationary & explorer_stationary
    valid_chaser_action = chaser_actions >= 0
    valid_explorer_action = explorer_actions >= 0

    same_state_steps = same_state.sum(dim=0)
    chaser_stationary_steps = (chaser_stationary & valid_chaser_action).sum(dim=0)
    explorer_stationary_steps = (explorer_stationary & valid_explorer_action).sum(dim=0)
    threshold_steps = torch.full(
        (episodes,),
        float(steps) * threshold_fraction,
        dtype=torch.float32,
        device=device,
    )
    degenerate = (
        (same_state_steps.float() > threshold_steps)
        | (chaser_stationary_steps.float() > threshold_steps)
        | (explorer_stationary_steps.float() > threshold_steps)
    )
    return {
        "degenerate": degenerate,
        "same_state_steps": same_state_steps,
        "chaser_stationary_steps": chaser_stationary_steps,
        "explorer_stationary_steps": explorer_stationary_steps,
        "threshold_steps": threshold_steps,
    }


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

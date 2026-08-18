"""Chaser movement geometry relative to a partner inside the field of view.

Grid positions are stored as ``(row, column)``. Figures use screen orientation,
so a movement or offset is reported as ``x = column`` and ``y = -row``: moving
up the grid points up in the plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mouse_run_run.core.simulation import TrajectoryBatch


CHASER_POSITION = "chaser_position"
EXPLORER_POSITION = "explorer_position"
PARTNER_VISIBLE = "chaser_partner_visible"


@dataclass(frozen=True)
class MovementSamples:
    """One row per step where the chaser moved with the partner in view."""

    offset: torch.Tensor
    """Chaser position minus partner position, as ``(x, y)``."""

    movement: torch.Tensor
    """Chaser displacement over the step, as ``(x, y)``."""

    to_partner: torch.Tensor
    """Unit vector from chaser to partner, as ``(x, y)``."""

    angle: torch.Tensor
    """Signed degrees from the direction to the partner to the movement."""

    def __len__(self) -> int:
        return int(self.angle.numel())


def movement_samples(trajectory: TrajectoryBatch) -> MovementSamples:
    """Collect chaser movement steps taken while the partner was in the FOV."""

    for key in (CHASER_POSITION, EXPLORER_POSITION, PARTNER_VISIBLE):
        if key not in trajectory.world:
            raise ValueError(f"trajectory world lacks {key!r}; collect with record_world")
    chaser = trajectory.world[CHASER_POSITION].double()
    partner = trajectory.world[EXPLORER_POSITION].double()
    steps = trajectory.active.shape[0]
    if chaser.shape[0] != steps + 1:
        raise ValueError("world positions must cover the reset state and every step")
    start_chaser = chaser[:steps]
    start_partner = partner[:steps]
    movement = _screen(chaser[1:] - start_chaser)
    offset = _screen(start_chaser - start_partner)
    to_partner = _screen(start_partner - start_chaser)
    moved = movement.abs().sum(dim=-1) > 0
    # World signals hold the state a step started from, so visibility is read
    # there rather than from the post-step event.
    visible = trajectory.world[PARTNER_VISIBLE][:steps].bool()
    keep = trajectory.active & visible & moved
    movement = movement[keep]
    to_partner = to_partner[keep]
    return MovementSamples(
        offset=offset[keep],
        movement=movement,
        to_partner=_normalize(to_partner),
        angle=_signed_angle(to_partner, movement),
    )


def flow_field_rows(
    samples: MovementSamples,
    *,
    vision_radius: int,
    minimum_samples: int = 1,
) -> list[dict[str, Any]]:
    """Mean chaser movement per relative position, one row per cell."""

    rows: list[dict[str, Any]] = []
    for x in range(-vision_radius, vision_radius + 1):
        for y in range(-vision_radius, vision_radius + 1):
            cell = (samples.offset[:, 0] == x) & (samples.offset[:, 1] == y)
            count = int(cell.sum())
            mean = (
                samples.movement[cell].mean(dim=0)
                if count >= max(minimum_samples, 1)
                else torch.zeros(2, dtype=torch.float64)
            )
            rows.append(
                {
                    "offset_x": x,
                    "offset_y": y,
                    "movement_x": float(mean[0]),
                    "movement_y": float(mean[1]),
                    "samples": count,
                }
            )
    return rows


def polar_rows(samples: MovementSamples, *, bins: int = 12) -> list[dict[str, Any]]:
    """Probability of the signed movement angle, over equal angular bins."""

    if bins < 1:
        raise ValueError("polar bins must be positive")
    width = 360.0 / bins
    wrapped = torch.remainder(samples.angle + width / 2, 360.0)
    index = torch.clamp((wrapped / width).long(), max=bins - 1)
    counts = torch.bincount(index, minlength=bins).double()
    total = float(counts.sum().clamp_min(1))
    return [
        {
            "bin_center": float(bin_index * width),
            "bin_width": width,
            "probability": float(counts[bin_index] / total),
            "samples": int(counts[bin_index]),
        }
        for bin_index in range(bins)
    ]


def angle_histogram_rows(
    samples: MovementSamples,
    *,
    bin_width: float = 30.0,
) -> list[dict[str, Any]]:
    """Probability of the unsigned movement-to-partner angle, in degree bins."""

    if bin_width <= 0 or 180.0 % bin_width != 0:
        raise ValueError("bin_width must divide 180 degrees")
    bins = int(180.0 / bin_width)
    magnitude = samples.angle.abs()
    index = torch.clamp((magnitude / bin_width).long(), max=bins - 1)
    counts = torch.bincount(index, minlength=bins).double()
    total = float(counts.sum().clamp_min(1))
    return [
        {
            "bin_start": float(bin_index * bin_width),
            "bin_end": float((bin_index + 1) * bin_width),
            "label": f"{int(bin_index * bin_width) + 1}-{int((bin_index + 1) * bin_width)}",
            "probability": float(counts[bin_index] / total),
            "samples": int(counts[bin_index]),
        }
        for bin_index in range(bins)
    ]


def _screen(delta: torch.Tensor) -> torch.Tensor:
    return torch.stack([delta[..., 1], -delta[..., 0]], dim=-1)


def _normalize(vectors: torch.Tensor) -> torch.Tensor:
    norm = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors / norm


def _signed_angle(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross = reference[..., 0] * target[..., 1] - reference[..., 1] * target[..., 0]
    dot = (reference * target).sum(dim=-1)
    return torch.rad2deg(torch.atan2(cross, dot))

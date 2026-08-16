"""Convert one saved episode into a finite JSON replay payload."""

from __future__ import annotations

import math
from typing import Any

import torch

from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.experiment import Experiment


def build_replay_payload(
    artifact: RolloutArtifact,
    experiment: Experiment,
    *,
    episode: int,
) -> dict[str, Any]:
    if artifact.manifest["experiment"] != experiment.name:
        raise ValueError(
            f"rollout experiment {artifact.manifest['experiment']!r} != {experiment.name!r}"
        )
    if experiment.renderer is None:
        raise ValueError(f"experiment {experiment.name!r} has no replay renderer")
    if not 0 <= episode < artifact.trajectory.batch_size:
        raise ValueError(f"episode {episode} is outside the rollout")
    trajectory = artifact.trajectory
    frames = []
    for step in range(trajectory.horizon + 1):
        rendered = experiment.renderer.frame(
            trajectory.world,
            batch_index=episode,
            step=step,
        )
        transition: dict[str, Any] | None = None
        if step < trajectory.horizon:
            transition = {
                "active": bool(trajectory.active[step, episode]),
                "terminated": bool(trajectory.terminated[step, episode]),
                "truncated": bool(trajectory.truncated[step, episode]),
                "agents": {
                    str(agent_id): {
                        "action": int(trajectory.agents[agent_id].actions[step, episode]),
                        "reward": float(trajectory.agents[agent_id].rewards[step, episode]),
                        "value": float(trajectory.agents[agent_id].values[step, episode]),
                    }
                    for agent_id in trajectory.agent_ids
                },
                "events": {
                    key: _scalar(value[step, episode])
                    for key, value in trajectory.events.items()
                    if value.ndim == 2
                },
            }
        frames.append({"world": rendered, "transition": transition})

    neural: dict[str, Any] = {}
    for agent_id in trajectory.agent_ids:
        activation_payload = {}
        for site, values in trajectory.agents[agent_id].activations.items():
            episode_values = values[:, episode].float()
            if episode_values.ndim != 2:
                continue
            activation_payload[site] = {
                "values": _finite_tensor(episode_values),
                "projection": _finite_tensor(_pca2(episode_values)),
                "width": episode_values.shape[1],
            }
        neural[str(agent_id)] = activation_payload

    summary = artifact.episodes.slice(episode, 1).to_pylist()[0]
    return {
        "format": "mrr-replay-payload-v2",
        "rollout": str(artifact.path),
        "experiment": experiment.name,
        "episode": episode,
        "episodes": trajectory.batch_size,
        "horizon": trajectory.horizon,
        "agent_ids": [str(agent_id) for agent_id in trajectory.agent_ids],
        "interventions": artifact.manifest["interventions"],
        "summary": summary,
        "frames": frames,
        "neural": neural,
    }


def _pca2(values: torch.Tensor) -> torch.Tensor:
    centered = values.double() - values.double().mean(dim=0)
    if centered.shape[0] < 2 or centered.shape[1] < 1:
        return torch.zeros(centered.shape[0], 2)
    _, _, right = torch.linalg.svd(centered, full_matrices=False)
    dimensions = min(2, right.shape[0])
    projected = centered @ right[:dimensions].T
    if dimensions == 1:
        projected = torch.cat([projected, torch.zeros_like(projected)], dim=1)
    return projected.float()


def _scalar(value: torch.Tensor) -> bool | int | float:
    item = value.item()
    if isinstance(item, bool):
        return item
    if isinstance(item, int):
        return item
    number = float(item)
    if not math.isfinite(number):
        raise ValueError("replay payload contains a non-finite scalar")
    return number


def _finite_tensor(value: torch.Tensor) -> list[Any]:
    if not torch.isfinite(value).all():
        raise ValueError("replay payload contains non-finite neural activity")
    return value.detach().cpu().tolist()

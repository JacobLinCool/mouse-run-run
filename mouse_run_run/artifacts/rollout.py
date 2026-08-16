"""Dense safetensors trajectories paired with episode-level Parquet tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from mouse_run_run.core.simulation import AgentTrajectory, TrajectoryBatch
from mouse_run_run.core.types import AgentId


ROLLOUT_FORMAT = "mrr-rollout-v2"
ROLLOUT_TENSOR_FORMAT = "mrr-rollout-tensors-v2"


@dataclass(frozen=True)
class RolloutArtifact:
    path: Path
    manifest: dict[str, Any]
    trajectory: TrajectoryBatch
    episodes: pa.Table


def save_rollout(
    path: Path,
    *,
    experiment: str,
    checkpoint: Path,
    trajectory: TrajectoryBatch,
    config: Mapping[str, Any],
    interventions: tuple[Mapping[str, Any], ...] = (),
) -> RolloutArtifact:
    if path.exists():
        raise FileExistsError(f"rollout output already exists: {path}")
    _validate_trajectory(trajectory)
    path.mkdir(parents=True)
    tensors = _flatten_trajectory(trajectory)
    save_file(
        tensors,
        str(path / "tensors.safetensors"),
        metadata={"format": ROLLOUT_TENSOR_FORMAT},
    )
    episodes = _episode_table(trajectory, interventions)
    schema_metadata = dict(episodes.schema.metadata or {})
    schema_metadata[b"mrr_format"] = ROLLOUT_FORMAT.encode()
    episodes = episodes.replace_schema_metadata(schema_metadata)
    pq.write_table(episodes, path / "episodes.parquet", compression="zstd")
    manifest = {
        "format": ROLLOUT_FORMAT,
        "experiment": experiment,
        "checkpoint": str(checkpoint),
        "agent_ids": [str(agent_id) for agent_id in trajectory.agent_ids],
        "episodes": trajectory.batch_size,
        "horizon": trajectory.horizon,
        "seed": trajectory.seed,
        "config": dict(config),
        "interventions": [dict(item) for item in interventions],
        "alignment": {
            "agent_series": "[time, episode, ...] at state s_t",
            "transition_series": "[time, episode, ...] for s_t -> s_(t+1)",
            "world_series": "[time+1, episode, ...] for states s_0..s_T",
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stored_trajectory = _unflatten_trajectory(manifest, tensors)
    _validate_trajectory(stored_trajectory)
    return RolloutArtifact(
        path=path,
        manifest=manifest,
        trajectory=stored_trajectory,
        episodes=episodes,
    )


def load_rollout(path: Path) -> RolloutArtifact:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != ROLLOUT_FORMAT:
        raise ValueError(
            f"{path} is not a {ROLLOUT_FORMAT} artifact; legacy rollouts are archived only"
        )
    tensor_path = path / "tensors.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        tensor_metadata = dict(handle.metadata() or {})
    if tensor_metadata.get("format") != ROLLOUT_TENSOR_FORMAT:
        raise ValueError("rollout tensor payload format is invalid")
    tensors = load_file(str(tensor_path), device="cpu")
    trajectory = _unflatten_trajectory(manifest, tensors)
    episodes = pq.read_table(path / "episodes.parquet")
    if (episodes.schema.metadata or {}).get(b"mrr_format") != ROLLOUT_FORMAT.encode():
        raise ValueError("episode table format is invalid")
    if episodes.num_rows != trajectory.batch_size:
        raise ValueError("episode table row count does not match trajectory batch")
    _validate_trajectory(trajectory)
    return RolloutArtifact(path=path, manifest=manifest, trajectory=trajectory, episodes=episodes)


def concatenate_trajectories(chunks: list[TrajectoryBatch], *, seed: int) -> TrajectoryBatch:
    if not chunks:
        raise ValueError("cannot concatenate an empty trajectory list")
    first = chunks[0]
    for chunk in chunks[1:]:
        if chunk.agent_ids != first.agent_ids or chunk.horizon != first.horizon:
            raise ValueError("trajectory chunks have incompatible agents or horizons")
        if chunk.world.keys() != first.world.keys() or chunk.events.keys() != first.events.keys():
            raise ValueError("trajectory chunks have incompatible world/event schemas")
    agents: dict[AgentId, AgentTrajectory] = {}
    for agent_id in first.agent_ids:
        references = [chunk.agents[agent_id] for chunk in chunks]
        activation_keys = references[0].activations.keys()
        if any(item.activations.keys() != activation_keys for item in references[1:]):
            raise ValueError(f"activation schema changed across chunks for {agent_id}")
        agents[agent_id] = AgentTrajectory(
            observations=torch.cat([item.observations for item in references], dim=1),
            actions=torch.cat([item.actions for item in references], dim=1),
            rewards=torch.cat([item.rewards for item in references], dim=1),
            logits=torch.cat([item.logits for item in references], dim=1),
            values=torch.cat([item.values for item in references], dim=1),
            log_probs=torch.cat([item.log_probs for item in references], dim=1),
            bootstrap_value=torch.cat([item.bootstrap_value for item in references], dim=0),
            activations={
                key: torch.cat([item.activations[key] for item in references], dim=1)
                for key in activation_keys
            },
            state_inputs={
                key: torch.cat([item.state_inputs[key] for item in references], dim=1)
                for key in references[0].state_inputs
            },
        )
    result = TrajectoryBatch(
        agent_ids=first.agent_ids,
        agents=agents,
        terminated=torch.cat([chunk.terminated for chunk in chunks], dim=1),
        truncated=torch.cat([chunk.truncated for chunk in chunks], dim=1),
        active=torch.cat([chunk.active for chunk in chunks], dim=1),
        world={
            key: torch.cat([chunk.world[key] for chunk in chunks], dim=1)
            for key in first.world
        },
        events={
            key: torch.cat([chunk.events[key] for chunk in chunks], dim=1)
            for key in first.events
        },
        seed=seed,
        episode_seeds=torch.cat([chunk.episode_seeds for chunk in chunks]),
    )
    _validate_trajectory(result)
    return result


def trajectory_to_cpu(trajectory: TrajectoryBatch) -> TrajectoryBatch:
    def cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu().contiguous()

    return TrajectoryBatch(
        agent_ids=trajectory.agent_ids,
        agents={
            agent_id: AgentTrajectory(
                observations=cpu(agent.observations),
                actions=cpu(agent.actions),
                rewards=cpu(agent.rewards),
                logits=cpu(agent.logits),
                values=cpu(agent.values),
                log_probs=cpu(agent.log_probs),
                bootstrap_value=cpu(agent.bootstrap_value),
                activations={key: cpu(value) for key, value in agent.activations.items()},
                state_inputs={key: cpu(value) for key, value in agent.state_inputs.items()},
            )
            for agent_id, agent in trajectory.agents.items()
        },
        terminated=cpu(trajectory.terminated),
        truncated=cpu(trajectory.truncated),
        active=cpu(trajectory.active),
        world={key: cpu(value) for key, value in trajectory.world.items()},
        events={key: cpu(value) for key, value in trajectory.events.items()},
        seed=trajectory.seed,
        episode_seeds=cpu(trajectory.episode_seeds),
    )


def _flatten_trajectory(trajectory: TrajectoryBatch) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "transition.terminated": trajectory.terminated,
        "transition.truncated": trajectory.truncated,
        "transition.active": trajectory.active,
        "episode.seed": trajectory.episode_seeds,
    }
    for agent_id, agent in trajectory.agents.items():
        prefix = f"agent.{agent_id}."
        tensors.update(
            {
                f"{prefix}observations": agent.observations,
                f"{prefix}actions": agent.actions,
                f"{prefix}rewards": agent.rewards,
                f"{prefix}logits": agent.logits,
                f"{prefix}values": agent.values,
                f"{prefix}log_probs": agent.log_probs,
                f"{prefix}bootstrap_value": agent.bootstrap_value,
            }
        )
        for key, value in agent.activations.items():
            tensors[f"{prefix}activation.{key}"] = value
        for key, value in agent.state_inputs.items():
            tensors[f"{prefix}state_input.{key}"] = value
    tensors.update({f"world.{key}": value for key, value in trajectory.world.items()})
    tensors.update({f"event.{key}": value for key, value in trajectory.events.items()})
    return {key: value.detach().cpu().contiguous() for key, value in tensors.items()}


def _unflatten_trajectory(
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> TrajectoryBatch:
    agent_ids = tuple(AgentId(value) for value in manifest["agent_ids"])
    agents: dict[AgentId, AgentTrajectory] = {}
    for agent_id in agent_ids:
        prefix = f"agent.{agent_id}."
        activation_prefix = f"{prefix}activation."
        state_prefix = f"{prefix}state_input."
        agents[agent_id] = AgentTrajectory(
            observations=tensors[f"{prefix}observations"],
            actions=tensors[f"{prefix}actions"],
            rewards=tensors[f"{prefix}rewards"],
            logits=tensors[f"{prefix}logits"],
            values=tensors[f"{prefix}values"],
            log_probs=tensors[f"{prefix}log_probs"],
            bootstrap_value=tensors[f"{prefix}bootstrap_value"],
            activations={
                key.removeprefix(activation_prefix): value
                for key, value in tensors.items()
                if key.startswith(activation_prefix)
            },
            state_inputs={
                key.removeprefix(state_prefix): value
                for key, value in tensors.items()
                if key.startswith(state_prefix)
            },
        )
    recognized = {
        "transition.terminated",
        "transition.truncated",
        "transition.active",
        "episode.seed",
    }
    for agent_id in agent_ids:
        prefix = f"agent.{agent_id}."
        recognized.update(
            {
                f"{prefix}observations",
                f"{prefix}actions",
                f"{prefix}rewards",
                f"{prefix}logits",
                f"{prefix}values",
                f"{prefix}log_probs",
                f"{prefix}bootstrap_value",
            }
        )
        recognized.update(key for key in tensors if key.startswith(f"{prefix}activation."))
        recognized.update(key for key in tensors if key.startswith(f"{prefix}state_input."))
    recognized.update(key for key in tensors if key.startswith("world."))
    recognized.update(key for key in tensors if key.startswith("event."))
    unknown = tensors.keys() - recognized
    if unknown:
        raise ValueError(f"rollout contains unknown tensors: {sorted(unknown)!r}")
    return TrajectoryBatch(
        agent_ids=agent_ids,
        agents=agents,
        terminated=tensors["transition.terminated"],
        truncated=tensors["transition.truncated"],
        active=tensors["transition.active"],
        world={
            key.removeprefix("world."): value
            for key, value in tensors.items()
            if key.startswith("world.")
        },
        events={
            key.removeprefix("event."): value
            for key, value in tensors.items()
            if key.startswith("event.")
        },
        seed=int(manifest["seed"]),
        episode_seeds=tensors["episode.seed"],
    )


def _episode_table(
    trajectory: TrajectoryBatch,
    interventions: tuple[Mapping[str, Any], ...],
) -> pa.Table:
    rows: dict[str, list[Any]] = {
        "episode_index": list(range(trajectory.batch_size)),
        "seed": trajectory.episode_seeds.tolist(),
        "length": trajectory.active.long().sum(dim=0).tolist(),
        "terminated": trajectory.terminated.any(dim=0).tolist(),
        "truncated": trajectory.truncated.any(dim=0).tolist(),
        "interventions": [
            ",".join(str(item.get("name", "unnamed")) for item in interventions)
        ]
        * trajectory.batch_size,
    }
    for agent_id, agent in trajectory.agents.items():
        rows[f"return.{agent_id}"] = agent.rewards.sum(dim=0).tolist()
    for key, value in trajectory.events.items():
        if value.ndim == 2:
            rows[f"event.{key}"] = value.to(torch.float64).sum(dim=0).tolist()
            if value.dtype == torch.bool:
                active_count = trajectory.active.sum(dim=0).clamp_min(1)
                rows[f"event_fraction.{key}"] = (
                    (value & trajectory.active).sum(dim=0) / active_count
                ).tolist()
    if "distance" in trajectory.world:
        distance = trajectory.world["distance"][:-1]
        active_count = trajectory.active.sum(dim=0).clamp_min(1)
        rows["world.distance_mean"] = (
            (distance * trajectory.active).sum(dim=0) / active_count
        ).tolist()
        rows["world.distance_final"] = trajectory.world["distance"][-1].tolist()
    stuck_runs = _longest_noncollision_stuck_runs(trajectory)
    rows["quality.longest_noncollision_stuck_run"] = stuck_runs.tolist()
    rows["quality.degenerate"] = (stuck_runs > 5).tolist()
    return pa.table(rows)


def _longest_noncollision_stuck_runs(trajectory: TrajectoryBatch) -> torch.Tensor:
    position_keys = [
        key for key in ("chaser_position", "explorer_position") if key in trajectory.world
    ]
    if len(position_keys) != 2 or "collision" not in trajectory.events:
        return torch.zeros(trajectory.batch_size, dtype=torch.long)
    unchanged = torch.ones_like(trajectory.active)
    for key in position_keys:
        values = trajectory.world[key]
        unchanged &= (values[1:] == values[:-1]).all(dim=-1)
    stuck = unchanged & ~trajectory.events["collision"].bool() & trajectory.active
    longest = torch.zeros(trajectory.batch_size, dtype=torch.long, device=stuck.device)
    current = torch.zeros_like(longest)
    for step in range(trajectory.horizon):
        current = torch.where(stuck[step], current + 1, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    return longest.cpu()


def _validate_trajectory(trajectory: TrajectoryBatch) -> None:
    time, batch = trajectory.terminated.shape
    if time < 1 or batch < 1:
        raise ValueError("trajectory must have positive time and episode axes")
    if tuple(trajectory.agents) != trajectory.agent_ids:
        raise ValueError("trajectory agent mapping is not ordered like agent_ids")
    if trajectory.episode_seeds.shape != (batch,) or trajectory.episode_seeds.dtype != torch.long:
        raise ValueError(f"episode_seeds must be int64 shape {(batch,)}")
    for name, value in (
        ("terminated", trajectory.terminated),
        ("truncated", trajectory.truncated),
        ("active", trajectory.active),
    ):
        if value.shape != (time, batch) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be bool shape {(time, batch)}")
    for agent_id, agent in trajectory.agents.items():
        for name, value in (
            ("observations", agent.observations),
            ("actions", agent.actions),
            ("rewards", agent.rewards),
            ("logits", agent.logits),
            ("values", agent.values),
            ("log_probs", agent.log_probs),
        ):
            if value.shape[:2] != (time, batch):
                raise ValueError(
                    f"{agent_id}.{name} leading shape {value.shape[:2]} != {(time, batch)}"
                )
        if agent.bootstrap_value.shape != (batch,):
            raise ValueError(f"{agent_id}.bootstrap_value must have shape {(batch,)}")
        for key, value in agent.activations.items():
            if value.shape[:2] != (time, batch):
                raise ValueError(f"{agent_id} activation {key!r} is misaligned")
        for key, value in agent.state_inputs.items():
            if value.shape[:2] != (time, batch):
                raise ValueError(f"{agent_id} state input {key!r} is misaligned")
        for name, value in (
            ("observations", agent.observations),
            ("rewards", agent.rewards),
            ("logits", agent.logits),
            ("values", agent.values),
            ("log_probs", agent.log_probs),
            *tuple((f"activation.{key}", value) for key, value in agent.activations.items()),
            *tuple((f"state_input.{key}", value) for key, value in agent.state_inputs.items()),
        ):
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f"{agent_id}.{name} contains non-finite values")
    for key, value in trajectory.world.items():
        if value.shape[:2] != (time + 1, batch):
            raise ValueError(f"world {key!r} must have leading shape {(time + 1, batch)}")
    for key, value in trajectory.events.items():
        if value.shape[:2] != (time, batch):
            raise ValueError(f"event {key!r} must have leading shape {(time, batch)}")

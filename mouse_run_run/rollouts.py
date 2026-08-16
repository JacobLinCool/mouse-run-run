"""Checkpoint-backed rollout collection through the shared simulation engine."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from mouse_run_run.artifacts.checkpoint import load_checkpoint
from mouse_run_run.artifacts.rollout import (
    RolloutArtifact,
    concatenate_trajectories,
    save_rollout,
    trajectory_to_cpu,
)
from mouse_run_run.core.experiment import Experiment, PolicyBuildContext, RuntimeConfig
from mouse_run_run.core.intervention import InterventionPipeline
from mouse_run_run.core.policy import PolicyModule
from mouse_run_run.core.types import AgentId
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, TerminalProgress
from mouse_run_run.core.simulation import SimulationConfig, SimulationEngine
from mouse_run_run.training.runner import select_device


def collect_experiment_rollout(
    experiment: Experiment,
    checkpoint: Path,
    output: Path,
    *,
    runtime: RuntimeConfig,
    episodes: int,
    horizon: int,
    deterministic: bool,
    interventions: InterventionPipeline | None = None,
    action_seed: int | None = None,
    progress: ProgressSink | None = None,
) -> RolloutArtifact:
    if episodes < 1 or horizon < 1:
        raise ValueError("rollout episodes and horizon must be positive")
    experiment.validate()
    runtime.validate()
    device = select_device(runtime.device)
    policies = {
        agent_id: experiment.policy_factories[agent_id](
            PolicyBuildContext(
                agent_id,
                experiment.observation_shapes[agent_id],
                experiment.action_counts[agent_id],
                device,
            )
        )
        for agent_id in experiment.agent_ids
    }
    metadata = load_checkpoint(checkpoint, policies=policies)
    if metadata.experiment != experiment.name:
        raise ValueError(
            f"checkpoint experiment {metadata.experiment!r} != {experiment.name!r}"
        )
    return collect_policy_rollout(
        experiment,
        policies,
        checkpoint,
        output,
        runtime=runtime,
        episodes=episodes,
        horizon=horizon,
        deterministic=deterministic,
        interventions=interventions,
        action_seed=action_seed,
        progress=progress,
    )


def collect_policy_rollout(
    experiment: Experiment,
    policies: dict[AgentId, PolicyModule],
    checkpoint: Path,
    output: Path,
    *,
    runtime: RuntimeConfig,
    episodes: int,
    horizon: int,
    deterministic: bool,
    interventions: InterventionPipeline | None = None,
    action_seed: int | None = None,
    progress: ProgressSink | None = None,
) -> RolloutArtifact:
    if tuple(policies) != experiment.agent_ids:
        raise ValueError("rollout policy keys must match experiment agents")
    device = select_device(runtime.device)
    tracker = ProgressTracker("rollout", episodes, progress or TerminalProgress())
    chunks = []
    completed = 0
    chunk_index = 0
    while completed < episodes:
        current = min(runtime.batch_size, episodes - completed)
        chunk_runtime = replace(runtime, batch_size=current)
        environment = experiment.make_environment(chunk_runtime, device)
        chunk = SimulationEngine(environment, policies).collect(
            SimulationConfig(
                horizon=horizon,
                deterministic=deterministic,
                seed=runtime.seed + chunk_index,
                action_seed=(
                    None if action_seed is None else action_seed + chunk_index
                ),
                record_activations=True,
                record_world=True,
            ),
            interventions=interventions,
        )
        chunks.append(trajectory_to_cpu(chunk))
        completed += current
        chunk_index += 1
        tracker.emit(completed)
    trajectory = concatenate_trajectories(chunks, seed=runtime.seed)
    intervention_manifest = tuple(
        {
            "name": item.name,
            "agents": sorted(str(agent_id) for agent_id in item.agent_ids),
            "site": item.site,
            "target": item.target,
        }
        for item in (interventions.interventions if interventions is not None else ())
    )
    artifact = save_rollout(
        output,
        experiment=experiment.name,
        checkpoint=checkpoint,
        trajectory=trajectory,
        config={
            "runtime": asdict(runtime),
            "episodes": episodes,
            "horizon": horizon,
            "deterministic": deterministic,
            "action_seed": action_seed,
        },
        interventions=intervention_manifest,
    )
    tracker.emit(episodes, state="completed")
    return artifact

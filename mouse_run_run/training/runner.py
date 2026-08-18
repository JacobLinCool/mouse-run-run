"""Experiment composition, training progress, checkpoints, and resume."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from mouse_run_run.artifacts.checkpoint import load_checkpoint, save_checkpoint
from mouse_run_run.core.experiment import Experiment, PolicyBuildContext, RuntimeConfig
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, StatusFileProgress, TerminalProgress
from mouse_run_run.core.simulation import (
    SimulationConfig,
    SimulationEngine,
    TrajectoryBatch,
    episode_behavior_summary,
)
from mouse_run_run.core.types import AgentId
from mouse_run_run.training.ppo import IndependentPPO


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    checkpoint: Path
    completed_updates: int
    metrics: dict[str, float]


def train_experiment(
    experiment: Experiment,
    *,
    runtime: RuntimeConfig,
    run_dir: Path,
    resume_from: Path | None = None,
    progress: ProgressSink | None = None,
) -> TrainingResult:
    experiment.validate()
    runtime.validate()
    device = select_device(runtime.device)
    torch.manual_seed(runtime.seed)
    policies = _build_policies(experiment, device)
    learner = IndependentPPO(policies, experiment.training.ppo, seed=runtime.seed + 10_000_019)
    start_update = 0
    if resume_from is not None:
        metadata = load_checkpoint(
            resume_from,
            policies=policies,
            optimizers=learner.optimizers,
            learner=learner,
            restore_rng=True,
        )
        if metadata.experiment != experiment.name:
            raise ValueError(
                f"checkpoint experiment {metadata.experiment!r} != {experiment.name!r}"
            )
        start_update = metadata.update
        _validate_resume_compatibility(metadata.config, experiment, runtime)
        if start_update >= experiment.training.updates:
            raise ValueError("resume checkpoint already reached requested updates")

    resolved_config = {
        "format": "mrr-run-v2",
        "experiment": experiment.name,
        "experiment_metadata": dict(experiment.metadata),
        "runtime": asdict(runtime),
        "training": asdict(experiment.training),
    }
    _prepare_run_directory(run_dir, resume_from=resume_from)
    (run_dir / "config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sink = StatusFileProgress(
        run_dir / "status.json",
        progress or TerminalProgress(),
    )
    tracker = ProgressTracker("train", experiment.training.updates, sink)
    latest = run_dir / "checkpoints" / "latest.safetensors"
    metrics: dict[str, float] = {}
    metrics_path = run_dir / "metrics.jsonl"
    writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=start_update + 1)
    writer.add_text("config/json", json.dumps(resolved_config, indent=2), start_update)
    completed = start_update
    try:
        for update in range(start_update + 1, experiment.training.updates + 1):
            environment = experiment.make_environment(runtime, device)
            rollout = SimulationEngine(environment, policies).collect(
                SimulationConfig(
                    horizon=experiment.training.horizon,
                    deterministic=False,
                    seed=runtime.seed + update - 1,
                    action_seed=runtime.seed + 1_000_000 + update - 1,
                    record_world=True,
                )
            )
            behavior_metrics = _behavior_metrics(rollout)
            ppo_metrics = learner.update(rollout)
            metrics = {**ppo_metrics.flat(), **behavior_metrics}
            _append_metric(metrics_path, {"update": update, **metrics})
            for key, value in metrics.items():
                writer.add_scalar(key, value, update)
            writer.flush()
            completed = update
            if update % experiment.training.progress_every == 0:
                tracker.emit(update, metrics)
            if (
                experiment.training.checkpoint_every
                and update % experiment.training.checkpoint_every == 0
            ):
                numbered = run_dir / "checkpoints" / f"update_{update:06d}.safetensors"
                save_checkpoint(
                    numbered,
                    experiment=experiment.name,
                    update=update,
                    config=resolved_config,
                    metrics=metrics,
                    policies=policies,
                    optimizers=learner.optimizers,
                    learner=learner,
                )
        save_checkpoint(
            latest,
            experiment=experiment.name,
            update=experiment.training.updates,
            config=resolved_config,
            metrics=metrics,
            policies=policies,
            optimizers=learner.optimizers,
            learner=learner,
        )
        tracker.emit(experiment.training.updates, metrics, state="completed")
    except Exception:
        tracker.emit(completed, metrics, state="failed")
        raise
    finally:
        writer.close()
    return TrainingResult(
        run_dir=run_dir,
        checkpoint=latest,
        completed_updates=experiment.training.updates,
        metrics=metrics,
    )


def _behavior_metrics(rollout: TrajectoryBatch) -> dict[str, float]:
    """Batch means of the same behaviour columns the rollout tables report."""

    return {
        key: float(column.to(torch.float64).mean())
        for key, column in episode_behavior_summary(rollout).items()
    }


def _build_policies(
    experiment: Experiment,
    device: torch.device,
) -> dict[AgentId, Any]:
    policies = {
        agent_id: experiment.policy_factories[agent_id](
            PolicyBuildContext(
                agent_id=agent_id,
                observation_shape=experiment.observation_shapes[agent_id],
                action_count=experiment.action_counts[agent_id],
                device=device,
            )
        )
        for agent_id in experiment.agent_ids
    }
    if len({id(policy) for policy in policies.values()}) != len(policies):
        raise ValueError("v2 does not support policy parameter sharing")
    return policies


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"unsupported device {name!r}")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _prepare_run_directory(run_dir: Path, *, resume_from: Path | None) -> None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return
    if not any(run_dir.iterdir()):
        return
    same_run_resume = (
        resume_from is not None
        and resume_from.resolve().is_relative_to(run_dir.resolve())
    )
    if not same_run_resume:
        raise FileExistsError(
            f"run directory is not empty: {run_dir}; choose a new directory or resume it"
        )


def _validate_resume_compatibility(
    checkpoint_config: dict[str, Any],
    experiment: Experiment,
    runtime: RuntimeConfig,
) -> None:
    previous_metadata = checkpoint_config.get("experiment_metadata")
    if not isinstance(previous_metadata, dict):
        raise ValueError("checkpoint has no resolved experiment metadata")
    previous_experiment_config = dict(previous_metadata.get("config") or {})
    current_experiment_config = json.loads(
        json.dumps(dict(experiment.metadata.get("config") or {}))
    )
    previous_experiment_config.pop("training", None)
    current_experiment_config.pop("training", None)
    if previous_experiment_config != current_experiment_config:
        raise ValueError("resume changes the environment or policy experiment config")
    previous_training = checkpoint_config.get("training")
    current_training = json.loads(json.dumps(asdict(experiment.training)))
    if not isinstance(previous_training, dict):
        raise ValueError("checkpoint has no resolved training config")
    for key in ("horizon", "ppo"):
        if previous_training.get(key) != current_training[key]:
            raise ValueError(f"resume changes training.{key}")
    previous_runtime = checkpoint_config.get("runtime")
    if not isinstance(previous_runtime, dict) or previous_runtime.get("seed") != runtime.seed:
        raise ValueError("resume must use the original runtime seed")

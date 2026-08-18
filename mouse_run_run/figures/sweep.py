"""Partner representation across training, collected from stored checkpoints.

The confirmatory study evaluates neural activity at one predeclared checkpoint.
Panels that track partner representation over training need the same measurement
at every checkpoint, which the training stage already wrote; this walks them and
records one row per unit and checkpoint.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from mouse_run_run.analyses.partner_variance import (
    PartnerVarianceConfig,
    behaviour_groups,
    neural_action_space,
    non_redundant_variance,
)
from mouse_run_run.core.experiment import RuntimeConfig
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, TerminalProgress
from mouse_run_run.figures.sources import discover_units, write_source_table
from mouse_run_run.loading import load_study
from mouse_run_run.rollouts import collect_policy_rollout
from mouse_run_run.studies.runner import _loaded_policies
from mouse_run_run.studies.types import StudyUnit


CHASER = "chaser"
EXPLORER = "explorer"
SWEEP_FORMAT = "mrr-partner-representation-v1"
VARIANCE_FORMAT = "mrr-partner-variance-v1"


@dataclass(frozen=True)
class SweepConfig:
    episodes: int = 10
    horizon: int = 500
    device: str = "cpu"
    backend: str = "torch"
    seed: int = 0
    variance: PartnerVarianceConfig = PartnerVarianceConfig(
        group_permutations=5, chance_shuffles=5
    )


def collect_partner_representation(
    study_output: Path,
    *,
    config: SweepConfig = SweepConfig(),
    progress: ProgressSink | None = None,
) -> Path:
    """One row per unit and checkpoint: partner representation and collisions."""

    manifest = json.loads((study_output / "study.json").read_text(encoding="utf-8"))
    definition = load_study(manifest["target"])
    units = discover_units(study_output)
    checkpoints = sorted(
        {
            int(path.stem.removeprefix("update_"))
            for unit in units
            for path in (unit.root / "train" / "checkpoints").glob("update_*.safetensors")
        }
    )
    if not checkpoints:
        raise FileNotFoundError(f"no numbered training checkpoints under {study_output}")
    tracker = ProgressTracker(
        "sweep:partner-representation",
        len(units) * len(checkpoints),
        progress or TerminalProgress(),
    )
    rows: list[dict[str, Any]] = []
    completed = 0
    for unit in units:
        study_unit = StudyUnit(condition=unit.condition, seed=unit.seed)
        experiment = definition.experiment(study_unit, horizon=config.horizon)
        for update in checkpoints:
            checkpoint = unit.root / "train" / "checkpoints" / f"update_{update:06d}.safetensors"
            rows.extend(
                _measure_checkpoint(
                    experiment,
                    checkpoint,
                    unit=unit,
                    update=update,
                    config=config,
                )
            )
            completed += 1
            tracker.emit(completed)
    tracker.emit(completed, state="completed")
    return write_source_table(
        study_output / "tables" / "partner_representation.parquet", rows, SWEEP_FORMAT
    )


def _measure_checkpoint(
    experiment: Any,
    checkpoint: Path,
    *,
    unit: Any,
    update: int,
    config: SweepConfig,
) -> list[dict[str, Any]]:
    policies = _loaded_policies(experiment, checkpoint, config.device)
    scratch = checkpoint.parent.parent.parent / "sweep_rollout"
    if scratch.exists():
        shutil.rmtree(scratch)
    artifact = collect_policy_rollout(
        experiment,
        policies,
        checkpoint,
        scratch,
        runtime=RuntimeConfig(
            device=config.device,
            environment_backend=config.backend,
            batch_size=config.episodes,
            seed=unit.seed * 1_000 + config.seed,
        ),
        episodes=config.episodes,
        horizon=config.horizon,
        deterministic=False,
        action_seed=unit.seed * 1_000 + config.seed + 1,
    )
    try:
        rows = _rows_for(artifact, policies, unit=unit, update=update, config=config)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return rows


def _rows_for(
    artifact: Any,
    policies: dict[str, Any],
    *,
    unit: Any,
    update: int,
    config: SweepConfig,
) -> list[dict[str, Any]]:
    trajectory = artifact.trajectory
    mask = trajectory.active.cpu()
    collisions = trajectory.events["collision"][mask].double()
    rows = []
    for agent_id, partner_id in ((CHASER, EXPLORER), (EXPLORER, CHASER)):
        hidden = trajectory.agents[agent_id].activations["hidden"][mask]
        readout = policies[agent_id].action_head.weight.detach().cpu()
        action_space = neural_action_space(hidden, readout)
        groups = behaviour_groups(
            artifact, agent_id=agent_id, partner_id=partner_id, mask=mask
        )
        contributions = non_redundant_variance(
            groups, action_space, config=config.variance
        )
        rows.append(
            {
                "condition": unit.condition,
                "seed": unit.seed,
                "checkpoint_update": update,
                "agent": agent_id,
                "partner_representation": 100.0 * contributions["partner"],
                "self_representation": 100.0 * contributions["self"],
                "full_model": 100.0 * contributions["full_model"],
                "collision_time_percent": 100.0 * float(collisions.mean()),
            }
        )
    return rows


def collect_partner_variance(
    study_output: Path,
    *,
    config: SweepConfig = SweepConfig(),
    progress: ProgressSink | None = None,
) -> Path:
    """Partner variance per unit in the full space, in PLSC1, and in action space.

    This reads the neural rollouts the study already stored, so it adds no
    simulation; only the regression is new.
    """

    from mouse_run_run.analyses.paper_plsc import load_paper_plsc
    from mouse_run_run.artifacts.rollout import load_rollout

    units = discover_units(study_output)
    tracker = ProgressTracker(
        "analyze:partner-variance", len(units), progress or TerminalProgress()
    )
    rows: list[dict[str, Any]] = []
    for completed, unit in enumerate(units, start=1):
        visibility = "partial" if unit.condition == "social" else "none"
        for checkpoint in sorted((unit.root / "neural").glob("update_*")):
            base = checkpoint / visibility
            if not (base / "rollout").is_dir():
                continue
            artifact = load_rollout(base / "rollout")
            fitted = load_paper_plsc(base / "plsc") if (base / "plsc").is_dir() else None
            update = int(checkpoint.name.removeprefix("update_"))
            rows.extend(
                _variance_rows(artifact, fitted, unit=unit, update=update, config=config)
            )
        tracker.emit(completed)
    tracker.emit(len(units), state="completed")
    return write_source_table(
        study_output / "tables" / "partner_variance.parquet", rows, VARIANCE_FORMAT
    )


def _variance_rows(
    artifact: Any,
    fitted: Any,
    *,
    unit: Any,
    update: int,
    config: SweepConfig,
) -> list[dict[str, Any]]:

    trajectory = artifact.trajectory
    mask = trajectory.active.cpu()
    rows = []
    for agent_id, partner_id in ((CHASER, EXPLORER), (EXPLORER, CHASER)):
        hidden = trajectory.agents[agent_id].activations["hidden"][mask].double()
        groups = behaviour_groups(
            artifact, agent_id=agent_id, partner_id=partner_id, mask=mask
        )
        spaces = {"full": hidden}
        if fitted is not None:
            standardized = (hidden - fitted.means[agent_id]) / fitted.scales[agent_id]
            spaces["plsc1"] = standardized @ fitted.bases[agent_id][:, :1]
        readout = _action_readout(unit.root, update, agent_id)
        if readout is not None:
            spaces["action"] = neural_action_space(hidden, readout)
        for space, response in spaces.items():
            contributions = non_redundant_variance(
                groups, response, config=config.variance
            )
            rows.append(
                {
                    "condition": unit.condition,
                    "seed": unit.seed,
                    "checkpoint_update": update,
                    "agent": agent_id,
                    "space": space,
                    "partner_variance": 100.0 * contributions["partner"],
                    "self_variance": 100.0 * contributions["self"],
                    "full_model": 100.0 * contributions["full_model"],
                }
            )
    return rows


def _action_readout(unit_root: Path, update: int, agent_id: str) -> "Any | None":
    """The action head this agent read its logits from, straight from the checkpoint."""

    from safetensors import safe_open

    checkpoint = unit_root / "train" / "checkpoints" / f"update_{update:06d}.safetensors"
    if not checkpoint.is_file():
        return None
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        key = f"policy.{agent_id}.action_head.weight"
        if key not in handle.keys():
            return None
        return handle.get_tensor(key)

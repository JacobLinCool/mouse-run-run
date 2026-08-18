"""Strict orchestration for typed train-to-causal reproduction studies."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file

from experiments.chase_grid.environment import CHASER, EXPLORER
from mouse_run_run.analyses.decoder import DecoderConfig, run_diagnostic_decoder
from mouse_run_run.analyses.paper_plsc import (
    PaperPLSCConfig,
    identical_action_mask,
    fit_paper_plsc,
    load_paper_plsc,
    removed_variance_fraction,
    save_paper_plsc,
    temporally_shifted_random_pc_basis,
)
from mouse_run_run.analyses.plsc import StandardizedSubspace
from mouse_run_run.artifacts.checkpoint import load_checkpoint
from mouse_run_run.artifacts.rollout import RolloutArtifact, load_rollout
from mouse_run_run.core.experiment import PolicyBuildContext, RuntimeConfig
from mouse_run_run.core.intervention import InterventionPipeline
from mouse_run_run.core.policy import PolicyModule
from mouse_run_run.interventions import SubspaceIntervention
from mouse_run_run.loading import load_study
from mouse_run_run.policies import UniformRandomPolicy
from mouse_run_run.rollouts import collect_policy_rollout
from mouse_run_run.studies.types import StudyDefinition, StudyStageName, StudyUnit
from mouse_run_run.studies.zhang_reference import NATURE_REFERENCES
from mouse_run_run.training.runner import select_device, train_experiment


STUDY_FORMAT = "mrr-study-v1"
StudyStage = Literal["train", "behavior", "neural", "causal", "all"]
STAGES = ("train", "behavior", "neural", "causal")


@dataclass(frozen=True)
class StudyRunResult:
    output: Path
    stages: tuple[str, ...]
    report: Path


def plan_study(definition: StudyDefinition) -> dict[str, Any]:
    definition.validate()
    checkpoint_count = definition.updates // definition.checkpoint_every
    social_units = len(definition.seeds)
    non_social_units = len(definition.seeds)
    behavior_rollouts = (
        len(definition.units) * len(definition.behavior.checkpoint_updates) * 2
        if "behavior" in definition.enabled_stages
        else 0
    )
    if "neural" in definition.enabled_stages:
        neural_checkpoints = len(definition.neural.checkpoint_updates)
        neural_rollouts = neural_checkpoints * (
            social_units
            + non_social_units
            * len(definition.neural.non_social_visibility_controls)
        )
        identical_action_plsc_fits = neural_checkpoints * social_units
        decoder_rollouts = neural_checkpoints * (social_units + non_social_units)
    else:
        neural_rollouts = 0
        identical_action_plsc_fits = 0
        decoder_rollouts = 0
    payload = {
        "format": "mrr-study-plan-v1",
        "name": definition.name,
        "source": definition.source,
        "purpose": definition.purpose,
        "emits_scientific_gates": definition.emits_scientific_gates,
        "enabled_stages": list(definition.enabled_stages),
        "conditions": list(definition.conditions),
        "seeds": list(definition.seeds),
        "units": len(definition.units),
        "unit_ids": [unit.unit_id for unit in definition.units],
        "updates_per_unit": definition.updates,
        "episodes_per_update": definition.episodes_per_update,
        "steps_per_episode": definition.steps_per_episode,
        "environment_steps_per_unit": definition.environment_steps_per_unit,
        "total_environment_steps": definition.total_environment_steps,
        "optimizer_minibatches_per_agent_update": (
            definition.optimizer_minibatches_per_agent_update
        ),
        "optimizer_steps_per_update": definition.optimizer_steps_per_update,
        "optimizer_steps_per_unit": definition.optimizer_steps_per_unit,
        "total_optimizer_steps": definition.total_optimizer_steps,
        "checkpoint_updates": list(
            range(
                definition.checkpoint_every,
                definition.updates + 1,
                definition.checkpoint_every,
            )
        ),
        "estimated_artifacts": {
            "training_checkpoints_including_latest": len(definition.units)
            * (checkpoint_count + 1),
            "random_opponent_rollouts": behavior_rollouts,
            "neural_rollouts": neural_rollouts,
            "paper_plsc_fits_including_identical_action_controls": (
                neural_rollouts + identical_action_plsc_fits
            ),
            "diagnostic_decoder_rollouts": decoder_rollouts,
            "causal_rollouts": (
                social_units * 4 if "causal" in definition.enabled_stages else 0
            ),
        },
        "recipes": {
            "behavior": {
                "enabled": "behavior" in definition.enabled_stages,
                **asdict(definition.behavior),
            },
            "neural": {
                "enabled": "neural" in definition.enabled_stages,
                **asdict(definition.neural),
            },
            "causal": {
                "enabled": "causal" in definition.enabled_stages,
                **asdict(definition.causal),
            },
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def run_study(
    target: str,
    definition: StudyDefinition,
    *,
    output: Path,
    device: str,
    backend: str,
    parallelism: int,
    stage: StudyStage,
    resume: bool,
) -> StudyRunResult:
    definition.validate()
    if parallelism < 1:
        raise ValueError("study parallelism must be positive")
    runtime_probe = RuntimeConfig(
        device=device,
        environment_backend=backend,
        batch_size=definition.episodes_per_update,
    )
    runtime_probe.validate()
    if backend == "triton" and device not in ("cuda", "auto"):
        raise ValueError("Triton study backend requires CUDA or auto device selection")
    selected: tuple[StudyStageName, ...]
    if stage == "all":
        selected = definition.enabled_stages
    else:
        if stage not in definition.enabled_stages:
            raise ValueError(
                f"study purpose {definition.purpose!r} does not enable stage {stage!r}"
            )
        selected = (stage,)
    _prepare_study(output, target, definition, resume=resume)
    for stage_name in selected:
        _write_status(output, stage_name, "running", 0, len(definition.units))
        results = _run_units(
            target,
            definition,
            output,
            stage_name,
            device=device,
            backend=backend,
            parallelism=parallelism,
            resume=resume,
        )
        failed = [result for result in results if result.get("state") == "failed"]
        if failed:
            _write_status(output, stage_name, "failed", len(results) - len(failed), len(results))
            raise RuntimeError(f"study {stage_name} failed for {failed!r}")
        _aggregate_stage(output, definition, stage_name)
        _write_status(output, stage_name, "completed", len(results), len(results))
    report = _write_report(output, definition)
    return StudyRunResult(output=output, stages=selected, report=report)


def _run_units(
    target: str,
    definition: StudyDefinition,
    output: Path,
    stage: str,
    *,
    device: str,
    backend: str,
    parallelism: int,
    resume: bool,
) -> list[dict[str, Any]]:
    arguments = [
        (target, unit.condition, unit.seed, str(output), stage, device, backend, resume)
        for unit in definition.units
    ]
    if parallelism == 1:
        results = []
        for argument in arguments:
            results.append(_study_unit_worker(argument))
            _write_status(output, stage, "running", len(results), len(arguments))
        return results
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=parallelism,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = [executor.submit(_study_unit_worker, argument) for argument in arguments]
        for future in as_completed(futures):
            results.append(future.result())
            _write_status(output, stage, "running", len(results), len(arguments))
    return results


def _study_unit_worker(
    argument: tuple[str, str, int, str, str, str, str, bool],
) -> dict[str, Any]:
    target, condition, seed, output, stage, device, backend, resume = argument
    definition = load_study(target)
    unit = next(
        unit
        for unit in definition.units
        if unit.condition == condition and unit.seed == seed
    )
    unit_root = _unit_root(Path(output), unit)
    _write_unit_status(unit_root, stage, "running")
    try:
        if stage == "train":
            _run_train_unit(definition, unit, unit_root, device, backend, resume)
        elif stage == "behavior":
            _run_behavior_unit(definition, unit, unit_root, device, backend, resume)
        elif stage == "neural":
            _run_neural_unit(definition, unit, unit_root, device, backend, resume)
        elif stage == "causal":
            if unit.condition == "social":
                _run_causal_unit(definition, unit, unit_root, device, backend, resume)
            else:
                _write_unit_status(unit_root, stage, "skipped")
                return {"unit": unit.unit_id, "stage": stage, "state": "skipped"}
        else:
            raise ValueError(f"unknown study stage {stage!r}")
        _write_unit_status(unit_root, stage, "completed")
        return {"unit": unit.unit_id, "stage": stage, "state": "completed"}
    except Exception as error:
        _write_unit_status(unit_root, stage, "failed", error=str(error))
        return {
            "unit": unit.unit_id,
            "stage": stage,
            "state": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def _run_train_unit(
    definition: StudyDefinition,
    unit: StudyUnit,
    unit_root: Path,
    device: str,
    backend: str,
    resume: bool,
) -> None:
    run_dir = unit_root / "train"
    completion = run_dir / "checkpoints" / "latest.safetensors"
    if completion.is_file():
        if resume:
            return
        raise FileExistsError(f"training unit already exists: {run_dir}")
    resume_from = _latest_training_checkpoint(run_dir) if resume else None
    if run_dir.exists() and any(run_dir.iterdir()) and resume_from is None:
        raise ValueError(f"training unit has no strict resumable checkpoint: {run_dir}")
    experiment = definition.experiment(unit)
    train_experiment(
        experiment,
        runtime=RuntimeConfig(
            device=device,
            environment_backend=backend,
            batch_size=definition.episodes_per_update,
            seed=unit.seed,
        ),
        run_dir=run_dir,
        resume_from=resume_from,
    )


def _run_behavior_unit(
    definition: StudyDefinition,
    unit: StudyUnit,
    unit_root: Path,
    device: str,
    backend: str,
    resume: bool,
) -> None:
    root = unit_root / "behavior"
    result_path = root / "results.parquet"
    if _completed_output(result_path, resume):
        return
    experiment = definition.experiment(unit, horizon=definition.behavior.horizon)
    rows: list[dict[str, Any]] = []
    matchups = {
        "trained_chaser_vs_uniform_explorer": EXPLORER,
        "trained_explorer_vs_uniform_chaser": CHASER,
    }
    for update in definition.behavior.checkpoint_updates:
        checkpoint = _required_checkpoint(unit_root, update)
        policies = _loaded_policies(experiment, checkpoint, device)
        for index, (name, random_agent) in enumerate(matchups.items()):
            rollout_path = root / f"update_{update:06d}" / name
            if rollout_path.exists():
                if not resume:
                    raise FileExistsError(f"behavior rollout exists: {rollout_path}")
                artifact = load_rollout(rollout_path)
            else:
                matchup_policies = dict(policies)
                matchup_policies[random_agent] = UniformRandomPolicy(
                    experiment.action_counts[random_agent]
                ).to(select_device(device))
                artifact = collect_policy_rollout(
                    experiment,
                    matchup_policies,
                    checkpoint,
                    rollout_path,
                    runtime=RuntimeConfig(
                        device=device,
                        environment_backend=backend,
                        batch_size=definition.behavior.batch_size,
                        # Keep evaluation streams paired across checkpoints.
                        seed=unit.seed * 10_000 + 1_000 + index,
                    ),
                    episodes=definition.behavior.episodes,
                    horizon=definition.behavior.horizon,
                    deterministic=False,
                    action_seed=unit.seed * 10_000 + 2_000 + index,
                )
            rows.extend(
                _annotated_episode_rows(
                    artifact,
                    unit,
                    matchup=name,
                    checkpoint=update,
                )
            )
    _write_table(result_path, rows, "mrr-zhang-2025-behavior-v1")


def _run_neural_unit(
    definition: StudyDefinition,
    unit: StudyUnit,
    unit_root: Path,
    device: str,
    backend: str,
    resume: bool,
) -> None:
    root = unit_root / "neural"
    result_path = root / "results.parquet"
    if _completed_output(result_path, resume):
        return
    rows: list[dict[str, Any]] = []
    visibilities = (
        ("partial",)
        if unit.condition == "social"
        else definition.neural.non_social_visibility_controls
    )
    for update in definition.neural.checkpoint_updates:
        checkpoint = _required_checkpoint(unit_root, update)
        for visibility_index, visibility in enumerate(visibilities):
            base = root / f"update_{update:06d}" / visibility
            experiment = definition.experiment(
                unit,
                horizon=definition.neural.horizon,
                visibility=visibility,
            )
            rollout_path = base / "rollout"
            if rollout_path.exists():
                if not resume:
                    raise FileExistsError(f"neural rollout exists: {rollout_path}")
                rollout = load_rollout(rollout_path)
            else:
                policies = _loaded_policies(experiment, checkpoint, device)
                rollout = collect_policy_rollout(
                    experiment,
                    policies,
                    checkpoint,
                    rollout_path,
                    runtime=RuntimeConfig(
                        device=device,
                        environment_backend=backend,
                        batch_size=definition.neural.batch_size,
                        seed=unit.seed * 100_000 + update + visibility_index,
                    ),
                    episodes=definition.neural.episodes,
                    horizon=definition.neural.horizon,
                    deterministic=False,
                    action_seed=unit.seed * 100_000 + 50_000 + update + visibility_index,
                )
            plsc_config = PaperPLSCConfig(
                agent_a=CHASER,
                agent_b=EXPLORER,
                permutations=definition.neural.plsc.permutations,
                percentile=definition.neural.plsc.percentile,
                min_shift=definition.neural.plsc.min_shift,
                seed=unit.seed + update,
                null_batch_size=definition.neural.plsc.null_batch_size,
                compute_device=("cuda" if select_device(device).type == "cuda" else "cpu"),
            )
            plsc_path = base / "plsc"
            if plsc_path.exists():
                fitted = load_paper_plsc(plsc_path)
            else:
                fitted = fit_paper_plsc(rollout, plsc_config)
                save_paper_plsc(plsc_path, fitted, rollout=rollout.path)
            rows.append(_plsc_row(unit, update, visibility, "primary", fitted))
            if unit.condition == "social":
                rows.append(
                    _identical_action_control_row(
                        unit,
                        update,
                        visibility,
                        base,
                        rollout,
                        plsc_config,
                    )
                )
            primary_visibility = (
                "partial" if unit.condition == "social" else "none"
            )
            if visibility == primary_visibility:
                decoder_path = base / "decoder"
                if (decoder_path / "results.parquet").is_file():
                    pass
                elif decoder_path.exists():
                    raise ValueError(
                        f"incomplete decoder artifact cannot be resumed: {decoder_path}"
                    )
                else:
                    run_diagnostic_decoder(
                        rollout,
                        decoder_path,
                        config=DecoderConfig(
                            folds=definition.neural.decoder.folds,
                            shuffled_controls=(
                                definition.neural.decoder.shuffled_controls
                            ),
                            seed=unit.seed + update,
                        ),
                    )
    _write_table(result_path, rows, "mrr-zhang-2025-neural-v1")


def _identical_action_control_row(
    unit: StudyUnit,
    update: int,
    visibility: str,
    base: Path,
    rollout: RolloutArtifact,
    plsc_config: PaperPLSCConfig,
) -> dict[str, Any]:
    path = base / "plsc_identical_action_excluded"
    if path.exists() and (path / "manifest.json").is_file():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") == "mrr-plsc-insufficient-v1":
            return _insufficient_plsc_row(
                unit,
                update,
                visibility,
                "identical_action_excluded",
                manifest["reason"],
            )
        return _plsc_row(
            unit,
            update,
            visibility,
            "identical_action_excluded",
            load_paper_plsc(path),
        )
    try:
        control = fit_paper_plsc(
            rollout,
            plsc_config,
            sample_mask=identical_action_mask(rollout, CHASER, EXPLORER),
        )
        save_paper_plsc(path, control, rollout=rollout.path)
        return _plsc_row(
            unit,
            update,
            visibility,
            "identical_action_excluded",
            control,
        )
    except ValueError as error:
        path.mkdir(parents=True)
        manifest = {
            "format": "mrr-plsc-insufficient-v1",
            "reason": str(error),
        }
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return _insufficient_plsc_row(
            unit,
            update,
            visibility,
            "identical_action_excluded",
            str(error),
        )


def _run_causal_unit(
    definition: StudyDefinition,
    unit: StudyUnit,
    unit_root: Path,
    device: str,
    backend: str,
    resume: bool,
) -> None:
    root = unit_root / "causal"
    result_path = root / "results.parquet"
    if _completed_output(result_path, resume):
        return
    update = definition.neural.checkpoint_updates[-1]
    checkpoint = _required_checkpoint(unit_root, update)
    fit_root = unit_root / "neural" / f"update_{update:06d}" / "partial"
    fit_rollout = load_rollout(fit_root / "rollout")
    fitted = load_paper_plsc(fit_root / "plsc")
    shared = fitted.subspace(CHASER, rank=definition.causal.shared_rank)
    control_basis = temporally_shifted_random_pc_basis(
        fitted,
        fit_rollout,
        agent_id=CHASER,
        shared_rank=definition.causal.shared_rank,
        control_rank=definition.causal.random_control_rank,
        seed=unit.seed + 90_000,
    )
    random_control = StandardizedSubspace(
        agent_id=CHASER,
        site="hidden",
        mean=fitted.means[CHASER],
        scale=fitted.scales[CHASER],
        basis=control_basis,
    )
    hidden = fit_rollout.trajectory.agents[CHASER].activations["hidden"][
        fitted.sample_mask
    ]
    variance = {
        "shared_readout": removed_variance_fraction(hidden, shared),
        "shifted_random_pc_readout": removed_variance_fraction(hidden, random_control),
        "shared_recurrent": removed_variance_fraction(hidden, shared),
    }
    root.mkdir(parents=True, exist_ok=True)
    control_path = root / "control_basis.safetensors"
    if control_path.is_file():
        stored_control = load_file(str(control_path), device="cpu")
        if set(stored_control) != {"temporal_shift_random_pc.chaser"} or not torch.equal(
            stored_control["temporal_shift_random_pc.chaser"], control_basis
        ):
            raise ValueError("resumed causal control basis differs from the fitted basis")
    else:
        save_file(
            {"temporal_shift_random_pc.chaser": control_basis.contiguous()},
            str(control_path),
            metadata={"format": "mrr-zhang-2025-causal-basis-v1"},
        )
    experiment = definition.experiment(unit, horizon=definition.causal.horizon)
    policies = _loaded_policies(experiment, checkpoint, device)
    pipelines = {
        "baseline": InterventionPipeline(),
        "shared_readout": InterventionPipeline(
            (
                SubspaceIntervention(
                    name="top_10_shared_readout_removal",
                    subspace=shared,
                    target="readout",
                ),
            )
        ),
        "shifted_random_pc_readout": InterventionPipeline(
            (
                SubspaceIntervention(
                    name="top_25_shifted_random_pc_readout_removal",
                    subspace=random_control,
                    target="readout",
                ),
            )
        ),
        "shared_recurrent": InterventionPipeline(
            (
                SubspaceIntervention(
                    name="top_10_shared_recurrent_removal",
                    subspace=shared,
                    target="recurrent",
                ),
            )
        ),
    }
    artifacts: dict[str, RolloutArtifact] = {}
    common_world_seed = unit.seed * 10_000 + 7_000
    common_action_seed = unit.seed * 10_000 + 8_000
    for condition, pipeline in pipelines.items():
        path = root / condition
        if path.exists():
            if not resume:
                raise FileExistsError(f"causal rollout exists: {path}")
            artifacts[condition] = load_rollout(path)
        else:
            artifacts[condition] = collect_policy_rollout(
                experiment,
                policies,
                checkpoint,
                path,
                runtime=RuntimeConfig(
                    device=device,
                    environment_backend=backend,
                    batch_size=definition.causal.batch_size,
                    seed=common_world_seed,
                ),
                episodes=definition.causal.episodes,
                horizon=definition.causal.horizon,
                deterministic=False,
                interventions=pipeline,
                action_seed=common_action_seed,
            )
    _validate_paired_worlds(artifacts)
    baseline_rows = artifacts["baseline"].episodes.to_pylist()
    rows = []
    metrics = (
        "event.collision",
        "event_fraction.chaser_partner_visible",
        "world.distance_mean",
    )
    for condition, artifact in artifacts.items():
        for episode, values in enumerate(artifact.episodes.to_pylist()):
            for metric in metrics:
                value = float(values[metric])
                baseline = float(baseline_rows[episode][metric])
                rows.append(
                    {
                        "condition": unit.condition,
                        "seed": unit.seed,
                        "episode_index": episode,
                        "intervention": condition,
                        "metric": metric,
                        "value": value,
                        "baseline": baseline,
                        "paired_delta": value - baseline,
                        "removed_variance": 0.0 if condition == "baseline" else variance[condition],
                    }
                )
    _write_table(result_path, rows, "mrr-zhang-2025-causal-v1")


def _loaded_policies(
    experiment: Any,
    checkpoint: Path,
    device_name: str,
) -> dict[Any, PolicyModule]:
    device = select_device(device_name)
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
        raise ValueError("study checkpoint experiment does not match evaluation experiment")
    return policies


def _annotated_episode_rows(
    artifact: RolloutArtifact,
    unit: StudyUnit,
    *,
    matchup: str,
    checkpoint: int,
) -> list[dict[str, Any]]:
    # Unit identity is written last: the episode table carries its own "seed"
    # column for the rollout stream, which must not become the unit seed.
    return [
        {
            **row,
            "rollout_seed": row.get("seed"),
            "condition": unit.condition,
            "seed": unit.seed,
            "checkpoint_update": checkpoint,
            "matchup": matchup,
        }
        for row in artifact.episodes.to_pylist()
    ]


def _plsc_row(
    unit: StudyUnit,
    update: int,
    visibility: str,
    control: str,
    fitted: Any,
) -> dict[str, Any]:
    return {
        "condition": unit.condition,
        "seed": unit.seed,
        "checkpoint_update": update,
        "visibility": visibility,
        "control": control,
        "status": "ok",
        "reason": None,
        "episodes": fitted.episodes,
        "samples": fitted.samples,
        "significant_dimensions": fitted.significant_count,
        "plsc1_pcc": fitted.plsc1_pcc,
        "null_plsc1_pcc": fitted.null_plsc1_pcc,
        "delta_pcc": fitted.delta_pcc,
    }


def _insufficient_plsc_row(
    unit: StudyUnit,
    update: int,
    visibility: str,
    control: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "condition": unit.condition,
        "seed": unit.seed,
        "checkpoint_update": update,
        "visibility": visibility,
        "control": control,
        "status": "insufficient",
        "reason": reason,
        "episodes": 0,
        "samples": 0,
        "significant_dimensions": None,
        "plsc1_pcc": None,
        "null_plsc1_pcc": None,
        "delta_pcc": None,
    }


def _validate_paired_worlds(artifacts: dict[str, RolloutArtifact]) -> None:
    baseline = artifacts["baseline"].trajectory
    for name, artifact in artifacts.items():
        if artifact.trajectory.episode_seeds.tolist() != baseline.episode_seeds.tolist():
            raise ValueError(f"causal rollout {name} has different episode seeds")
        for key in ("chaser_position", "explorer_position"):
            if not torch.equal(artifact.trajectory.world[key][0], baseline.world[key][0]):
                raise ValueError(f"causal rollout {name} has different initial worlds")


def _aggregate_stage(output: Path, definition: StudyDefinition, stage: str) -> None:
    tables = []
    for unit in definition.units:
        path = _unit_root(output, unit) / stage / "results.parquet"
        if path.is_file():
            tables.append(pq.read_table(path))
    if not tables:
        return
    target = output / "tables" / f"{stage}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    combined = pa.concat_tables(tables, promote_options="default")
    pq.write_table(combined, target, compression="zstd")
    if stage == "behavior":
        summary = output / "tables" / "behavior_checkpoint_summary.parquet"
        _write_behavior_checkpoint_summary(combined, summary)
        _write_behavior_checkpoint_contrasts(
            pq.read_table(summary),
            output / "tables" / "behavior_checkpoint_contrasts.parquet",
        )
    elif stage == "neural":
        _write_seed_level_plsc(combined, output / "tables" / "plsc_seed_summary.parquet")
        decoder_tables = []
        for unit in definition.units:
            visibilities = (
                ("partial",)
                if unit.condition == "social"
                else definition.neural.non_social_visibility_controls
            )
            for update in definition.neural.checkpoint_updates:
                for visibility in visibilities:
                    path = (
                        _unit_root(output, unit)
                        / "neural"
                        / f"update_{update:06d}"
                        / visibility
                        / "decoder"
                        / "results.parquet"
                    )
                    if path.is_file():
                        table = pq.read_table(path)
                        decoder_tables.append(
                            pa.Table.from_pylist(
                            [
                                {
                                    **row,
                                    "condition": unit.condition,
                                    "seed": unit.seed,
                                    "checkpoint_update": update,
                                    "visibility": visibility,
                                }
                                for row in table.to_pylist()
                            ]
                            )
                        )
        if decoder_tables:
            pq.write_table(
                pa.concat_tables(decoder_tables, promote_options="default"),
                output / "tables" / "decoder.parquet",
                compression="zstd",
            )


def _write_behavior_checkpoint_summary(table: pa.Table, path: Path) -> None:
    metric_columns = (
        "event.collision",
        "event_fraction.chaser_partner_visible",
        "world.distance_mean",
        "return.chaser",
        "return.explorer",
        "event.chaser_new_field",
        "event.explorer_new_field",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        key = (
            row["condition"],
            int(row["seed"]),
            int(row["checkpoint_update"]),
            row["matchup"],
        )
        grouped.setdefault(key, []).append(row)
    rows = []
    for (condition, seed, checkpoint, matchup), values in sorted(grouped.items()):
        missing = [column for column in metric_columns if column not in values[0]]
        if missing:
            raise ValueError(f"behavior rollout table lacks metrics {missing!r}")
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "checkpoint_update": checkpoint,
                "matchup": matchup,
                "episodes": len(values),
                **{
                    column: fmean(float(value[column]) for value in values)
                    for column in metric_columns
                },
            }
        )
    _write_table(path, rows, "mrr-zhang-2025-behavior-checkpoint-summary-v1")


def _write_behavior_checkpoint_contrasts(table: pa.Table, path: Path) -> None:
    rows = table.to_pylist()
    checkpoints = sorted({int(row["checkpoint_update"]) for row in rows})

    def mean(checkpoint: int, condition: str, matchup: str, metric: str) -> float:
        selected = [
            float(row[metric])
            for row in rows
            if int(row["checkpoint_update"]) == checkpoint
            and row["condition"] == condition
            and row["matchup"] == matchup
        ]
        if not selected:
            raise ValueError(
                "behavior checkpoint summary lacks a required condition/matchup"
            )
        return fmean(selected)

    chaser_matchup = "trained_chaser_vs_uniform_explorer"
    explorer_matchup = "trained_explorer_vs_uniform_chaser"
    contrasts = []
    for checkpoint in checkpoints:
        social_collision = mean(
            checkpoint, "social", chaser_matchup, "event.collision"
        )
        non_social_collision = mean(
            checkpoint, "non_social", chaser_matchup, "event.collision"
        )
        social_fov = mean(
            checkpoint,
            "social",
            chaser_matchup,
            "event_fraction.chaser_partner_visible",
        )
        non_social_fov = mean(
            checkpoint,
            "non_social",
            chaser_matchup,
            "event_fraction.chaser_partner_visible",
        )
        social_distance = mean(
            checkpoint, "social", chaser_matchup, "world.distance_mean"
        )
        non_social_distance = mean(
            checkpoint, "non_social", chaser_matchup, "world.distance_mean"
        )
        social_explorer_collision = mean(
            checkpoint, "social", explorer_matchup, "event.collision"
        )
        non_social_explorer_collision = mean(
            checkpoint, "non_social", explorer_matchup, "event.collision"
        )
        social_explorer_distance = mean(
            checkpoint, "social", explorer_matchup, "world.distance_mean"
        )
        non_social_explorer_distance = mean(
            checkpoint, "non_social", explorer_matchup, "world.distance_mean"
        )
        contrasts.append(
            {
                "checkpoint_update": checkpoint,
                "chaser_collision_advantage": (
                    social_collision - non_social_collision
                ),
                "chaser_fov_advantage": social_fov - non_social_fov,
                "chaser_distance_advantage": non_social_distance - social_distance,
                "explorer_collision_avoidance": (
                    non_social_explorer_collision - social_explorer_collision
                ),
                "explorer_distance_advantage": (
                    social_explorer_distance - non_social_explorer_distance
                ),
            }
        )
    _write_table(path, contrasts, "mrr-zhang-2025-behavior-checkpoint-contrasts-v1")


def _write_seed_level_plsc(table: pa.Table, path: Path) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        if row["status"] != "ok":
            continue
        key = (row["condition"], row["seed"], row["visibility"], row["control"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for (condition, seed, visibility, control), values in grouped.items():
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "visibility": visibility,
                "control": control,
                "checkpoints": len(values),
                "significant_dimensions_mean": fmean(
                    float(value["significant_dimensions"]) for value in values
                ),
                "plsc1_pcc_mean": fmean(float(value["plsc1_pcc"]) for value in values),
                "delta_pcc_mean": fmean(float(value["delta_pcc"]) for value in values),
            }
        )
    _write_table(path, rows, "mrr-zhang-2025-plsc-seed-summary-v1")


def _write_report(output: Path, definition: StudyDefinition) -> Path:
    available = {stage: (output / "tables" / f"{stage}.parquet").is_file() for stage in STAGES}
    gates = {
        "behavior": _behavior_gate(output, definition, available["behavior"]),
        "plsc": _plsc_gate(output, definition, available["neural"]),
        "causal": _causal_gate(output, definition, available["causal"]),
        "decoder": "DIAGNOSTIC_ONLY",
    }
    comparison = _write_nature_comparison(output, definition, available)
    report = output / "report.md"
    lines = [
        f"# {definition.name}",
        "",
        "This report is generated from strict study artifacts. Development and smoke studies never emit scientific conclusions.",
        "",
        "| Gate | Status |",
        "| --- | --- |",
        *[f"| {name} | {status} |" for name, status in gates.items()],
        "",
        "## Available stages",
        "",
        *[f"- {stage}: {'complete' if complete else 'not run'}" for stage, complete in available.items()],
        "",
        "Behavior checkpoint summaries: `tables/behavior_checkpoint_summary.parquet` and `tables/behavior_checkpoint_contrasts.parquet` when behavior is complete.",
        "",
        "## Nature source-data comparison",
        "",
        f"Comparison-only table: `{comparison.relative_to(output)}`. It is never used as a hard gate.",
        "",
        "Checkpoints and dense states use safetensors; canonical tables use Parquet.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "gates.json").write_text(
        json.dumps({"format": "mrr-zhang-2025-gates-v1", "gates": gates}, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _write_nature_comparison(
    output: Path,
    definition: StudyDefinition,
    available: dict[str, bool],
) -> Path:
    observed = (
        _observed_reference_values(output, available)
        if definition.emits_scientific_gates
        else {}
    )
    rows = []
    for reference in NATURE_REFERENCES:
        key = (reference.category, reference.metric, reference.group)
        value = observed.get(key)
        rows.append(
            {
                "category": reference.category,
                "metric": reference.metric,
                "group": reference.group,
                "panel": reference.panel,
                "label": reference.label,
                "unit": reference.unit,
                "paper_mean": reference.paper_mean,
                "paper_sd": reference.paper_sd,
                "observed_mean": value,
                "observed_minus_paper": (
                    None if value is None else value - reference.paper_mean
                ),
                "role": "comparison_only",
                "scientific_status": "AVAILABLE" if value is not None else "NOT_RUN",
            }
        )
    path = output / "tables" / "nature_comparison.parquet"
    _write_table(path, rows, "mrr-zhang-2025-nature-comparison-v1")
    return path


def _observed_reference_values(
    output: Path,
    available: dict[str, bool],
) -> dict[tuple[str, str, str], float]:
    observed: dict[tuple[str, str, str], float] = {}
    if available["behavior"]:
        rows = pq.read_table(output / "tables" / "behavior.parquet").to_pylist()
        matchup = "trained_chaser_vs_uniform_explorer"
        metric_columns = {
            "collision": ("event.collision", 1.0),
            "fov": ("event_fraction.chaser_partner_visible", 100.0),
            "distance": ("world.distance_mean", 1.0),
        }
        for condition in ("social", "non_social"):
            selected = [
                row
                for row in rows
                if row["condition"] == condition and row["matchup"] == matchup
            ]
            for metric, (column, scale) in metric_columns.items():
                observed[("behavior", metric, condition)] = (
                    fmean(float(row[column]) for row in selected) * scale
                )
    if available["neural"]:
        rows = pq.read_table(output / "tables" / "plsc_seed_summary.parquet").to_pylist()
        mapping = {
            "social": ("social", "partial"),
            "non_social": ("non_social", "none"),
            "non_social_with_vision": ("non_social", "partial"),
        }
        for group, (condition, visibility) in mapping.items():
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["visibility"] == visibility
                and row["control"] == "primary"
            ]
            observed[("plsc", "significant_dimensions", group)] = fmean(
                float(row["significant_dimensions_mean"]) for row in selected
            )
            observed[("plsc", "delta_pcc", group)] = fmean(
                float(row["delta_pcc_mean"]) for row in selected
            )
    if available["causal"]:
        rows = pq.read_table(output / "tables" / "causal.parquet").to_pylist()
        metric_columns = {
            "collision": ("event.collision", 1.0),
            "fov": ("event_fraction.chaser_partner_visible", 100.0),
            "distance": ("world.distance_mean", 1.0),
        }
        for group in ("baseline", "shared_readout", "shifted_random_pc_readout"):
            for metric, (column, scale) in metric_columns.items():
                selected = [
                    float(row["value"])
                    for row in rows
                    if row["intervention"] == group and row["metric"] == column
                ]
                observed[("causal", metric, group)] = fmean(selected) * scale
    return observed


def _gate_status(definition: StudyDefinition, available: bool) -> str | None:
    if not available or not definition.emits_scientific_gates:
        return "NOT_RUN"
    return None


def _behavior_gate(output: Path, definition: StudyDefinition, available: bool) -> str:
    precondition = _gate_status(definition, available)
    if precondition is not None:
        return precondition
    rows = pq.read_table(output / "tables" / "behavior.parquet").to_pylist()
    matchup = "trained_chaser_vs_uniform_explorer"
    per_seed: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        if row["matchup"] != matchup:
            continue
        key = (row["condition"], int(row["seed"]))
        metrics = per_seed.setdefault(key, {"collision": [], "fov": [], "distance": []})
        metrics["collision"].append(float(row["event.collision"]))
        metrics["fov"].append(float(row["event_fraction.chaser_partner_visible"]))
        metrics["distance"].append(float(row["world.distance_mean"]))
    if len(per_seed) != len(definition.units):
        return "NOT_RUN"
    means = {
        condition: {
            metric: fmean(
                fmean(per_seed[(condition, seed)][metric]) for seed in definition.seeds
            )
            for metric in ("collision", "fov", "distance")
        }
        for condition in definition.conditions
    }
    supported = (
        means["social"]["collision"] > means["non_social"]["collision"]
        and means["social"]["fov"] > means["non_social"]["fov"]
        and means["social"]["distance"] < means["non_social"]["distance"]
    )
    return "SUPPORTED" if supported else "FAILED"


def _plsc_gate(output: Path, definition: StudyDefinition, available: bool) -> str:
    precondition = _gate_status(definition, available)
    if precondition is not None:
        return precondition
    rows = pq.read_table(output / "tables" / "plsc_seed_summary.parquet").to_pylist()
    indexed = {
        (row["condition"], int(row["seed"]), row["visibility"], row["control"]): row
        for row in rows
    }
    required = [
        (condition, seed, visibility, control)
        for seed in definition.seeds
        for condition, visibility, control in (
            ("social", "partial", "primary"),
            ("social", "partial", "identical_action_excluded"),
            ("non_social", "none", "primary"),
            ("non_social", "partial", "primary"),
            ("non_social", "full", "primary"),
        )
    ]
    if any(key not in indexed for key in required):
        return "NOT_RUN"

    def group_mean(condition: str, visibility: str, control: str, metric: str) -> float:
        return fmean(
            float(indexed[(condition, seed, visibility, control)][metric])
            for seed in definition.seeds
        )

    for metric in ("significant_dimensions_mean", "delta_pcc_mean"):
        social = group_mean("social", "partial", "primary", metric)
        identical_excluded = group_mean(
            "social", "partial", "identical_action_excluded", metric
        )
        non_social_controls = [
            group_mean("non_social", visibility, "primary", metric)
            for visibility in ("none", "partial", "full")
        ]
        if social <= max(non_social_controls) or identical_excluded <= max(
            non_social_controls
        ):
            return "FAILED"
    return "SUPPORTED"


def _causal_gate(output: Path, definition: StudyDefinition, available: bool) -> str:
    precondition = _gate_status(definition, available)
    if precondition is not None:
        return precondition
    rows = pq.read_table(output / "tables" / "causal.parquet").to_pylist()
    grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in rows:
        if row["intervention"] not in ("shared_readout", "shifted_random_pc_readout"):
            continue
        grouped.setdefault(
            (int(row["seed"]), row["intervention"], row["metric"]), []
        ).append(float(row["paired_delta"]))
    comparisons = []
    for seed in definition.seeds:
        values = {
            (intervention, metric): fmean(grouped[(seed, intervention, metric)])
            for intervention in ("shared_readout", "shifted_random_pc_readout")
            for metric in (
                "event.collision",
                "event_fraction.chaser_partner_visible",
                "world.distance_mean",
            )
            if (seed, intervention, metric) in grouped
        }
        if len(values) != 6:
            return "NOT_RUN"
        comparisons.append(
            values[("shared_readout", "event.collision")]
            < values[("shifted_random_pc_readout", "event.collision")]
            and values[("shared_readout", "event_fraction.chaser_partner_visible")]
            < values[("shifted_random_pc_readout", "event_fraction.chaser_partner_visible")]
            and values[("shared_readout", "world.distance_mean")]
            > values[("shifted_random_pc_readout", "world.distance_mean")]
        )
    return "SUPPORTED" if sum(comparisons) > len(comparisons) / 2 else "FAILED"


def _prepare_study(
    output: Path,
    target: str,
    definition: StudyDefinition,
    *,
    resume: bool,
) -> None:
    payload = {
        "format": STUDY_FORMAT,
        "target": target,
        "definition": definition.as_dict(),
    }
    payload["config_sha256"] = _json_hash(payload["definition"])
    payload = json.loads(json.dumps(payload, sort_keys=True))
    manifest = output / "study.json"
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"study output is not empty: {output}; use --resume")
        if not manifest.is_file():
            raise ValueError("resume target has no strict study manifest")
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("resume study definition/config does not match existing output")
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_status(
    output: Path,
    stage: str,
    state: str,
    completed: int,
    total: int,
) -> None:
    payload = {
        "format": "mrr-study-status-v1",
        "stage": stage,
        "state": state,
        "completed_units": completed,
        "total_units": total,
    }
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "status.json")


def _write_unit_status(
    unit_root: Path,
    stage: str,
    state: str,
    *,
    error: str | None = None,
) -> None:
    path = unit_root / "stage_status" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "mrr-study-unit-status-v1",
        "stage": stage,
        "state": state,
        "error": error,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_table(path: Path, rows: list[dict[str, Any]], format_name: str) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"mrr_format": format_name.encode()}
    )
    pq.write_table(table, path, compression="zstd")


def _completed_output(path: Path, resume: bool) -> bool:
    if not path.exists():
        return False
    if resume:
        return True
    raise FileExistsError(f"study stage output already exists: {path}")


def _required_checkpoint(unit_root: Path, update: int) -> Path:
    path = unit_root / "train" / "checkpoints" / f"update_{update:06d}.safetensors"
    if not path.is_file():
        latest = unit_root / "train" / "checkpoints" / "latest.safetensors"
        if latest.is_file():
            from mouse_run_run.artifacts.checkpoint import read_checkpoint_metadata

            if read_checkpoint_metadata(latest).update == update:
                return latest
        raise FileNotFoundError(f"required study checkpoint is missing: {path}")
    return path


def _latest_training_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    from mouse_run_run.artifacts.checkpoint import read_checkpoint_metadata

    candidates = []
    for path in checkpoint_dir.glob("*.safetensors"):
        candidates.append((read_checkpoint_metadata(path).update, path))
    return max(candidates, default=(0, None))[1]


def _unit_root(output: Path, unit: StudyUnit) -> Path:
    return output / "units" / unit.condition / f"seed_{unit.seed:04d}"


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

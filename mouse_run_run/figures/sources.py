"""Figure source data assembled from study artifacts.

Figures read the artifacts a study writes — per-unit ``metrics.jsonl`` for
training curves and Parquet/rollout artifacts for evaluation panels — never the
TensorBoard event files, which are a display side-channel with no schema
guarantee.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mouse_run_run.analyses.flow_field import (
    MovementSamples,
    angle_histogram_rows,
    flow_field_rows,
    movement_samples,
    polar_rows,
)
from mouse_run_run.analyses.statistics import mean_sem, permutation_comparison
from mouse_run_run.artifacts.rollout import load_rollout


CHASER_MATCHUP = "trained_chaser_vs_uniform_explorer"
EXPLORER_MATCHUP = "trained_explorer_vs_uniform_chaser"


@dataclass(frozen=True)
class Unit:
    condition: str
    seed: int
    root: Path


def discover_units(study_output: Path) -> list[Unit]:
    """Every ``units/<condition>/seed_XXXX`` directory, in a stable order."""

    units_root = study_output / "units"
    if not units_root.is_dir():
        raise FileNotFoundError(f"study output has no units directory: {units_root}")
    units = [
        Unit(condition=condition.name, seed=int(seed.name.removeprefix("seed_")), root=seed)
        for condition in sorted(units_root.iterdir())
        if condition.is_dir()
        for seed in sorted(condition.iterdir())
        if seed.is_dir() and seed.name.startswith("seed_")
    ]
    if not units:
        raise FileNotFoundError(f"study output has no units: {units_root}")
    return units


def training_rows(study_output: Path, metrics: Sequence[str]) -> list[dict[str, Any]]:
    """Per-update training metrics for every unit, one row per metric value."""

    wanted = set(metrics)
    rows: list[dict[str, Any]] = []
    for unit in discover_units(study_output):
        path = unit.root / "train" / "metrics.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for metric in wanted & record.keys():
                rows.append(
                    {
                        "condition": unit.condition,
                        "seed": unit.seed,
                        "update": int(record["update"]),
                        "metric": metric,
                        "value": float(record[metric]),
                    }
                )
    if not rows:
        raise FileNotFoundError(
            f"no training metrics found under {study_output / 'units'}; "
            "run the train stage first"
        )
    return rows


def summarize_curves(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and s.e.m. across seeds for each metric, condition, and update."""

    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = (row["metric"], row["condition"], int(row["update"]))
        grouped.setdefault(key, []).append(float(row["value"]))
    summary = []
    for (metric, condition, update), values in sorted(grouped.items()):
        mean, sem = mean_sem(values)
        summary.append(
            {
                "metric": metric,
                "condition": condition,
                "update": update,
                "mean": mean,
                "sem": sem,
                "seeds": len(values),
            }
        )
    return summary


def behavior_table(study_output: Path) -> pa.Table:
    """The behavior stage episode table, combined from units when needed."""

    combined = study_output / "tables" / "behavior.parquet"
    if combined.is_file():
        return pq.read_table(combined)
    tables = [
        pq.read_table(unit.root / "behavior" / "results.parquet")
        for unit in discover_units(study_output)
        if (unit.root / "behavior" / "results.parquet").is_file()
    ]
    if not tables:
        raise FileNotFoundError(
            f"no behavior results under {study_output}; run the behavior stage first"
        )
    return pa.concat_tables(tables)


def behavior_seed_rows(
    table: pa.Table,
    *,
    matchup: str,
    metric: str,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Per-seed episode means, one row per condition, seed, and checkpoint."""

    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in table.to_pylist():
        if row["matchup"] != matchup or row.get(metric) is None:
            continue
        key = (row["condition"], int(row["seed"]), int(row["checkpoint_update"]))
        grouped.setdefault(key, []).append(float(row[metric]) * scale)
    return [
        {
            "condition": condition,
            "seed": seed,
            "checkpoint_update": checkpoint,
            "metric": metric,
            "value": sum(values) / len(values),
            "episodes": len(values),
        }
        for (condition, seed, checkpoint), values in sorted(grouped.items())
    ]


def summarize_behavior(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and s.e.m. across seeds for each condition and checkpoint."""

    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], int(row["checkpoint_update"])), []).append(
            float(row["value"])
        )
    summary = []
    for (condition, checkpoint), values in sorted(grouped.items()):
        mean, sem = mean_sem(values)
        summary.append(
            {
                "condition": condition,
                "checkpoint_update": checkpoint,
                "mean": mean,
                "sem": sem,
                "seeds": len(values),
            }
        )
    return summary


def compare_conditions(
    rows: Sequence[dict[str, Any]],
    *,
    checkpoint: int,
    left: str = "social",
    right: str = "non_social",
    permutations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Permutation comparison of two conditions at one checkpoint."""

    def values(condition: str) -> list[float]:
        return [
            float(row["value"])
            for row in rows
            if row["condition"] == condition and int(row["checkpoint_update"]) == checkpoint
        ]

    left_values = values(left)
    right_values = values(right)
    if not left_values or not right_values:
        return {
            "checkpoint_update": checkpoint,
            "left": left,
            "right": right,
            "status": "insufficient_groups",
            "p_value": None,
            "stars": "",
            "left_n": len(left_values),
            "right_n": len(right_values),
        }
    comparison = permutation_comparison(
        left_values, right_values, permutations=permutations, seed=seed
    )
    return {
        "checkpoint_update": checkpoint,
        "left": left,
        "right": right,
        "status": "ok",
        "left_mean": comparison.left_mean,
        "right_mean": comparison.right_mean,
        "difference": comparison.difference,
        "p_value": comparison.p_value,
        "stars": comparison.stars,
        "permutations": comparison.permutations,
        "exact": comparison.exact,
        "left_n": comparison.left_n,
        "right_n": comparison.right_n,
    }


def movement_samples_by_seed(
    study_output: Path,
    *,
    checkpoint: int,
    condition: str = "social",
    matchup: str = CHASER_MATCHUP,
) -> dict[int, MovementSamples]:
    """Chaser movement samples per seed, from stored behavior rollouts."""

    samples: dict[int, MovementSamples] = {}
    for unit in discover_units(study_output):
        if unit.condition != condition:
            continue
        path = unit.root / "behavior" / f"update_{checkpoint:06d}" / matchup
        if not path.is_dir():
            continue
        samples[unit.seed] = movement_samples(load_rollout(path).trajectory)
    if not samples:
        raise FileNotFoundError(
            f"no {condition} behavior rollouts for checkpoint {checkpoint} under {study_output}"
        )
    return samples


def pooled_samples(samples: Iterable[MovementSamples]) -> MovementSamples:
    """Concatenate per-seed samples for pooled flow-field and polar panels."""

    items = list(samples)
    return MovementSamples(
        offset=torch.cat([item.offset for item in items]),
        movement=torch.cat([item.movement for item in items]),
        to_partner=torch.cat([item.to_partner for item in items]),
        angle=torch.cat([item.angle for item in items]),
    )


def flow_rows(
    samples: MovementSamples,
    *,
    stage: str,
    checkpoint: int,
    vision_radius: int,
) -> list[dict[str, Any]]:
    return [
        {"stage": stage, "checkpoint_update": checkpoint, **row}
        for row in flow_field_rows(samples, vision_radius=vision_radius)
    ]


def polar_stage_rows(
    samples: MovementSamples,
    *,
    stage: str,
    checkpoint: int,
    bins: int = 12,
) -> list[dict[str, Any]]:
    return [
        {"stage": stage, "checkpoint_update": checkpoint, **row}
        for row in polar_rows(samples, bins=bins)
    ]


def angle_rows_by_seed(
    samples: dict[int, MovementSamples],
    *,
    stage: str,
    checkpoint: int,
    bin_width: float = 30.0,
) -> list[dict[str, Any]]:
    return [
        {"stage": stage, "checkpoint_update": checkpoint, "seed": seed, **row}
        for seed, seed_samples in sorted(samples.items())
        for row in angle_histogram_rows(seed_samples, bin_width=bin_width)
    ]


def write_source_table(path: Path, rows: Sequence[dict[str, Any]], format_name: str) -> Path:
    """Persist one figure panel's source data next to the rendered figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    table = pa.Table.from_pylist([{key: row.get(key) for key in columns} for row in rows])
    table = table.replace_schema_metadata({b"mrr_format": format_name.encode()})
    pq.write_table(table, path, compression="zstd")
    return path

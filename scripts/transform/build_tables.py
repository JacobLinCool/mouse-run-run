from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors import safe_open

from mouse_run_run.health import checkpoint_health
from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]

# Newest record schema each table knows how to ingest. Rows written by newer
# code are skipped with a warning instead of being silently folded into
# canonical tables; rows without a schema_version predate versioning and are
# ingested as legacy.
EVALUATION_MAX_SCHEMA_VERSION = 3
ROLLOUT_RECORD_MAX_SCHEMA_VERSION = 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    experiment_root = args.experiment_root
    if not experiment_root.exists():
        raise FileNotFoundError(str(experiment_root))
    output_root = args.output_root or REPO_ROOT / "runs" / "tables" / experiment_root.resolve().name
    output_root.mkdir(parents=True, exist_ok=True)

    runs = _run_rows(experiment_root)
    evaluations = _dedupe_last(
        _jsonl_rows(
            experiment_root.rglob("*eval*.jsonl"),
            max_schema_version=EVALUATION_MAX_SCHEMA_VERSION,
        ),
        key=lambda row: (
            row.get("checkpoint"),
            row.get("opponent_mode"),
            row.get("evaluation_protocol"),
        ),
    )
    rollouts, exclusions = _rollout_rows(experiment_root)
    panels = _panels()

    outputs = {
        "runs": output_root / "runs.jsonl",
        "evaluations": output_root / "evaluations.jsonl",
        "rollouts": output_root / "rollouts.jsonl",
        "exclusions": output_root / "exclusions.jsonl",
    }
    _write_jsonl(outputs["runs"], runs)
    _write_jsonl(outputs["evaluations"], evaluations)
    _write_jsonl(outputs["rollouts"], rollouts)
    _write_jsonl(outputs["exclusions"], exclusions)
    write_json_atomic(output_root / "panels.json", panels)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_root": str(experiment_root),
        "row_counts": {
            "runs": len(runs),
            "evaluations": len(evaluations),
            "rollouts": len(rollouts),
            "exclusions": len(exclusions),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_hashes": {
            name: hash_file(path)
            for name, path in outputs.items()
            if path.exists()
        },
        "panel_path": str(output_root / "panels.json"),
        "transform_script": str(Path(__file__).resolve()),
        "provenance": collect_provenance(cwd=REPO_ROOT),
    }
    write_json_atomic(output_root / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _run_rows(experiment_root: Path) -> list[dict[str, Any]]:
    rows = []
    for status_path in sorted(experiment_root.rglob("status.json")):
        if "tensorboard" in status_path.parts:
            continue
        status = _read_json(status_path)
        run_dir = status_path.parent
        checkpoint = run_dir / "checkpoints" / "latest.safetensors"
        health = checkpoint_health(checkpoint)
        config = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        task = _path_part_after(experiment_root, run_dir, 0)
        seed_text = _path_part_after(experiment_root, run_dir, 1)
        attempt_id = _path_part_after(experiment_root, run_dir, 2)
        seed = int(seed_text.removeprefix("seed_")) if seed_text.startswith("seed_") else None
        row = {
            "schema_version": 1,
            "experiment": experiment_root.name,
            "task": task,
            "seed": seed,
            "unit_id": f"{task}/{seed_text}",
            "attempt_id": attempt_id,
            # Resolved so downstream readers (metrics.jsonl lookups) do not
            # depend on the cwd this transform happened to run from.
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "status": status.get("state"),
            "healthy": bool(status.get("healthy")) and health.ok,
            "checkpoint_ok": health.ok,
            "checkpoint_health": health.to_dict(include_path=False),
            "update": status.get("update"),
            "total_updates": status.get("total_updates"),
            "environment_steps": status.get("environment_steps"),
            "elapsed_seconds": status.get("elapsed_seconds"),
            "eta_seconds": status.get("eta_seconds"),
            "error": status.get("error"),
            "metrics": status.get("metrics", {}),
            "config": config,
        }
        rows.append(row)
    _mark_latest_successful(rows)
    return rows


def _mark_latest_successful(rows: list[dict[str, Any]]) -> None:
    """Set is_latest_successful on the newest valid attempt of each unit.

    This materializes the SPEC's analysis unit ("latest successful finite
    attempt per unit") as a column so downstream consumers filter on it
    instead of re-deriving it.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        row["is_latest_successful"] = False
        if row.get("status") != "completed" or not row.get("healthy") or not row.get("checkpoint_ok"):
            continue
        unit_id = str(row.get("unit_id"))
        current = latest.get(unit_id)
        if current is None or str(row.get("attempt_id")) > str(current.get("attempt_id")):
            latest[unit_id] = row
    for row in latest.values():
        row["is_latest_successful"] = True


def _jsonl_rows(paths: object, *, max_schema_version: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not _schema_version_compatible(row.get("schema_version"), max_schema_version):
                    print(
                        f"warning: skipping {path}:{line_number}:"
                        f" schema_version={row.get('schema_version')!r} is not"
                        f" ingestible (supported: absent or <= {max_schema_version})",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                row["source_path"] = str(path)
                row["source_line"] = line_number
                rows.append(row)
    return rows


def _schema_version_compatible(version: Any, max_schema_version: int) -> bool:
    if version is None:
        return True  # Legacy rows predate schema versioning.
    return (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version <= max_schema_version
    )


def _dedupe_last(
    rows: list[dict[str, Any]],
    *,
    key: Any,
) -> list[dict[str, Any]]:
    """Keep only the last record per key, preserving first-seen order.

    Records files are append-only, so re-runs (e.g. --force) append newer
    records for the same checkpoint; the newest one wins in canonical tables.
    """
    deduped: dict[Any, dict[str, Any]] = {}
    for row in rows:
        deduped[key(row)] = row
    return list(deduped.values())


def _rollout_rows(experiment_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _dedupe_last(
        _jsonl_rows(
            experiment_root.rglob("*rollout*_records.jsonl"),
            max_schema_version=ROLLOUT_RECORD_MAX_SCHEMA_VERSION,
        ),
        key=lambda row: (row.get("checkpoint"), row.get("output")),
    )
    rows = []
    exclusions = []
    for record in records:
        output = Path(str(record.get("output", "")))
        row = dict(record)
        degenerate_count = None
        episode_count = None
        if output.exists() and output.suffix == ".safetensors":
            # Read the single flag tensor instead of the whole rollout file.
            with safe_open(str(output), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                degenerate = (
                    handle.get_tensor("rollout.episode_degenerate").bool()
                    if "rollout.episode_degenerate" in keys
                    else None
                )
            if degenerate is not None:
                degenerate_count = int(degenerate.sum().item())
                episode_count = int(degenerate.numel())
                if degenerate_count:
                    exclusions.append(
                        {
                            "schema_version": 1,
                            "source_rollout": str(output),
                            "reason": "paper_degenerate_episode_v1",
                            "excluded_count": degenerate_count,
                            "total_count": episode_count,
                            "threshold_fraction": record.get("degenerate_threshold_fraction"),
                        }
                    )
        row["degenerate_episode_count"] = degenerate_count
        row["episode_count"] = episode_count
        rows.append(row)
    return rows, exclusions


def _panels() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "panels": {
            "primary_valid_runs_v1": {
                "table": "runs.jsonl",
                "include": {
                    "status": "completed",
                    "healthy": True,
                    "checkpoint_ok": True,
                    "is_latest_successful": True,
                },
                "unit": "task x seed latest successful attempt",
            },
            "paper_random_opponent_valid_v1": {
                "table": "evaluations.jsonl",
                "include": {"status": "ok", "evaluation_protocol": "paper_random_opponent_v1"},
                "unit": "checkpoint x opponent_mode",
            },
            "paper_neural_behavior_non_degenerate_v1": {
                "table": "rollouts.jsonl",
                "exclude_by": "exclusions.jsonl reason=paper_degenerate_episode_v1",
                "unit": "checkpoint rollout episode",
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_part_after(root: Path, path: Path, index: int) -> str:
    try:
        return path.relative_to(root).parts[index]
    except IndexError:
        return ""


if __name__ == "__main__":
    main()

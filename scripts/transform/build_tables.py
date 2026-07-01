from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from mouse_run_run.health import checkpoint_health
from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    experiment_root = args.experiment_root
    if not experiment_root.exists():
        raise FileNotFoundError(str(experiment_root))
    output_root = args.output_root or Path("runs/tables") / experiment_root.name
    output_root.mkdir(parents=True, exist_ok=True)

    runs = _run_rows(experiment_root)
    evaluations = _jsonl_rows(experiment_root.rglob("*eval*.jsonl"))
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
        "provenance": collect_provenance(cwd=Path.cwd()),
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
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "status": status.get("state"),
            "healthy": bool(status.get("healthy")) and health.ok,
            "checkpoint_ok": health.ok,
            "checkpoint_health": {
                "ok": health.ok,
                "format_ok": health.format_ok,
                "tensor_count": health.tensor_count,
                "nonfinite_tensor_count": health.nonfinite_tensor_count,
                "max_abs": health.max_abs,
                "failures": list(health.failures),
            },
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
    return rows


def _jsonl_rows(paths: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["source_path"] = str(path)
                row["source_line"] = line_number
                rows.append(row)
    return rows


def _rollout_rows(experiment_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _jsonl_rows(experiment_root.rglob("*rollout*_records.jsonl"))
    rows = []
    exclusions = []
    for record in records:
        output = Path(str(record.get("output", "")))
        row = dict(record)
        degenerate_count = None
        episode_count = None
        if output.exists() and output.suffix == ".safetensors":
            tensors = load_file(str(output), device="cpu")
            if "rollout.episode_degenerate" in tensors:
                degenerate = tensors["rollout.episode_degenerate"].bool()
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
                "include": {"status": "completed", "healthy": True, "checkpoint_ok": True},
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

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables_root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    tables_root = args.tables_root
    output_root = args.output_root or Path("runs/reports") / tables_root.name
    output_root.mkdir(parents=True, exist_ok=True)

    runs = _read_jsonl(tables_root / "runs.jsonl")
    evaluations = _read_jsonl(tables_root / "evaluations.jsonl")
    rollouts = _read_jsonl(tables_root / "rollouts.jsonl")
    exclusions = _read_jsonl(tables_root / "exclusions.jsonl")

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "tables_root": str(tables_root),
        "valid_pairs_by_task": _valid_pairs_by_task(runs),
        "failed_attempts": [row for row in runs if row.get("status") == "failed"],
        "training_metrics": _training_metrics(runs),
        "evaluation_metrics": _evaluation_metrics(evaluations),
        "rollout_exclusions": _rollout_exclusions(exclusions),
    }
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    write_json_atomic(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "input_tables": {
            name: str(tables_root / name)
            for name in ("runs.jsonl", "evaluations.jsonl", "rollouts.jsonl", "exclusions.jsonl")
        },
        "input_hashes": {
            name: hash_file(tables_root / name)
            for name in ("runs.jsonl", "evaluations.jsonl", "rollouts.jsonl", "exclusions.jsonl")
            if (tables_root / name).exists()
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "analysis_script": str(Path(__file__).resolve()),
        "provenance": collect_provenance(cwd=Path.cwd()),
    }
    write_json_atomic(output_root / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _valid_pairs_by_task(runs: list[dict[str, Any]]) -> dict[str, int]:
    valid_units: dict[str, set[str]] = {}
    for row in runs:
        if row.get("status") != "completed" or not row.get("healthy") or not row.get("checkpoint_ok"):
            continue
        task = str(row.get("task"))
        valid_units.setdefault(task, set()).add(str(row.get("unit_id")))
    return {task: len(units) for task, units in sorted(valid_units.items())}


def _training_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        row
        for row in runs
        if row.get("status") == "completed" and row.get("healthy") and row.get("checkpoint_ok")
    ]
    output: dict[str, Any] = {}
    for task in sorted({str(row.get("task")) for row in metrics}):
        task_rows = [row for row in metrics if row.get("task") == task]
        output[task] = {
            key: _mean_std(
                [
                    float(row.get("metrics", {}).get(key))
                    for row in task_rows
                    if row.get("metrics", {}).get(key) is not None
                ]
            )
            for key in (
                "collisions_per_episode",
                "chaser_return",
                "explorer_return",
                "chaser_partner_vision",
                "explorer_partner_vision",
                "chaser_new_fields",
                "explorer_new_fields",
                "final_distance",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "grad_norm",
                "max_param_abs",
            )
        }
    return output


def _evaluation_metrics(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    ok_rows = [row for row in evaluations if row.get("status") == "ok"]
    for mode in sorted({str(row.get("opponent_mode")) for row in ok_rows}):
        rows = [row for row in ok_rows if row.get("opponent_mode") == mode]
        output[mode] = {
            key: _mean_std(
                [
                    float(row.get("metrics", {}).get(key))
                    for row in rows
                    if row.get("metrics", {}).get(key) is not None
                ]
            )
            for key in (
                "collisions_per_episode",
                "chaser_return",
                "explorer_return",
                "chaser_partner_vision",
                "explorer_partner_vision",
                "chaser_new_fields",
                "explorer_new_fields",
                "final_distance",
                "average_distance",
                "degenerate_fraction",
            )
        }
        output[mode]["n"] = len(rows)
    output["failed_count"] = len([row for row in evaluations if row.get("status") != "ok"])
    return output


def _rollout_exclusions(exclusions: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = sum(int(row.get("excluded_count", 0)) for row in exclusions)
    total = sum(int(row.get("total_count", 0)) for row in exclusions)
    return {
        "excluded_degenerate_episodes": excluded,
        "episodes_with_exclusion_records": total,
        "records": len(exclusions),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# MARL Experiment Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Valid Pairs",
    ]
    for task, count in summary["valid_pairs_by_task"].items():
        lines.append(f"- `{task}`: {count}")
    lines.extend(
        [
            "",
            "## Training Metrics",
            "```json",
            json.dumps(summary["training_metrics"], indent=2, sort_keys=True),
            "```",
            "",
            "## Random-Opponent Evaluation",
            "```json",
            json.dumps(summary["evaluation_metrics"], indent=2, sort_keys=True),
            "```",
            "",
            "## Degenerate Episode Exclusions",
            "```json",
            json.dumps(summary["rollout_exclusions"], indent=2, sort_keys=True),
            "```",
            "",
            "## Failed Attempts",
            f"- count: {len(summary['failed_attempts'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()

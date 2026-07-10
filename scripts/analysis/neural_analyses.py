from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mouse_run_run.analysis import (
    decode_balanced_accuracy,
    decoding_targets,
    episode_hidden_pairs,
    load_rollout,
    plsc_shared_dimensions,
)
from mouse_run_run.checkpoint_select import checkpoint_identity
from mouse_run_run.provenance import append_jsonl, collect_provenance, write_json_atomic


DECODING_TARGETS = (
    "chaser_collision",
    "chaser_partner_escape",
    "explorer_collision",
    "explorer_partner_approach",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rollout_root",
        type=Path,
        nargs="?",
        default=Path("runs/mouse-run-run-1/paper_rollouts"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output_root = args.output or Path("runs/analyses") / args.rollout_root.parent.name
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "neural_records.jsonl"

    rollout_paths = sorted(args.rollout_root.glob("*.safetensors"))
    if args.limit is not None:
        rollout_paths = rollout_paths[: args.limit]
    if not rollout_paths:
        raise SystemExit(f"no rollout safetensors under {args.rollout_root}")

    records = []
    for path in rollout_paths:
        record = _analyze_rollout(path, seed=args.seed, permutations=args.permutations)
        append_jsonl(records_path, record)
        records.append(record)
        if record.get("status", "ok") == "ok":
            message = (
                f"plsc_dims={record['plsc']['n_significant']}"
                f" top_r={record['plsc']['top_dim_correlation']:.3f}"
            )
        else:
            message = f"degenerate ({record['n_valid_episodes']}/{record['n_total_episodes']} valid)"
        print(f"analyzed {record['unit_id']}: {message}", flush=True)

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "rollout_root": str(args.rollout_root),
        "seed": args.seed,
        "permutations": args.permutations,
        "n_rollouts": len(records),
        "by_task": _summarize_by_task(records),
        "provenance": collect_provenance(cwd=Path.cwd()),
    }
    write_json_atomic(output_root / "neural_summary.json", summary)
    print(json.dumps(summary["by_task"], indent=2, sort_keys=True))


MIN_VALID_EPISODES = 3


def _analyze_rollout(path: Path, *, seed: int, permutations: int) -> dict[str, Any]:
    metadata, tensors = load_rollout(path)
    config = metadata.get("checkpoint_config") or {}
    identity = checkpoint_identity(config)
    chaser_episodes, explorer_episodes = episode_hidden_pairs(tensors)
    n_valid = len(chaser_episodes)

    record: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "rollout": str(path),
        "checkpoint": metadata.get("checkpoint"),
        "task": identity["task"],
        "seed": identity["seed"],
        "architecture": config.get("architecture", "rnn"),
        "unit_id": identity["unit_id"] or str(path),
        "n_valid_episodes": n_valid,
        "n_total_episodes": int(tensors["episode_degenerate"].shape[0]),
    }
    # A unit with almost no non-degenerate episodes has no analyzable
    # representation; recorded as degenerate (itself a result) rather than
    # forcing a PLSC/decoding estimate from a handful of timesteps.
    if n_valid < MIN_VALID_EPISODES:
        record["status"] = "degenerate"
        record["plsc"] = None
        record["decoding"] = None
        return record

    record["status"] = "ok"
    plsc = plsc_shared_dimensions(
        chaser_episodes,
        explorer_episodes,
        permutations=permutations,
        seed=seed,
    )
    record["plsc"] = asdict(plsc)
    record["decoding"] = {
        name: asdict(decode_balanced_accuracy(hidden, labels, seed=seed))
        for name, (hidden, labels) in decoding_targets(tensors).items()
    }
    return record


def _summarize_by_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task in sorted({str(record.get("task")) for record in records}):
        task_rows = [record for record in records if str(record.get("task")) == task]
        rows = [row for row in task_rows if row.get("status", "ok") == "ok"]
        summary: dict[str, Any] = {
            "n": len(task_rows),
            "n_analyzable": len(rows),
            "n_degenerate": len(task_rows) - len(rows),
            "plsc_n_significant": _mean_std(
                [row["plsc"]["n_significant"] for row in rows]
            ),
            "plsc_top_dim_correlation": _mean_std(
                [row["plsc"]["top_dim_correlation"] for row in rows]
            ),
        }
        for target in DECODING_TARGETS:
            accuracies = [
                row["decoding"][target]["balanced_accuracy"]
                for row in rows
                if row["decoding"][target]["balanced_accuracy"] is not None
            ]
            shuffled = [
                row["decoding"][target]["shuffled_accuracy"]
                for row in rows
                if row["decoding"][target]["shuffled_accuracy"] is not None
            ]
            summary[f"decode_{target}"] = _mean_std(accuracies)
            summary[f"decode_{target}_shuffled"] = _mean_std(shuffled)
        output[task] = summary
    return output


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


if __name__ == "__main__":
    main()

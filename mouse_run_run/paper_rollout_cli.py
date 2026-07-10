from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

from safetensors import SafetensorError

from mouse_run_run.checkpoint_select import checkpoint_identity, select_checkpoints
from mouse_run_run.health import checkpoint_health
from mouse_run_run.rollout import collect_rollouts
from mouse_run_run.serialization import read_checkpoint_metadata
from mouse_run_run.training_config import DEVICE_CHOICES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="*", default=[Path("runs")])
    parser.add_argument("--output-root", type=Path, default=Path("runs/raw/rollouts"))
    parser.add_argument("--records", type=Path)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--degenerate-threshold-fraction", type=float, default=0.01)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed shared by every collection so rollouts are reproducible.",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Collect for every checkpoint under the given directories,"
        " including mid-training update_* files and failed attempts. Default"
        " is the latest successful attempt per unit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-collect rollouts that already have an ok record.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    checkpoints = select_checkpoints(args.paths, all_checkpoints=args.all_checkpoints)
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit("no checkpoint safetensors found")

    records_path = args.records or args.output_root / "paper_rollout_records.jsonl"
    already_collected = set() if args.force else _collected_checkpoints(records_path)
    failure_count = 0
    for checkpoint in checkpoints:
        output = args.output_root / f"{_output_stem(checkpoint)}.safetensors"
        if str(checkpoint) in already_collected and output.exists():
            print(f"skipped_already_collected={checkpoint}", flush=True)
            continue
        health = checkpoint_health(checkpoint)
        config = None
        try:
            config, _ = read_checkpoint_metadata(checkpoint)
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            SafetensorError,
        ) as exc:
            print(
                f"warning: checkpoint_metadata_unreadable={checkpoint} "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        identity = checkpoint_identity(config)
        if not health.ok:
            failure_count += 1
            _append_record(
                records_path,
                {
                    "schema_version": 2,
                    "status": "invalid_checkpoint",
                    "created_at": datetime.now(UTC).isoformat(),
                    "command": shlex.join(sys.argv),
                    "checkpoint": str(checkpoint),
                    **identity,
                    "output": str(output),
                    "checkpoint_health": health.to_dict(include_path=False),
                    "error": "checkpoint failed finite/format validation",
                },
            )
            continue
        try:
            collect_rollouts(
                checkpoint,
                output,
                episodes=args.episodes,
                batch_size=args.batch_size,
                device_name=args.device,
                deterministic=args.deterministic,
                opponent_mode="self_play",
                max_steps=args.max_steps,
                degenerate_threshold_fraction=args.degenerate_threshold_fraction,
                analysis_protocol="paper_neural_behavior_v1",
                command=shlex.join(sys.argv),
                seed=args.seed,
            )
        except Exception as exc:
            failure_count += 1
            status = "failed"
            error = repr(exc)
        else:
            status = "ok"
            error = None
        _append_record(
            records_path,
            {
                "schema_version": 2,
                "status": status,
                "created_at": datetime.now(UTC).isoformat(),
                "command": shlex.join(sys.argv),
                "checkpoint": str(checkpoint),
                **identity,
                "output": str(output),
                "checkpoint_health": health.to_dict(include_path=False),
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "device": args.device,
                "deterministic": args.deterministic,
                "seed": args.seed,
                "degenerate_threshold_fraction": args.degenerate_threshold_fraction,
                "error": error,
            },
        )
    if failure_count:
        raise SystemExit(1)


def _collected_checkpoints(records_path: Path) -> set[str]:
    """Checkpoints that already have an ok rollout record."""
    if not records_path.exists():
        return set()
    collected: set[str] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                collected.add(str(record.get("checkpoint")))
    return collected


def _append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _output_stem(path: Path) -> str:
    """Readable stem plus a full-path hash so distinct checkpoints never collide."""
    parts = [part for part in path.with_suffix("").parts if part not in ("", "/")]
    readable = "__".join(parts[-6:]).replace(" ", "_")
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{readable}__{digest}"


if __name__ == "__main__":
    main()

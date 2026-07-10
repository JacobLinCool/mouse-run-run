from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors import SafetensorError

from mouse_run_run.checkpoint_select import checkpoint_identity, select_checkpoints
from mouse_run_run.evaluate import evaluate_checkpoint
from mouse_run_run.health import CheckpointHealth, checkpoint_health
from mouse_run_run.serialization import read_checkpoint_metadata
from mouse_run_run.training_config import DEVICE_CHOICES


PAPER_RANDOM_OPPONENT_MODES = ("random_explorer", "random_chaser")

# Evaluation parameters that determine a record's results, with the parser
# defaults that were in effect before resume keys included them. Records from
# before these fields existed (or with null values) are treated as having been
# produced under those defaults, so completed work under unchanged settings is
# never re-run. The requested --device is deliberately excluded: it names a
# host preference ("auto"), not an evaluation setting.
_RESUME_KEY_FIELDS: tuple[tuple[str, object], ...] = (
    ("episodes", 100),
    ("max_steps", 100),
    ("batch_size", 128),
    ("deterministic", False),
    ("seed", 0),
    ("degenerate_threshold_fraction", 0.01),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="*", default=[Path("runs")])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--degenerate-threshold-fraction", type=float, default=0.01)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed shared by every evaluation, so all checkpoints face the"
        " same standardized random opponent.",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every checkpoint under the given directories, including"
        " mid-training update_* files and failed attempts. Default is the"
        " latest successful attempt per unit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate checkpoints that already have an ok record in the"
        " output file.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    checkpoint_paths = select_checkpoints(args.paths, all_checkpoints=args.all_checkpoints)
    if args.limit is not None:
        checkpoint_paths = checkpoint_paths[: args.limit]
    if not checkpoint_paths:
        raise SystemExit("no checkpoint safetensors found")
    already_evaluated = (
        set()
        if args.force or args.output is None
        else _evaluated_keys(args.output)
    )
    settings = _settings_key(vars(args))

    output_handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("a", encoding="utf-8")
    try:
        failure_count = 0
        for checkpoint in checkpoint_paths:
            pending_modes = [
                mode
                for mode in PAPER_RANDOM_OPPONENT_MODES
                if (str(checkpoint), mode, settings) not in already_evaluated
            ]
            if not pending_modes:
                print(f"skipped_already_evaluated={checkpoint}", flush=True)
                continue
            health = checkpoint_health(checkpoint)
            config = None
            checkpoint_metrics = None
            try:
                config, checkpoint_metrics = read_checkpoint_metadata(checkpoint)
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
            if not health.ok:
                failure_count += 1
                for opponent_mode in pending_modes:
                    record = _base_record(
                        checkpoint=checkpoint,
                        args=args,
                        opponent_mode=opponent_mode,
                        checkpoint_config=config,
                        checkpoint_health=health,
                        status="invalid_checkpoint",
                        error="checkpoint failed finite/format validation",
                    )
                    _write_record(record, output_handle)
                continue
            for opponent_mode in pending_modes:
                try:
                    metrics = evaluate_checkpoint(
                        checkpoint,
                        episodes=args.episodes,
                        batch_size=args.batch_size,
                        device_name=args.device,
                        deterministic=args.deterministic,
                        opponent_mode=opponent_mode,
                        max_steps=args.max_steps,
                        degenerate_threshold_fraction=args.degenerate_threshold_fraction,
                        seed=args.seed,
                    )
                except Exception as exc:
                    failure_count += 1
                    record = _base_record(
                        checkpoint=checkpoint,
                        args=args,
                        opponent_mode=opponent_mode,
                        checkpoint_config=config,
                        checkpoint_metrics=checkpoint_metrics,
                        checkpoint_health=health,
                        status="failed",
                        error=repr(exc),
                    )
                else:
                    record = _base_record(
                        checkpoint=checkpoint,
                        args=args,
                        opponent_mode=opponent_mode,
                        checkpoint_config=config,
                        checkpoint_metrics=checkpoint_metrics,
                        checkpoint_health=health,
                        status="ok",
                        metrics=asdict(metrics),
                    )
                _write_record(record, output_handle)
        if failure_count:
            raise SystemExit(1)
    finally:
        if output_handle is not None:
            output_handle.close()


def _evaluated_keys(output: Path) -> set[tuple[str, str, tuple[object, ...]]]:
    """(checkpoint, opponent_mode, settings) triples with an ok record."""
    if not output.exists():
        return set()
    keys: set[tuple[str, str, tuple[object, ...]]] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") != "ok":
                continue
            keys.add(
                (
                    str(record.get("checkpoint")),
                    str(record.get("opponent_mode")),
                    _settings_key(record),
                )
            )
    return keys


def _settings_key(values: Mapping[str, Any]) -> tuple[object, ...]:
    """Resume-key view of the evaluation settings in a record or args dict."""
    return tuple(
        default if values.get(field) is None else values.get(field)
        for field, default in _RESUME_KEY_FIELDS
    )


def _base_record(
    *,
    checkpoint: Path,
    args: argparse.Namespace,
    opponent_mode: str,
    checkpoint_health: CheckpointHealth,
    status: str,
    checkpoint_config: dict[str, object] | None = None,
    checkpoint_metrics: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "evaluation_protocol": "paper_random_opponent_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "command": shlex.join(sys.argv),
        "checkpoint": str(checkpoint),
        **checkpoint_identity(checkpoint_config),
        "checkpoint_config": checkpoint_config,
        "checkpoint_metrics": checkpoint_metrics,
        "checkpoint_health": checkpoint_health.to_dict(include_path=False),
        "status": status,
        "error": error,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "device": args.device,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "opponent_mode": opponent_mode,
        "focus_agent": "chaser" if opponent_mode == "random_explorer" else "explorer",
        "max_steps": args.max_steps,
        "degenerate_threshold_fraction": args.degenerate_threshold_fraction,
        "metrics": metrics,
    }


def _write_record(record: dict[str, object], output_handle: object | None) -> None:
    line = json.dumps(record, sort_keys=True)
    if output_handle is None:
        print(line, flush=True)
        return
    output_handle.write(line + "\n")
    output_handle.flush()


if __name__ == "__main__":
    main()

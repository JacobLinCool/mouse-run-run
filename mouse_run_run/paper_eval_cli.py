from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mouse_run_run.evaluate import evaluate_checkpoint
from mouse_run_run.health import checkpoint_health
from mouse_run_run.serialization import CHECKPOINT_FORMAT, load_checkpoint, read_metadata
from mouse_run_run.train import DEVICE_CHOICES


PAPER_RANDOM_OPPONENT_MODES = ("random_explorer", "random_chaser")


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
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    checkpoint_paths = list(_checkpoint_paths(args.paths))
    if args.limit is not None:
        checkpoint_paths = checkpoint_paths[: args.limit]
    if not checkpoint_paths:
        raise SystemExit("no checkpoint safetensors found")

    output_handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("a", encoding="utf-8")
    try:
        failure_count = 0
        for checkpoint in checkpoint_paths:
            health = checkpoint_health(checkpoint)
            if not health.ok:
                failure_count += 1
                for opponent_mode in PAPER_RANDOM_OPPONENT_MODES:
                    record = _base_record(
                        checkpoint=checkpoint,
                        args=args,
                        opponent_mode=opponent_mode,
                        checkpoint_health=health,
                        status="invalid_checkpoint",
                        error="checkpoint failed finite/format validation",
                    )
                    _write_record(record, output_handle)
                continue
            config, checkpoint_metrics, _, _ = load_checkpoint(checkpoint)
            for opponent_mode in PAPER_RANDOM_OPPONENT_MODES:
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


def _base_record(
    *,
    checkpoint: Path,
    args: argparse.Namespace,
    opponent_mode: str,
    checkpoint_health: object,
    status: str,
    checkpoint_config: dict[str, object] | None = None,
    checkpoint_metrics: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "evaluation_protocol": "paper_random_opponent_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "command": shlex.join(sys.argv),
        "checkpoint": str(checkpoint),
        "checkpoint_config": checkpoint_config,
        "checkpoint_metrics": checkpoint_metrics,
        "checkpoint_health": {
            "ok": checkpoint_health.ok,
            "format_ok": checkpoint_health.format_ok,
            "tensor_count": checkpoint_health.tensor_count,
            "nonfinite_tensor_count": checkpoint_health.nonfinite_tensor_count,
            "max_abs": checkpoint_health.max_abs,
            "failures": list(checkpoint_health.failures),
        },
        "status": status,
        "error": error,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "device": args.device,
        "deterministic": args.deterministic,
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


def _checkpoint_paths(paths: list[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            _require_checkpoint(path)
            checkpoints.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        for candidate in sorted(path.rglob("*.safetensors")):
            try:
                metadata = read_metadata(candidate)
            except Exception:
                continue
            if metadata.get("format") == CHECKPOINT_FORMAT:
                checkpoints.append(candidate)
    return sorted(dict.fromkeys(checkpoints))


def _require_checkpoint(path: Path) -> None:
    metadata = read_metadata(path)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}: {metadata.get('format')}")


if __name__ == "__main__":
    main()

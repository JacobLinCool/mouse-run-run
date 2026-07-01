from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

from mouse_run_run.health import checkpoint_health
from mouse_run_run.rollout import collect_rollouts
from mouse_run_run.serialization import CHECKPOINT_FORMAT, read_metadata
from mouse_run_run.train import DEVICE_CHOICES


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
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    checkpoints = _checkpoint_paths(args.paths)
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit("no checkpoint safetensors found")

    records_path = args.records or args.output_root / "paper_rollout_records.jsonl"
    failure_count = 0
    for checkpoint in checkpoints:
        health = checkpoint_health(checkpoint)
        output = args.output_root / f"{_safe_stem(checkpoint)}.safetensors"
        if not health.ok:
            failure_count += 1
            _append_record(
                records_path,
                {
                    "schema_version": 1,
                    "status": "invalid_checkpoint",
                    "created_at": datetime.now(UTC).isoformat(),
                    "command": shlex.join(sys.argv),
                    "checkpoint": str(checkpoint),
                    "output": str(output),
                    "checkpoint_health": _health_dict(health),
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
                "schema_version": 1,
                "status": status,
                "created_at": datetime.now(UTC).isoformat(),
                "command": shlex.join(sys.argv),
                "checkpoint": str(checkpoint),
                "output": str(output),
                "checkpoint_health": _health_dict(health),
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "device": args.device,
                "deterministic": args.deterministic,
                "degenerate_threshold_fraction": args.degenerate_threshold_fraction,
                "error": error,
            },
        )
    if failure_count:
        raise SystemExit(1)


def _checkpoint_paths(paths: list[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
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


def _append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _safe_stem(path: Path) -> str:
    parts = [part for part in path.with_suffix("").parts if part not in ("", "/")]
    return "__".join(parts[-6:]).replace(" ", "_")


def _health_dict(health: object) -> dict[str, object]:
    return {
        "ok": health.ok,
        "format_ok": health.format_ok,
        "tensor_count": health.tensor_count,
        "nonfinite_tensor_count": health.nonfinite_tensor_count,
        "max_abs": health.max_abs,
        "failures": list(health.failures),
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mouse_run_run.experiment_runner import (
    ExperimentRunner,
    require_fresh_or_resume,
    validate_args,
)
from mouse_run_run.train import train
from mouse_run_run.training_config import (
    add_training_arguments,
    apply_preset_defaults,
    train_config_from_payload,
)


DEFAULT_UPDATES = 20_000


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--experiment", default="mouse-run-run-0701")
    add_training_arguments(parser, updates=DEFAULT_UPDATES, preset="paper_text", device="cuda")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--tasks", choices=("social", "non_social", "both"), default="both")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=1800.0)
    parser.add_argument("--status-every-seconds", type=float, default=1800.0)
    parser.add_argument("--cost-per-hour", type=float)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--max-total-attempts",
        type=int,
        default=10,
        help=(
            "Upper bound on attempt directories per unit, counting interrupted"
            " attempts. Guards against unbounded crash/interrupt loops."
        ),
    )
    parser.add_argument(
        "--attempt-timeout-hours",
        type=float,
        default=0.0,
        help="Kill a worker after this many wall-clock hours (0 disables).",
    )
    parser.add_argument("--experiment-spec", type=Path, default=Path("experiments/paper_marl_2026/SPEC.md"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Allow --no-resume to run inside a non-empty experiment directory.",
    )
    parser.add_argument("--worker", action="store_true")
    if config_args.config:
        parser.set_defaults(**_validated_config(parser, _load_config(config_args.config)))
    args = parser.parse_args()

    if args.worker:
        _worker()
        return

    apply_preset_defaults(args)
    validate_args(args)
    require_fresh_or_resume(args)
    runner = ExperimentRunner(args, worker_script=Path(__file__).resolve())
    runner.run()


def _worker() -> None:
    payload = json.loads(os.environ["MRR_WORKER_CONFIG"])
    started = time.perf_counter()
    metrics = train(train_config_from_payload(payload))
    print(
        "MRR_JOB_RESULT="
        + json.dumps(
            {
                "task": payload["task"],
                "seed": payload["seed"],
                "unit_id": payload["unit_id"],
                "run_id": payload["run_id"],
                "attempt_id": payload["attempt_id"],
                "run_dir": payload["run_dir"],
                "checkpoint": payload["checkpoint"],
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": asdict(metrics),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return data


def _validated_config(parser: argparse.ArgumentParser, data: dict[str, Any]) -> dict[str, Any]:
    """Validate --config values as argparse would: known keys, types, choices."""
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    unknown = sorted(set(data) - set(actions))
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    validated: dict[str, Any] = {}
    for key, value in data.items():
        action = actions[key]
        if value is not None and action.type is not None:
            if isinstance(value, str):
                try:
                    value = action.type(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"config key {key!r}: cannot parse {value!r} as"
                        f" {action.type.__name__}"
                    ) from exc
            elif action.type is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            elif isinstance(value, bool) or not isinstance(value, action.type):
                raise ValueError(
                    f"config key {key!r}: expected {action.type.__name__},"
                    f" got {type(value).__name__}"
                )
        if isinstance(action, argparse.BooleanOptionalAction) and not isinstance(value, bool):
            raise ValueError(f"config key {key!r}: expected a boolean, got {type(value).__name__}")
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(map(repr, action.choices))
            raise ValueError(f"config key {key!r}: invalid choice {value!r} (choose from {choices})")
        validated[key] = value
    return validated


if __name__ == "__main__":
    main()

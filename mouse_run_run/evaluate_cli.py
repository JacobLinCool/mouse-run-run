import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mouse_run_run.evaluate import evaluate_checkpoint
from mouse_run_run.training_config import DEVICE_CHOICES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--degenerate-threshold-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, help="RNG seed for reproducible evaluation.")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--opponent",
        choices=("self_play", "random_chaser", "random_explorer"),
        default="self_play",
    )
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        batch_size=args.batch_size,
        device_name=args.device,
        deterministic=not args.stochastic,
        opponent_mode=args.opponent,
        max_steps=args.max_steps,
        degenerate_threshold_fraction=args.degenerate_threshold_fraction,
        seed=args.seed,
    )
    print(
        f"episodes={metrics.episodes} "
        f"collisions={metrics.collisions_per_episode:.2f} "
        f"chaser_return={metrics.chaser_return:.2f} "
        f"explorer_return={metrics.explorer_return:.2f} "
        f"vision=({metrics.chaser_partner_vision:.3f},"
        f"{metrics.explorer_partner_vision:.3f}) "
        f"new_fields=({metrics.chaser_new_fields:.1f},"
        f"{metrics.explorer_new_fields:.1f}) "
        f"distance={metrics.final_distance:.2f} "
        f"average_distance={metrics.average_distance:.2f} "
        f"degenerate={metrics.degenerate_episodes}/{metrics.episodes}",
        flush=True,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "command": shlex.join(sys.argv),
            "checkpoint": str(args.checkpoint),
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "device": args.device,
            "deterministic": not args.stochastic,
            "seed": args.seed,
            "opponent_mode": args.opponent,
            "max_steps": args.max_steps,
            "degenerate_threshold_fraction": args.degenerate_threshold_fraction,
            "metrics": asdict(metrics),
        }
        with args.json_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

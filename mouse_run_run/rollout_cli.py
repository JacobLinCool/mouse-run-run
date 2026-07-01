import argparse
import shlex
import sys
from pathlib import Path

from mouse_run_run.rollout import collect_rollouts
from mouse_run_run.train import DEVICE_CHOICES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/rollouts.safetensors"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--paper-analysis", action="store_true")
    parser.add_argument("--degenerate-threshold-fraction", type=float, default=0.01)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--opponent",
        choices=("self_play", "random_chaser", "random_explorer"),
        default="self_play",
    )
    args = parser.parse_args()
    episodes = args.episodes
    max_steps = args.max_steps
    analysis_protocol = "custom_rollout_v2"
    if args.paper_analysis:
        episodes = 25 if episodes is None else episodes
        max_steps = 500 if max_steps is None else max_steps
        analysis_protocol = "paper_neural_behavior_v1"
    else:
        episodes = 128 if episodes is None else episodes

    collect_rollouts(
        args.checkpoint,
        args.output,
        episodes=episodes,
        batch_size=args.batch_size,
        device_name=args.device,
        deterministic=args.deterministic,
        opponent_mode=args.opponent,
        max_steps=max_steps,
        degenerate_threshold_fraction=args.degenerate_threshold_fraction,
        analysis_protocol=analysis_protocol,
        command=shlex.join(sys.argv),
    )

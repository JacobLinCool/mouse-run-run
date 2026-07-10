import argparse
from pathlib import Path

from mouse_run_run.env import GridWorldConfig
from mouse_run_run.train import train
from mouse_run_run.training_config import (
    add_training_arguments,
    apply_preset_defaults,
    config_payload,
    resolve_partner_visibility,
    train_config_from_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_training_arguments(parser, updates=200, preset="modern_fast", device="cpu")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--vision-radius", type=int, default=3)
    parser.add_argument("--task", choices=("social", "non_social"), default="social")
    parser.add_argument("--partner-visibility", choices=("partial", "none", "full"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=0.0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--status-every-seconds", type=float, default=300.0)
    parser.add_argument("--cost-per-hour", type=float)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/marl_ppo.safetensors"),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Checkpoint with training state to resume from.",
    )
    args = parser.parse_args()
    apply_preset_defaults(args)

    env = GridWorldConfig(
        grid_size=args.grid_size,
        vision_radius=args.vision_radius,
        max_steps=args.max_steps,
        task=args.task,
        partner_visibility=resolve_partner_visibility(args.task, args.partner_visibility),
        spawn_mode=args.spawn_mode,
    )
    config = train_config_from_payload(
        config_payload(args),
        seed=args.seed,
        log_every=args.log_every,
        checkpoint=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        checkpoint_every_seconds=args.checkpoint_every_seconds,
        run_dir=args.run_dir,
        metrics_path=args.metrics_path,
        status_path=args.status_path,
        tensorboard_dir=args.tensorboard_dir,
        status_every_seconds=args.status_every_seconds,
        cost_per_hour=args.cost_per_hour,
        resume_from=args.resume_from,
        env=env,
    )
    train(config)

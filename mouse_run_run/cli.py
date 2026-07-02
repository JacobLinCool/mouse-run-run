import argparse
from pathlib import Path

from mouse_run_run.env import GridWorldConfig
from mouse_run_run.train import DEVICE_CHOICES, TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--vision-radius", type=int, default=3)
    parser.add_argument("--task", choices=("social", "non_social"), default="social")
    parser.add_argument("--partner-visibility", choices=("partial", "none", "full"))
    parser.add_argument(
        "--architecture",
        choices=("rnn", "mlp", "ssm", "transformer"),
        default="rnn",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--preset",
        choices=("modern_fast", "paper_text", "official_code"),
        default="modern_fast",
    )
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--value-coef", type=float)
    parser.add_argument(
        "--value-clip",
        type=float,
        default=10.0,
        help="RLlib-style vf_clip_param: per-sample squared value error bound (0 disables).",
    )
    parser.add_argument("--recurrent-l2-coef", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=0.0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--status-every-seconds", type=float, default=300.0)
    parser.add_argument("--cost-per-hour", type=float)
    parser.add_argument("--cuda-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subspace-metric-period", type=int, default=1)
    parser.add_argument("--triton-env-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-agent-rollout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finite-guard", action=argparse.BooleanOptionalAction, default=True)
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
    _apply_preset(args)

    partner_visibility = args.partner_visibility
    if partner_visibility is None:
        partner_visibility = "none" if args.task == "non_social" else "partial"

    env = GridWorldConfig(
        grid_size=args.grid_size,
        vision_radius=args.vision_radius,
        max_steps=args.max_steps,
        task=args.task,
        partner_visibility=partner_visibility,
    )
    config = TrainConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        architecture=args.architecture,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        value_clip=args.value_clip,
        recurrent_l2_coef=args.recurrent_l2_coef,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
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
        cuda_tf32=args.cuda_tf32,
        subspace_metric_period=args.subspace_metric_period,
        triton_env_step=args.triton_env_step,
        fused_agent_rollout=args.fused_agent_rollout,
        finite_guard=args.finite_guard,
        resume_from=args.resume_from,
        env=env,
    )
    train(config)


def _apply_preset(args: argparse.Namespace) -> None:
    defaults = {
        "modern_fast": {
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "recurrent_l2_coef": 0.0,
            "grad_clip": 1.0,
        },
        "paper_text": {
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "recurrent_l2_coef": 0.3,
            "grad_clip": 1.0,
        },
        "official_code": {
            "clip_epsilon": 0.3,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "recurrent_l2_coef": 3.0,
            "grad_clip": 1.0,
        },
    }[args.preset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

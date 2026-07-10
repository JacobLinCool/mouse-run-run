import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch

from mouse_run_run.env import GridWorldConfig, PartnerVisibility, SpawnMode, TaskName
from mouse_run_run.provenance import json_ready


# Source IDs used below are defined with fixed commit/tag permalinks in
# experiments/paper_marl_official_dynamics_2026/SPEC.md,
# "Implementation Source Registry". In particular, [PAPER-METHODS],
# [OFFICIAL-TRAIN], [OFFICIAL-MODEL], and [RAY-PPO-CONFIG] are upstream
# experimental sources; [LOCAL-CALIBRATION] marks our execution choices.
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")
ARCHITECTURE_CHOICES = ("rnn", "mlp", "ssm", "transformer")
SPAWN_MODE_CHOICES = ("full_grid", "official_exclude_last")
PRESET_CHOICES = ("modern_fast", "paper_text", "official_code")
TrainingPreset = Literal["modern_fast", "paper_text", "official_code"]
LearnerMode = Literal["full_batch", "rllib_2_2"]
RNNInitialization = Literal["modern", "pytorch_default"]


@dataclass(frozen=True)
class TrainingPresetDefaults:
    """Complete learner defaults for a named, auditable training regime."""

    learning_rate: float
    gae_lambda: float
    ppo_epochs: int
    clip_epsilon: float
    entropy_coef: float
    value_coef: float
    value_clip: float
    recurrent_l2_coef: float
    grad_clip: float | None
    sgd_minibatch_size: int
    max_seq_len: int
    kl_coeff: float
    kl_target: float
    learner_mode: LearnerMode
    rnn_initialization: RNNInitialization


TRAINING_PRESETS: dict[TrainingPreset, TrainingPresetDefaults] = {
    "modern_fast": TrainingPresetDefaults(
        learning_rate=3e-4,
        gae_lambda=0.95,
        ppo_epochs=4,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        value_clip=10.0,
        recurrent_l2_coef=0.0,
        grad_clip=1.0,
        sgd_minibatch_size=0,
        max_seq_len=100,
        kl_coeff=0.0,
        kl_target=0.01,
        learner_mode="full_batch",
        rnn_initialization="modern",
    ),
    # This preserves the already-completed experiment's interpretation of the
    # Methods-text L2 coefficient while retaining the original fast learner.
    "paper_text": TrainingPresetDefaults(
        learning_rate=3e-4,
        gae_lambda=0.95,
        ppo_epochs=4,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        value_clip=10.0,
        recurrent_l2_coef=0.3,
        grad_clip=1.0,
        sgd_minibatch_size=0,
        max_seq_len=100,
        kl_coeff=0.0,
        kl_target=0.01,
        learner_mode="full_batch",
        rnn_initialization="modern",
    ),
    # The paper says "RLlib default parameters" without enumerating them.
    # [RAY-PPO-CONFIG] fixes those version-sensitive values at Ray 2.2.0;
    # [OFFICIAL-TRAIN] supplies the values explicitly overridden by the release.
    "official_code": TrainingPresetDefaults(
        # [RAY-PPO-CONFIG] PPOConfig defaults inherited by the official script.
        learning_rate=5e-5,
        gae_lambda=1.0,
        ppo_epochs=30,
        # [OFFICIAL-TRAIN] --clip-param=0.3 (also Ray 2.2.0's default).
        clip_epsilon=0.3,
        # [RAY-PPO-CONFIG] inherited loss coefficients and value-error bound.
        entropy_coef=0.0,
        value_coef=1.0,
        value_clip=10.0,
        # [OFFICIAL-TRAIN] --l2-curr=3; [OFFICIAL-MODEL] applies an unsquared
        # Frobenius norm only to rnn.weight_hh_l0.
        recurrent_l2_coef=3.0,
        # [OFFICIAL-TRAIN] explicitly disables clipping; [RAY-PPO-CONFIG]
        # independently has the same default.
        grad_clip=None,
        # [RAY-PPO-CONFIG] recurrent SGD batch size.
        sgd_minibatch_size=128,
        # [OFFICIAL-TRAIN] released model config, not an RLlib default chosen
        # by this port.
        max_seq_len=20,
        # [OFFICIAL-TRAIN] --kl-coeff=0.2; target is [RAY-PPO-CONFIG].
        kl_coeff=0.2,
        kl_target=0.01,
        learner_mode="rllib_2_2",
        # [OFFICIAL-MODEL] constructs nn.RNN/nn.Linear without custom resets.
        rnn_initialization="pytorch_default",
    ),
}


@dataclass(frozen=True)
class TrainConfig:
    updates: int = 200
    batch_size: int = 40
    # [PAPER-METHODS], [OFFICIAL-MODEL]: the reproduction target is a vanilla
    # 256-unit ReLU RNN. MLP/SSM/transformer remain separately labeled analyses.
    architecture: str = "rnn"
    hidden_size: int = 256
    # [RAY-PPO-CONFIG]: inherited AlgorithmConfig default for official_code.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    value_clip: float = 10.0
    recurrent_l2_coef: float = 0.0
    grad_clip: float | None = 1.0
    sgd_minibatch_size: int = 0
    max_seq_len: int = 100
    kl_coeff: float = 0.0
    kl_target: float = 0.01
    learner_mode: LearnerMode = "full_batch"
    rnn_initialization: RNNInitialization = "modern"
    seed: int = 7
    device: str = "cpu"
    log_every: int = 20
    checkpoint: Path = Path("runs/marl_ppo.safetensors")
    checkpoint_every: int = 0
    checkpoint_every_seconds: float = 0.0
    run_dir: Path | None = None
    metrics_path: Path | None = None
    status_path: Path | None = None
    tensorboard_dir: Path | None = None
    status_every_seconds: float = 300.0
    cost_per_hour: float | None = None
    cuda_tf32: bool = True
    subspace_metric_period: int = 1
    triton_env_step: bool = False
    # Run both agents' rollout forwards as one stacked batch. The two policy
    # parameter sets remain independent.
    fused_agent_rollout: bool = False
    finite_guard: bool = True
    experiment_id: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    resume_from: Path | None = None
    env: GridWorldConfig = GridWorldConfig()


def add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    updates: int,
    preset: TrainingPreset | None,
    device: str,
    devices: tuple[str, ...] = DEVICE_CHOICES,
    architectures: tuple[str, ...] = ARCHITECTURE_CHOICES,
    spawn_mode: SpawnMode = "full_grid",
    cuda_tf32: bool = True,
    subspace_metric_period: int = 1,
    triton_env_step: bool = False,
    fused_agent_rollout: bool = False,
) -> None:
    """Register the shared training hyperparameter flags.

    Keyword parameters are the per-command deltas; ``preset=None`` makes
    ``--preset`` required. Learner flags that default to ``None`` are filled
    from the preset by :func:`apply_preset_defaults` after parsing.
    """
    parser.add_argument("--updates", type=int, default=updates)
    # 40 complete 100-step episodes = 4,000 env steps/update [PAPER-METHODS].
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--spawn-mode", choices=SPAWN_MODE_CHOICES, default=spawn_mode)
    parser.add_argument("--architecture", choices=architectures, default="rnn")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--ppo-epochs", type=int)
    if preset is None:
        parser.add_argument("--preset", choices=PRESET_CHOICES, required=True)
    else:
        parser.add_argument("--preset", choices=PRESET_CHOICES, default=preset)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--value-coef", type=float)
    parser.add_argument(
        "--value-clip",
        type=float,
        default=None,
        help="RLlib-style vf_clip_param: per-sample squared value error bound (0 disables).",
    )
    parser.add_argument("--recurrent-l2-coef", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--sgd-minibatch-size", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--kl-coeff", type=float)
    parser.add_argument("--kl-target", type=float)
    parser.add_argument("--learner-mode", choices=("full_batch", "rllib_2_2"))
    parser.add_argument("--rnn-initialization", choices=("modern", "pytorch_default"))
    parser.add_argument("--device", choices=devices, default=device)
    parser.add_argument("--cuda-tf32", action=argparse.BooleanOptionalAction, default=cuda_tf32)
    parser.add_argument("--subspace-metric-period", type=int, default=subspace_metric_period)
    parser.add_argument(
        "--triton-env-step", action=argparse.BooleanOptionalAction, default=triton_env_step
    )
    parser.add_argument(
        "--fused-agent-rollout",
        action=argparse.BooleanOptionalAction,
        default=fused_agent_rollout,
    )
    parser.add_argument("--finite-guard", action=argparse.BooleanOptionalAction, default=True)


def apply_preset_defaults(args: argparse.Namespace, preset: TrainingPreset | None = None) -> None:
    """Fill unset learner hyperparameters from a named training preset."""
    defaults = TRAINING_PRESETS[preset if preset is not None else args.preset]
    for key, value in asdict(defaults).items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def config_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize the flags of add_training_arguments into a JSON-ready payload."""
    return {
        "updates": args.updates,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "spawn_mode": args.spawn_mode,
        "architecture": args.architecture,
        "hidden_size": args.hidden_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "learning_rate": args.learning_rate,
        "ppo_epochs": args.ppo_epochs,
        "preset": args.preset,
        "clip_epsilon": args.clip_epsilon,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "value_clip": args.value_clip,
        "recurrent_l2_coef": args.recurrent_l2_coef,
        "grad_clip": args.grad_clip,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "max_seq_len": args.max_seq_len,
        "kl_coeff": args.kl_coeff,
        "kl_target": args.kl_target,
        "learner_mode": args.learner_mode,
        "rnn_initialization": args.rnn_initialization,
        "device": args.device,
        "cuda_tf32": args.cuda_tf32,
        "subspace_metric_period": args.subspace_metric_period,
        "triton_env_step": args.triton_env_step,
        "fused_agent_rollout": args.fused_agent_rollout,
        "finite_guard": args.finite_guard,
    }


# Payload keys mapped 1:1 onto TrainConfig fields by train_config_from_payload.
_PAYLOAD_CONFIG_KEYS = (
    "updates",
    "batch_size",
    "architecture",
    "hidden_size",
    "gamma",
    "gae_lambda",
    "learning_rate",
    "ppo_epochs",
    "clip_epsilon",
    "entropy_coef",
    "value_coef",
    "value_clip",
    "recurrent_l2_coef",
    "grad_clip",
    "sgd_minibatch_size",
    "max_seq_len",
    "kl_coeff",
    "kl_target",
    "learner_mode",
    "rnn_initialization",
    "seed",
    "device",
    "log_every",
    "checkpoint_every",
    "checkpoint_every_seconds",
    "status_every_seconds",
    "cost_per_hour",
    "cuda_tf32",
    "subspace_metric_period",
    "triton_env_step",
    "fused_agent_rollout",
    "finite_guard",
    "run_id",
    "attempt_id",
)
_PAYLOAD_PATH_KEYS = ("checkpoint", "run_dir", "resume_from")
_PAYLOAD_ENV_KEYS = ("task", "max_steps", "spawn_mode")
# Bookkeeping the orchestrators carry alongside the hyperparameters: the
# provenance-only preset name plus the launching job's attempt identity.
_PAYLOAD_IGNORED_KEYS = ("preset", "unit_id", "attempt", "log")


def train_config_from_payload(payload: Mapping[str, Any], **overrides: Any) -> TrainConfig:
    """Rebuild a TrainConfig from a config_payload-style worker payload.

    Unknown payload keys raise ValueError so typos and drifted key sets fail
    loudly. Keyword overrides must be TrainConfig fields and win over payload
    values; an ``env`` override replaces the payload-derived environment.
    """
    unknown = sorted(
        set(payload)
        - set(_PAYLOAD_CONFIG_KEYS)
        - set(_PAYLOAD_PATH_KEYS)
        - set(_PAYLOAD_ENV_KEYS)
        - set(_PAYLOAD_IGNORED_KEYS)
        - {"experiment"}
    )
    if unknown:
        raise ValueError(f"unknown training payload keys: {', '.join(unknown)}")
    field_names = {field.name for field in fields(TrainConfig)}
    unknown_overrides = sorted(set(overrides) - field_names)
    if unknown_overrides:
        raise ValueError(f"unknown TrainConfig overrides: {', '.join(unknown_overrides)}")

    kwargs: dict[str, Any] = {key: payload[key] for key in _PAYLOAD_CONFIG_KEYS if key in payload}
    for key in ("checkpoint", "run_dir"):
        if key in payload:
            kwargs[key] = Path(payload[key])
    if payload.get("resume_from"):
        kwargs["resume_from"] = Path(payload["resume_from"])
    if "experiment" in payload:
        kwargs["experiment_id"] = payload["experiment"]
    if "env" not in overrides:
        task = payload["task"]
        kwargs["env"] = GridWorldConfig(
            grid_size=10,
            vision_radius=3,
            max_steps=payload["max_steps"],
            task=task,
            partner_visibility=resolve_partner_visibility(task, None),
            spawn_mode=payload["spawn_mode"],
        )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def resolve_partner_visibility(
    task: TaskName,
    partner_visibility: PartnerVisibility | None,
) -> PartnerVisibility:
    """Default non-social runs to hidden partners and social runs to partial vision."""
    if partner_visibility is not None:
        return partner_visibility
    return "none" if task == "non_social" else "partial"


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not _mps_available():
        raise RuntimeError("MPS was requested but is not available.")
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def configure_device_math(device: torch.device, config: TrainConfig) -> None:
    if device.type != "cuda":
        return
    if config.cuda_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        # PyTorch 1.12 (the official environment) disabled TF32 matmul by
        # default while keeping cuDNN TF32 enabled.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True


def checkpoint_config(config: TrainConfig) -> dict[str, object]:
    return json_ready(asdict(config))


def validate_train_config(config: TrainConfig) -> None:
    if config.updates < 1 or config.batch_size < 1 or config.env.max_steps < 1:
        raise ValueError("updates, batch_size, and max_steps must be positive")
    if not 0.0 <= config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if config.learning_rate <= 0 or config.ppo_epochs < 1:
        raise ValueError("learning_rate and ppo_epochs must be positive")
    if config.clip_epsilon <= 0 or config.value_clip < 0:
        raise ValueError("clip_epsilon must be positive and value_clip non-negative")
    if config.entropy_coef < 0 or config.value_coef < 0 or config.recurrent_l2_coef < 0:
        raise ValueError("loss coefficients must be non-negative")
    if config.grad_clip is not None and config.grad_clip <= 0:
        raise ValueError("grad_clip must be positive when enabled")
    if config.max_seq_len < 1 or config.sgd_minibatch_size < 0:
        raise ValueError("max_seq_len must be positive and minibatch size non-negative")
    if config.kl_coeff < 0 or config.kl_target <= 0:
        raise ValueError("kl_coeff must be non-negative and kl_target positive")
    if config.fused_agent_rollout and config.architecture not in ("rnn", "ssm"):
        raise ValueError("fused_agent_rollout is only implemented for rnn and ssm")
    if config.learner_mode == "rllib_2_2":
        if config.architecture != "rnn":
            raise ValueError("rllib_2_2 learner mode requires architecture='rnn'")
        if config.sgd_minibatch_size <= config.max_seq_len:
            raise ValueError("sgd_minibatch_size must be larger than max_seq_len")
        if config.batch_size * config.env.max_steps < config.sgd_minibatch_size:
            raise ValueError("train batch must contain at least one SGD minibatch")

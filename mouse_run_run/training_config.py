from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from mouse_run_run.env import GridWorldConfig, PartnerVisibility, TaskName


# Source IDs used below are defined with fixed commit/tag permalinks in
# experiments/paper_marl_official_dynamics_2026/SPEC.md,
# "Implementation Source Registry". In particular, [PAPER-METHODS],
# [OFFICIAL-TRAIN], [OFFICIAL-MODEL], and [RAY-PPO-CONFIG] are upstream
# experimental sources; [LOCAL-CALIBRATION] marks our execution choices.
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")
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


def apply_preset_defaults(target: object, preset: TrainingPreset | None = None) -> None:
    """Fill unset learner hyperparameters from a named training preset."""
    preset_name = preset or getattr(target, "preset")
    defaults = TRAINING_PRESETS[preset_name]
    for key, value in asdict(defaults).items():
        if getattr(target, key, None) is None:
            setattr(target, key, value)


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
    return _json_ready(asdict(config))


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


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value

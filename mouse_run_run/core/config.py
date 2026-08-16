"""Validated execution configuration shared by experiments and commands."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 1.0
    learning_rate: float = 5e-5
    epochs: int = 30
    minibatch_size: int = 128
    sequence_length: int = 20
    clip_epsilon: float = 0.3
    value_clip: float = 10.0
    entropy_coefficient: float = 0.0
    value_coefficient: float = 1.0
    initial_kl_coefficient: float = 0.2
    kl_target: float = 0.01
    recurrent_l2_coefficient: float = 3.0
    max_gradient_norm: float | None = None

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("PPO gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("PPO GAE lambda must be in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("PPO learning rate must be positive")
        if self.epochs < 1:
            raise ValueError("PPO epochs must be positive")
        if self.minibatch_size < 1 or self.sequence_length < 1:
            raise ValueError("PPO minibatch and recurrent sequence sizes must be positive")
        if self.minibatch_size < self.sequence_length:
            raise ValueError("PPO minibatch_size must be at least sequence_length")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("PPO clip epsilon must be in (0, 1)")
        if self.value_clip < 0.0:
            raise ValueError("PPO value clip must be non-negative")
        if self.entropy_coefficient < 0.0 or self.value_coefficient < 0.0:
            raise ValueError("PPO loss coefficients must be non-negative")
        if self.initial_kl_coefficient < 0.0 or self.kl_target <= 0.0:
            raise ValueError("PPO KL coefficient must be non-negative and target positive")
        if self.recurrent_l2_coefficient < 0.0:
            raise ValueError("PPO recurrent L2 coefficient must be non-negative")
        if self.max_gradient_norm is not None and self.max_gradient_norm <= 0.0:
            raise ValueError("PPO max gradient norm must be positive or None")


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 200
    horizon: int = 100
    checkpoint_every: int = 0
    progress_every: int = 1
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self) -> None:
        if self.updates < 1 or self.horizon < 1:
            raise ValueError("training updates and horizon must be positive")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive")
        self.ppo.validate()

"""Editable PyTorch policy architectures for the chase-grid experiment."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch
from torch import nn

from mouse_run_run.core.experiment import PolicyBuildContext
from mouse_run_run.core.policy import (
    ActivationSite,
    PolicyFeatures,
    PolicyModule,
    PolicyReadout,
    SequenceEvaluation,
)
from mouse_run_run.core.types import TensorMap, TensorTree


ModelKind = Literal["rnn", "cnn_rnn"]
Initialization = Literal["pytorch_default", "modern"]


@dataclass(frozen=True)
class ModelConfig:
    kind: ModelKind = "rnn"
    hidden_size: int = 256
    cnn_channels: tuple[int, int] = (16, 32)
    initialization: Initialization = "pytorch_default"

    def validate(self) -> None:
        if self.kind not in ("rnn", "cnn_rnn"):
            raise ValueError(f"unsupported policy kind: {self.kind!r}")
        if self.hidden_size < 1 or any(channel < 1 for channel in self.cnn_channels):
            raise ValueError("model dimensions must be positive")
        if self.initialization not in ("pytorch_default", "modern"):
            raise ValueError(f"unsupported initialization: {self.initialization!r}")


class RecurrentActorCritic(PolicyModule):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoded_size: int,
        hidden_size: int,
        action_count: int,
        initialization: Initialization,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.cell = nn.RNNCell(encoded_size, hidden_size, nonlinearity="relu")
        self.action_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)
        self._activation_sites = {
            "hidden": ActivationSite(
                name="hidden",
                width=hidden_size,
                analysis=True,
                readout=True,
                recurrent=True,
            )
        }
        if initialization == "modern":
            self.reset_parameters()

    @property
    def activation_sites(self) -> dict[str, ActivationSite]:
        return self._activation_sites

    def reset_parameters(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.kaiming_uniform_(self.cell.weight_ih, nonlinearity="relu")
        nn.init.orthogonal_(self.cell.weight_hh, gain=0.8)
        nn.init.zeros_(self.cell.bias_ih)
        nn.init.zeros_(self.cell.bias_hh)
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.1)
        nn.init.zeros_(self.action_head.bias)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> TensorTree:
        return {"hidden": torch.zeros(batch_size, self.hidden_size, device=device)}

    def advance(self, observation: torch.Tensor, state: TensorTree) -> PolicyFeatures:
        hidden = self.cell(self.encoder(observation), self._state_hidden(state))
        return PolicyFeatures(activations={"hidden": hidden}, next_state={"hidden": hidden})

    def readout(
        self,
        features: PolicyFeatures,
        activation_overrides: TensorMap | None = None,
    ) -> PolicyReadout:
        hidden = self._activation(features, activation_overrides)
        return PolicyReadout(
            logits=self.action_head(hidden),
            value=self.value_head(hidden).squeeze(-1),
        )

    def carry_state(
        self,
        features: PolicyFeatures,
        activation_overrides: TensorMap | None = None,
    ) -> TensorTree:
        return {"hidden": self._activation(features, activation_overrides)}

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        *,
        initial_state: TensorTree | None = None,
    ) -> SequenceEvaluation:
        if observations.ndim < 3:
            raise ValueError("observations must have time and batch axes")
        state = initial_state or self.initial_state(observations.shape[1], observations.device)
        logits: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        hiddens: list[torch.Tensor] = []
        for observation in observations:
            features = self.advance(observation, state)
            readout = self.readout(features)
            state = self.carry_state(features)
            logits.append(readout.logits)
            values.append(readout.value)
            hiddens.append(features.activations["hidden"])
        return SequenceEvaluation(
            logits=torch.stack(logits),
            values=torch.stack(values),
            activations={"hidden": torch.stack(hiddens)},
        )

    def recurrent_weight_norm(self) -> torch.Tensor:
        return self.cell.weight_hh.norm()

    @staticmethod
    def _state_hidden(state: TensorTree) -> torch.Tensor:
        if set(state) != {"hidden"}:
            raise ValueError(f"RNN state must contain only hidden, got {sorted(state)!r}")
        return state["hidden"]

    @staticmethod
    def _activation(
        features: PolicyFeatures,
        overrides: TensorMap | None,
    ) -> torch.Tensor:
        if overrides is not None and set(overrides) - {"hidden"}:
            raise ValueError(f"unknown RNN activation overrides: {sorted(overrides)!r}")
        return features.activations["hidden"] if not overrides else overrides["hidden"]


def build_policy(config: ModelConfig, context: PolicyBuildContext) -> PolicyModule:
    config.validate()
    if config.kind == "rnn":
        encoder: nn.Module = nn.Flatten(start_dim=1)
        encoded_size = prod(context.observation_shape)
    else:
        if len(context.observation_shape) != 3:
            raise ValueError("cnn_rnn requires (channels, height, width) observations")
        channels, height, width = context.observation_shape
        encoder = nn.Sequential(
            nn.Conv2d(channels, config.cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                config.cnn_channels[0], config.cnn_channels[1], kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        encoded_size = config.cnn_channels[1] * height * width
    return RecurrentActorCritic(
        encoder=encoder,
        encoded_size=encoded_size,
        hidden_size=config.hidden_size,
        action_count=context.action_count,
        initialization=config.initialization,
    ).to(context.device)

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor


class RNNActorCritic(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        action_size: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity="relu")
        self.action_layer = nn.Linear(hidden_size, action_size)
        self.value_layer = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.rnn.weight_ih_l0, nonlinearity="relu")
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0, gain=0.8)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.xavier_uniform_(self.action_layer.weight, gain=0.1)
        nn.init.zeros_(self.action_layer.bias)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.zeros_(self.value_layer.bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor) -> PolicyOutput:
        next_hidden = torch.relu(
            F.linear(observation, self.rnn.weight_ih_l0, self.rnn.bias_ih_l0)
            + F.linear(hidden, self.rnn.weight_hh_l0, self.rnn.bias_hh_l0)
        )
        return PolicyOutput(
            logits=self.action_layer(next_hidden),
            value=self.value_layer(next_hidden).squeeze(-1),
            hidden=next_hidden,
        )

    def forward_one_hot_indices(
        self,
        own_index: torch.Tensor,
        other_index: torch.Tensor,
        other_visible: torch.Tensor,
        hidden: torch.Tensor,
    ) -> PolicyOutput:
        input_weight = self.rnn.weight_ih_l0.T
        input_projection = input_weight[own_index] + (
            input_weight[other_index] * other_visible.unsqueeze(1).to(input_weight.dtype)
        )
        next_hidden = torch.relu(
            input_projection
            + self.rnn.bias_ih_l0
            + F.linear(hidden, self.rnn.weight_hh_l0, self.rnn.bias_hh_l0)
        )
        return PolicyOutput(
            logits=self.action_layer(next_hidden),
            value=self.value_layer(next_hidden).squeeze(-1),
            hidden=next_hidden,
        )

    def sequence(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = observations.shape[1]
        hidden = self.initial_hidden(batch_size, observations.device).unsqueeze(0)
        hidden_sequence, final_hidden = self.rnn(observations, hidden)
        return (
            self.action_layer(hidden_sequence),
            self.value_layer(hidden_sequence).squeeze(-1),
            final_hidden.squeeze(0),
        )

    def recurrent_weight_norm(self) -> torch.Tensor:
        return self.rnn.weight_hh_l0.norm()

    def neural_action_subspace(self, hidden: torch.Tensor) -> torch.Tensor:
        action_weight = self.action_layer.weight
        return hidden @ action_weight.T @ action_weight

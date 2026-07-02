from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PolicyOutput:
    """Per-step policy output.

    ``hidden`` is the analysis representation at this step — the vector the
    neural analyses (PLSC, decoding, action-subspace, perturbation) operate
    on and rollouts record. ``state`` is the opaque carrier threaded into the
    next ``forward`` call; for a vanilla RNN they coincide, but e.g. an MLP
    carries its frame-stack buffer while exposing its last activation.
    """

    logits: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor
    state: Any = None


class PolicyBase(nn.Module):
    """Interface shared by all agent architectures.

    Subclasses must set ``architecture`` (checkpoint config tag) and
    ``hidden_size`` (analysis-representation dimension), and implement
    ``initial_hidden`` (initial recurrent state, opaque), ``forward``
    (one step), and ``sequence`` (teacher-forced full episode for the PPO
    update).
    """

    architecture: str = "base"

    def initial_hidden(self, batch_size: int, device: torch.device) -> Any:
        raise NotImplementedError

    def forward(self, observation: torch.Tensor, state: Any) -> PolicyOutput:
        raise NotImplementedError

    def sequence(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def recurrent_weight_norm(self) -> torch.Tensor:
        """L2-penalized recurrent weight norm; zero when no analog exists."""
        return torch.zeros((), device=next(self.parameters()).device)

    def neural_action_subspace(self, hidden: torch.Tensor) -> torch.Tensor:
        action_weight = self.action_layer.weight
        return hidden @ action_weight.T @ action_weight


class RNNActorCritic(PolicyBase):
    architecture = "rnn"

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
            state=next_hidden,
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
            state=next_hidden,
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


class MLPStackActorCritic(PolicyBase):
    """Memoryless control: an MLP over the last ``frame_stack`` observations.

    Its representation is a pure function of the current observation window,
    so any cross-agent "shared dimensions" measured on it establish the
    common-input floor that recurrent architectures must exceed.
    """

    architecture = "mlp"

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        action_size: int = 4,
        frame_stack: int = 8,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.frame_stack = frame_stack
        self.layer1 = nn.Linear(input_size * frame_stack, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.action_layer = nn.Linear(hidden_size, action_size)
        self.value_layer = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer2.bias)
        nn.init.xavier_uniform_(self.action_layer.weight, gain=0.1)
        nn.init.zeros_(self.action_layer.bias)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.zeros_(self.value_layer.bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # State is the rolling observation window (batch, frame_stack, obs).
        return torch.zeros(batch_size, self.frame_stack, self.input_size, device=device)

    def _heads(self, window: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer2(torch.relu(self.layer1(window.flatten(-2)))))

    def forward(self, observation: torch.Tensor, state: torch.Tensor) -> PolicyOutput:
        window = torch.cat([state[:, 1:], observation.unsqueeze(1)], dim=1)
        hidden = self._heads(window)
        return PolicyOutput(
            logits=self.action_layer(hidden),
            value=self.value_layer(hidden).squeeze(-1),
            hidden=hidden,
            state=window,
        )

    def sequence(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps, batch_size = observations.shape[:2]
        padded = torch.cat(
            [
                torch.zeros(
                    self.frame_stack - 1,
                    batch_size,
                    self.input_size,
                    device=observations.device,
                    dtype=observations.dtype,
                ),
                observations,
            ],
            dim=0,
        )
        # (steps, batch, frame_stack, obs): window t covers obs[t-k+1 .. t].
        windows = padded.unfold(0, self.frame_stack, 1).permute(0, 1, 3, 2)
        hidden = self._heads(windows)
        return (
            self.action_layer(hidden),
            self.value_layer(hidden).squeeze(-1),
            hidden[-1],
        )


class SSMActorCritic(PolicyBase):
    """Gated diagonal linear recurrence (LRU/Mamba-style selective state).

    Keeps a compressed recurrent state like the RNN but with *linear* state
    dynamics: h_t = a_t * h_{t-1} + (1 - a_t) * u_t with an input-dependent
    decay a_t in (0, 1). Isolates the nonlinearity of the recurrent dynamics
    as the experimental variable; stability is structural (|a| < 1), so the
    recurrent L2 penalty has no analog here.
    """

    architecture = "ssm"

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        action_size: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_projections = nn.ModuleList(
            [
                nn.Linear(input_size if index == 0 else hidden_size, hidden_size)
                for index in range(layers)
            ]
        )
        self.decay_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )
        self.mix_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )
        self.action_layer = nn.Linear(hidden_size, action_size)
        self.value_layer = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [*self.input_projections, *self.mix_projections]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in self.decay_projections:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            # Bias > 0 starts decays near sigmoid(1) ~ 0.73: remember by default.
            nn.init.ones_(module.bias)
        nn.init.xavier_uniform_(self.action_layer.weight, gain=0.1)
        nn.init.zeros_(self.action_layer.bias)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.zeros_(self.value_layer.bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.layers, batch_size, self.hidden_size, device=device)

    def _cell(
        self,
        x: torch.Tensor,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_states = []
        for index in range(self.layers):
            u = self.input_projections[index](x)
            decay = torch.sigmoid(self.decay_projections[index](u))
            state = decay * states[index] + (1.0 - decay) * u
            next_states.append(state)
            x = torch.relu(self.mix_projections[index](state))
        return x, torch.stack(next_states)

    def forward(self, observation: torch.Tensor, state: torch.Tensor) -> PolicyOutput:
        hidden, next_state = self._cell(observation, state)
        return PolicyOutput(
            logits=self.action_layer(hidden),
            value=self.value_layer(hidden).squeeze(-1),
            hidden=hidden,
            state=next_state,
        )

    def sequence(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps, batch_size = observations.shape[:2]
        state = self.initial_hidden(batch_size, observations.device)
        hiddens = []
        for step in range(steps):
            hidden, state = self._cell(observations[step], state)
            hiddens.append(hidden)
        hidden_sequence = torch.stack(hiddens)
        return (
            self.action_layer(hidden_sequence),
            self.value_layer(hidden_sequence).squeeze(-1),
            hidden_sequence[-1],
        )


class TransformerActorCritic(PolicyBase):
    """Causal transformer over the episode's observation history.

    No evolving recurrent state: the step-t representation is recomputed by
    attention over positions 0..t (KV cache during rollout, parallel causal
    attention in the PPO update). The analysis representation is the final
    pre-head residual at position t.
    """

    architecture = "transformer"
    max_positions = 512

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        action_size: int = 4,
        layers: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position_embedding = nn.Embedding(self.max_positions, hidden_size)
        self.attention_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(layers)])
        self.qkv_projections = nn.ModuleList(
            [nn.Linear(hidden_size, 3 * hidden_size) for _ in range(layers)]
        )
        self.output_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )
        self.mlp_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(layers)])
        self.mlp_in = nn.ModuleList(
            [nn.Linear(hidden_size, mlp_ratio * hidden_size) for _ in range(layers)]
        )
        self.mlp_out = nn.ModuleList(
            [nn.Linear(mlp_ratio * hidden_size, hidden_size) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.action_layer = nn.Linear(hidden_size, action_size)
        self.value_layer = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        for module in [*self.qkv_projections, *self.mlp_in]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in [*self.output_projections, *self.mlp_out]:
            # Scaled-down residual writes stabilize small-model PPO training.
            nn.init.xavier_uniform_(module.weight, gain=0.5 / (2 * self.layers) ** 0.5)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.action_layer.weight, gain=0.1)
        nn.init.zeros_(self.action_layer.bias)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.zeros_(self.value_layer.bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        return {
            "position": 0,
            "keys": [None] * self.layers,
            "values": [None] * self.layers,
        }

    def _attention(
        self,
        x: torch.Tensor,
        index: int,
        state: dict[str, Any] | None,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # x: (tokens, batch, hidden)
        tokens, batch_size = x.shape[:2]
        qkv = self.qkv_projections[index](self.attention_norms[index](x))
        q, k, v = qkv.chunk(3, dim=-1)
        # (batch, heads, tokens, head_dim)
        q = q.view(tokens, batch_size, self.heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(tokens, batch_size, self.heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(tokens, batch_size, self.heads, self.head_dim).permute(1, 2, 0, 3)
        if state is not None:
            if state["keys"][index] is not None:
                k = torch.cat([state["keys"][index], k], dim=2)
                v = torch.cat([state["values"][index], v], dim=2)
            state["keys"][index] = k
            state["values"][index] = v
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_mask,
            is_causal=causal_mask is None and state is None,
        )
        attended = attended.permute(2, 0, 1, 3).reshape(tokens, batch_size, self.hidden_size)
        return self.output_projections[index](attended)

    def _block_mlp(self, x: torch.Tensor, index: int) -> torch.Tensor:
        return self.mlp_out[index](torch.relu(self.mlp_in[index](self.mlp_norms[index](x))))

    def _trunk(
        self,
        observations: torch.Tensor,
        positions: torch.Tensor,
        state: dict[str, Any] | None,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.input_projection(observations) + self.position_embedding(positions).unsqueeze(1)
        for index in range(self.layers):
            x = x + self._attention(x, index, state, causal_mask)
            x = x + self._block_mlp(x, index)
        return self.final_norm(x)

    def forward(self, observation: torch.Tensor, state: dict[str, Any]) -> PolicyOutput:
        position = torch.tensor([state["position"]], device=observation.device)
        hidden = self._trunk(observation.unsqueeze(0), position, state, causal_mask=None)[0]
        state["position"] += 1
        return PolicyOutput(
            logits=self.action_layer(hidden),
            value=self.value_layer(hidden).squeeze(-1),
            hidden=hidden,
            state=state,
        )

    def sequence(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = observations.shape[0]
        positions = torch.arange(steps, device=observations.device)
        mask = torch.ones(steps, steps, dtype=torch.bool, device=observations.device).tril()
        hidden = self._trunk(observations, positions, state=None, causal_mask=mask)
        return (
            self.action_layer(hidden),
            self.value_layer(hidden).squeeze(-1),
            hidden[-1],
        )


ARCHITECTURES = {
    "rnn": RNNActorCritic,
    "mlp": MLPStackActorCritic,
    "ssm": SSMActorCritic,
    "transformer": TransformerActorCritic,
}


def build_policy(
    architecture: str,
    input_size: int,
    hidden_size: int = 256,
) -> PolicyBase:
    try:
        policy_class = ARCHITECTURES[architecture]
    except KeyError:
        raise ValueError(
            f"Unknown architecture {architecture!r}; expected one of {sorted(ARCHITECTURES)}"
        ) from None
    return policy_class(input_size, hidden_size=hidden_size)

import torch

from mouse_run_run.env import ACTION_COUNT, BatchedChaseEnv
from mouse_run_run.health import module_max_abs
from mouse_run_run.policy import PolicyBase, PolicyOutput, RNNActorCritic
from mouse_run_run.training_config import TrainConfig
from mouse_run_run.training_types import (
    AgentRollout,
    Rollout,
    RolloutMetrics,
    empty_metrics,
    tensor_mean,
)


@torch.no_grad()
def collect_rollout(
    *,
    config: TrainConfig,
    env: BatchedChaseEnv,
    chaser: PolicyBase,
    explorer: PolicyBase,
    device: torch.device,
    capture_metrics: bool,
) -> Rollout:
    env.reset_state()
    chaser_hidden = chaser.initial_hidden(config.batch_size, device)
    explorer_hidden = explorer.initial_hidden(config.batch_size, device)
    grid_size = config.env.grid_size
    grid_cells = grid_size * grid_size
    max_steps = config.env.max_steps
    batch_size = config.batch_size

    chaser_position_tensor = torch.empty(max_steps, batch_size, 2, dtype=torch.long, device=device)
    explorer_position_tensor = torch.empty_like(chaser_position_tensor)
    partner_visible_tensor = torch.empty(max_steps, batch_size, dtype=torch.bool, device=device)
    chaser_action_tensor = torch.empty(max_steps, batch_size, dtype=torch.long, device=device)
    explorer_action_tensor = torch.empty_like(chaser_action_tensor)
    chaser_log_prob_tensor = torch.empty(max_steps, batch_size, device=device)
    explorer_log_prob_tensor = torch.empty_like(chaser_log_prob_tensor)
    chaser_logits_tensor = torch.empty(max_steps, batch_size, ACTION_COUNT, device=device)
    explorer_logits_tensor = torch.empty_like(chaser_logits_tensor)
    chaser_value_tensor = torch.empty(max_steps, batch_size, device=device)
    explorer_value_tensor = torch.empty_like(chaser_value_tensor)
    chaser_reward_tensor = torch.empty(max_steps, batch_size, device=device)
    explorer_reward_tensor = torch.empty_like(chaser_reward_tensor)
    done_tensor = torch.empty(max_steps, batch_size, dtype=torch.bool, device=device)

    chaser_state_inputs = None
    explorer_state_inputs = None
    if config.learner_mode == "rllib_2_2":
        if config.architecture != "rnn":
            raise ValueError("rllib_2_2 learner mode requires architecture='rnn'")
        chaser_state_inputs = torch.empty(
            max_steps,
            batch_size,
            config.hidden_size,
            device=device,
        )
        explorer_state_inputs = torch.empty_like(chaser_state_inputs)

    collision_tensor = None
    chaser_new_field_tensor = None
    explorer_new_field_tensor = None
    chaser_partner_visible_tensor = None
    explorer_partner_visible_tensor = None
    distance_tensor = None
    if capture_metrics:
        collision_tensor = torch.empty(max_steps, batch_size, dtype=torch.bool, device=device)
        chaser_new_field_tensor = torch.empty_like(collision_tensor)
        explorer_new_field_tensor = torch.empty_like(collision_tensor)
        chaser_partner_visible_tensor = torch.empty_like(collision_tensor)
        explorer_partner_visible_tensor = torch.empty_like(collision_tensor)
        distance_tensor = torch.empty(max_steps, batch_size, device=device)

    chaser_subspace_norms: list[torch.Tensor] = []
    explorer_subspace_norms: list[torch.Tensor] = []
    record_subspace_metrics = capture_metrics and config.subspace_metric_period > 0
    use_index_path = config.architecture == "rnn"
    fused_pair = None
    if config.fused_agent_rollout:
        if config.architecture not in ("rnn", "ssm"):
            raise ValueError(
                "fused_agent_rollout supports only the rnn and ssm architectures; "
                f"got architecture={config.architecture!r} (disable fused_agent_rollout)"
            )
        fused_class = _FusedAgentPair if config.architecture == "rnn" else _FusedSSMPair
        fused_pair = fused_class(
            chaser,
            explorer,
            observation_size=config.env.observation_size,
            batch_size=batch_size,
            device=device,
        )

    for step_index in range(max_steps):
        chaser_flat_index = _flat_position(env.chaser_position, grid_size)
        explorer_flat_index = _flat_position(env.explorer_position, grid_size)
        partner_visible = env.partner_visible()

        chaser_position_tensor[step_index].copy_(env.chaser_position)
        explorer_position_tensor[step_index].copy_(env.explorer_position)
        partner_visible_tensor[step_index].copy_(partner_visible)

        if chaser_state_inputs is not None:
            assert explorer_state_inputs is not None
            chaser_state_inputs[step_index].copy_(chaser_hidden)
            explorer_state_inputs[step_index].copy_(explorer_hidden)

        if fused_pair is not None:
            chaser_output, explorer_output = fused_pair.step(
                chaser_flat_index,
                explorer_flat_index,
                partner_visible,
            )
        elif use_index_path:
            chaser_output = chaser.forward_one_hot_indices(
                own_index=chaser_flat_index,
                other_index=grid_cells + explorer_flat_index,
                other_visible=partner_visible,
                hidden=chaser_hidden,
            )
            explorer_output = explorer.forward_one_hot_indices(
                own_index=explorer_flat_index,
                other_index=grid_cells + chaser_flat_index,
                other_visible=partner_visible,
                hidden=explorer_hidden,
            )
        else:
            chaser_output = chaser(
                _observation_from_flat(
                    chaser_flat_index, explorer_flat_index, partner_visible, grid_cells
                ),
                chaser_hidden,
            )
            explorer_output = explorer(
                _observation_from_flat(
                    explorer_flat_index, chaser_flat_index, partner_visible, grid_cells
                ),
                explorer_hidden,
            )
        # No per-step finite check here: even the sync-free accumulated-flag
        # variant costs ~24% of rollout wall time in kernel launches, and a
        # non-finite hidden/logit necessarily propagates into the stored
        # log-probs/values that _assert_rollout_finite validates before the
        # optimizer can consume them.
        if config.learner_mode == "rllib_2_2":
            # [OFFICIAL-TRAIN] policy-map order: policy1/explorer samples before
            # policy2/chaser. Source IDs resolve in the official-dynamics SPEC.
            explorer_action, explorer_log_prob = sample_categorical(explorer_output.logits)
            chaser_action, chaser_log_prob = sample_categorical(chaser_output.logits)
        else:
            chaser_action, chaser_log_prob = sample_categorical(chaser_output.logits)
            explorer_action, explorer_log_prob = sample_categorical(explorer_output.logits)

        if config.triton_env_step:
            result = env.step_training_fused(chaser_action, explorer_action)
        else:
            result = env.step_training(chaser_action, explorer_action)

        chaser_action_tensor[step_index].copy_(chaser_action)
        explorer_action_tensor[step_index].copy_(explorer_action)
        chaser_log_prob_tensor[step_index].copy_(chaser_log_prob)
        explorer_log_prob_tensor[step_index].copy_(explorer_log_prob)
        chaser_logits_tensor[step_index].copy_(chaser_output.logits)
        explorer_logits_tensor[step_index].copy_(explorer_output.logits)
        chaser_value_tensor[step_index].copy_(chaser_output.value)
        explorer_value_tensor[step_index].copy_(explorer_output.value)
        chaser_reward_tensor[step_index].copy_(result.chaser_reward)
        explorer_reward_tensor[step_index].copy_(result.explorer_reward)
        done_tensor[step_index].copy_(result.done)
        if capture_metrics:
            assert collision_tensor is not None
            assert chaser_new_field_tensor is not None
            assert explorer_new_field_tensor is not None
            assert chaser_partner_visible_tensor is not None
            assert explorer_partner_visible_tensor is not None
            assert distance_tensor is not None
            collision_tensor[step_index].copy_(result.collision)
            chaser_new_field_tensor[step_index].copy_(result.chaser_new_field)
            explorer_new_field_tensor[step_index].copy_(result.explorer_new_field)
            chaser_partner_visible_tensor[step_index].copy_(result.chaser_partner_visible)
            explorer_partner_visible_tensor[step_index].copy_(result.explorer_partner_visible)
            distance_tensor[step_index].copy_(result.distance)
        if record_subspace_metrics and step_index % config.subspace_metric_period == 0:
            chaser_subspace_norms.append(
                chaser.neural_action_subspace(chaser_output.hidden).norm(dim=1)
            )
            explorer_subspace_norms.append(
                explorer.neural_action_subspace(explorer_output.hidden).norm(dim=1)
            )

        chaser_hidden = chaser_output.state
        explorer_hidden = explorer_output.state

    chaser_advantage, chaser_return = gae(
        rewards=chaser_reward_tensor,
        values=chaser_value_tensor,
        dones=done_tensor,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    explorer_advantage, explorer_return = gae(
        rewards=explorer_reward_tensor,
        values=explorer_value_tensor,
        dones=done_tensor,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    chaser_agent = AgentRollout(
        observations=_build_grid_observations(
            own_positions=chaser_position_tensor,
            other_positions=explorer_position_tensor,
            other_visible=partner_visible_tensor,
            grid_size=grid_size,
            device=device,
        ).detach(),
        actions=chaser_action_tensor.detach(),
        old_log_probs=chaser_log_prob_tensor.detach(),
        old_logits=chaser_logits_tensor.detach(),
        old_values=chaser_value_tensor.detach(),
        rewards=chaser_reward_tensor.detach(),
        # [RAY-SGD-SLICING] standardizes each policy batch independently and
        # clamps the denominator to 1e-4.
        advantages=normalize_advantages(
            chaser_advantage,
            min_std=1e-4 if config.learner_mode == "rllib_2_2" else None,
        ).detach(),
        returns=chaser_return.detach(),
        state_inputs=None if chaser_state_inputs is None else chaser_state_inputs.detach(),
    )
    explorer_agent = AgentRollout(
        observations=_build_grid_observations(
            own_positions=explorer_position_tensor,
            other_positions=chaser_position_tensor,
            other_visible=partner_visible_tensor,
            grid_size=grid_size,
            device=device,
        ).detach(),
        actions=explorer_action_tensor.detach(),
        old_log_probs=explorer_log_prob_tensor.detach(),
        old_logits=explorer_logits_tensor.detach(),
        old_values=explorer_value_tensor.detach(),
        rewards=explorer_reward_tensor.detach(),
        # [RAY-SGD-SLICING], independently for the explorer policy batch.
        advantages=normalize_advantages(
            explorer_advantage,
            min_std=1e-4 if config.learner_mode == "rllib_2_2" else None,
        ).detach(),
        returns=explorer_return.detach(),
        state_inputs=None if explorer_state_inputs is None else explorer_state_inputs.detach(),
    )
    metrics = empty_metrics()
    if capture_metrics:
        assert collision_tensor is not None
        assert chaser_new_field_tensor is not None
        assert explorer_new_field_tensor is not None
        assert chaser_partner_visible_tensor is not None
        assert explorer_partner_visible_tensor is not None
        assert distance_tensor is not None
        metrics = RolloutMetrics(
            collisions_per_episode=collision_tensor.float().sum(dim=0).mean().item(),
            chaser_return=chaser_reward_tensor.sum(dim=0).mean().item(),
            explorer_return=explorer_reward_tensor.sum(dim=0).mean().item(),
            chaser_partner_vision=chaser_partner_visible_tensor.float().mean().item(),
            explorer_partner_vision=explorer_partner_visible_tensor.float().mean().item(),
            chaser_new_fields=chaser_new_field_tensor.float().sum(dim=0).mean().item(),
            explorer_new_fields=explorer_new_field_tensor.float().sum(dim=0).mean().item(),
            final_distance=distance_tensor[-1].mean().item(),
            chaser_subspace_norm=tensor_mean(chaser_subspace_norms),
            explorer_subspace_norm=tensor_mean(explorer_subspace_norms),
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            chaser_grad_norm=0.0,
            explorer_grad_norm=0.0,
            max_param_abs=module_max_abs(chaser, explorer),
            chaser_kl_coeff=config.kl_coeff,
            explorer_kl_coeff=config.kl_coeff,
        )
    return Rollout(chaser=chaser_agent, explorer=explorer_agent, metrics=metrics)


def sample_categorical(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gumbel = -torch.empty_like(logits).exponential_().log()
    action = (logits + gumbel).argmax(dim=-1)
    log_prob = logits.log_softmax(dim=-1).gather(
        dim=-1,
        index=action.unsqueeze(-1),
    ).squeeze(-1)
    return action, log_prob


def gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = (~dones[step]).float()
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[step] = last_advantage
        next_value = values[step]
    return advantages, advantages + values


def normalize_advantages(values: torch.Tensor, *, min_std: float | None = None) -> torch.Tensor:
    std = values.std(unbiased=False)
    denominator = std.clamp_min(min_std) if min_std is not None else std + 1e-8
    return (values - values.mean()) / denominator


def _flat_position(position: torch.Tensor, grid_size: int) -> torch.Tensor:
    return position[:, 0] * grid_size + position[:, 1]


def _observation_from_flat(
    own_flat: torch.Tensor,
    other_flat: torch.Tensor,
    other_visible: torch.Tensor,
    grid_cells: int,
) -> torch.Tensor:
    """Materialize the 2-channel one-hot observation from flat positions."""
    batch_size = own_flat.shape[0]
    observation = torch.zeros(batch_size, 2 * grid_cells, device=own_flat.device)
    batch_index = torch.arange(batch_size, device=own_flat.device)
    observation[batch_index, own_flat] = 1.0
    observation[batch_index, grid_cells + other_flat] = other_visible.to(observation.dtype)
    return observation


class _FusedSSMPair:
    """Both SSM agents' rollout forwards as one stacked batch.

    Mirrors _FusedAgentPair: the first layer's input projection over one-hot
    observations becomes a row gather from a stacked table, every dense layer
    becomes a bmm over the (2, batch, hidden) stack, and the diagonal linear
    recurrence stays elementwise. Weights are snapshotted once per rollout.
    """

    def __init__(
        self,
        chaser,
        explorer,
        *,
        observation_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.observation_size = observation_size
        self.hidden_size = chaser.hidden_size
        self.layers = chaser.layers
        self.input_table = torch.cat(
            [chaser.input_projections[0].weight.T, explorer.input_projections[0].weight.T],
            dim=0,
        ).contiguous()
        self.input_bias0 = torch.stack(
            [chaser.input_projections[0].bias, explorer.input_projections[0].bias]
        ).unsqueeze(1)
        self.deep_input_weights = [
            torch.stack(
                [
                    chaser.input_projections[index].weight.T,
                    explorer.input_projections[index].weight.T,
                ]
            ).contiguous()
            for index in range(1, self.layers)
        ]
        self.deep_input_biases = [
            torch.stack(
                [chaser.input_projections[index].bias, explorer.input_projections[index].bias]
            ).unsqueeze(1)
            for index in range(1, self.layers)
        ]
        self.decay_weights = [
            torch.stack(
                [
                    chaser.decay_projections[index].weight.T,
                    explorer.decay_projections[index].weight.T,
                ]
            ).contiguous()
            for index in range(self.layers)
        ]
        self.decay_biases = [
            torch.stack(
                [chaser.decay_projections[index].bias, explorer.decay_projections[index].bias]
            ).unsqueeze(1)
            for index in range(self.layers)
        ]
        self.mix_weights = [
            torch.stack(
                [
                    chaser.mix_projections[index].weight.T,
                    explorer.mix_projections[index].weight.T,
                ]
            ).contiguous()
            for index in range(self.layers)
        ]
        self.mix_biases = [
            torch.stack(
                [chaser.mix_projections[index].bias, explorer.mix_projections[index].bias]
            ).unsqueeze(1)
            for index in range(self.layers)
        ]
        self.action_weight_t = torch.stack(
            [chaser.action_layer.weight.T, explorer.action_layer.weight.T]
        ).contiguous()
        self.action_bias = torch.stack(
            [chaser.action_layer.bias, explorer.action_layer.bias]
        ).unsqueeze(1)
        self.value_weight_t = torch.stack(
            [chaser.value_layer.weight.T, explorer.value_layer.weight.T]
        ).contiguous()
        self.value_bias = torch.stack(
            [chaser.value_layer.bias, explorer.value_layer.bias]
        ).unsqueeze(1)
        self.states = torch.zeros(self.layers, 2, batch_size, self.hidden_size, device=device)

    def step(
        self,
        chaser_flat_index: torch.Tensor,
        explorer_flat_index: torch.Tensor,
        partner_visible: torch.Tensor,
    ) -> tuple[PolicyOutput, PolicyOutput]:
        observation_size = self.observation_size
        half = observation_size // 2
        own_index = torch.cat([chaser_flat_index, observation_size + explorer_flat_index])
        other_index = torch.cat(
            [half + explorer_flat_index, observation_size + half + chaser_flat_index]
        )
        visible = partner_visible.to(self.input_table.dtype).unsqueeze(1)
        x = (
            self.input_table[own_index] + self.input_table[other_index] * visible.repeat(2, 1)
        ).view(2, -1, self.hidden_size) + self.input_bias0

        next_states = []
        for index in range(self.layers):
            u = (
                x
                if index == 0
                else torch.baddbmm(
                    self.deep_input_biases[index - 1], x, self.deep_input_weights[index - 1]
                )
            )
            decay = torch.sigmoid(
                torch.baddbmm(self.decay_biases[index], u, self.decay_weights[index])
            )
            state = decay * self.states[index] + (1.0 - decay) * u
            next_states.append(state)
            x = torch.relu(torch.baddbmm(self.mix_biases[index], state, self.mix_weights[index]))
        self.states = torch.stack(next_states)

        logits = torch.baddbmm(self.action_bias, x, self.action_weight_t)
        values = torch.baddbmm(self.value_bias, x, self.value_weight_t).squeeze(-1)
        return (
            PolicyOutput(logits=logits[0], value=values[0], hidden=x[0], state=None),
            PolicyOutput(logits=logits[1], value=values[1], hidden=x[1], state=None),
        )


class _FusedAgentPair:
    """Both agents' rollout forwards as one stacked batch.

    The per-agent math is identical to ``forward_one_hot_indices``; stacking
    replaces 2x (gather + addmm + linear + linear) with one gather and three
    bmm calls per step, halving kernel launches in the latency-bound rollout
    loop. Weights are snapshotted once per rollout (they only change in the
    PPO update, which never overlaps a rollout).
    """

    def __init__(
        self,
        chaser: RNNActorCritic,
        explorer: RNNActorCritic,
        *,
        observation_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.observation_size = observation_size
        # (2 * observation_size, hidden): rows [0, obs) are the chaser's
        # W_ih rows, rows [obs, 2*obs) the explorer's.
        self.input_table = torch.cat(
            [chaser.rnn.weight_ih_l0.T, explorer.rnn.weight_ih_l0.T], dim=0
        ).contiguous()
        self.bias_ih = torch.stack([chaser.rnn.bias_ih_l0, explorer.rnn.bias_ih_l0]).unsqueeze(1)
        self.weight_hh_t = torch.stack(
            [chaser.rnn.weight_hh_l0.T, explorer.rnn.weight_hh_l0.T]
        ).contiguous()
        self.bias_hh = torch.stack([chaser.rnn.bias_hh_l0, explorer.rnn.bias_hh_l0]).unsqueeze(1)
        self.action_weight_t = torch.stack(
            [chaser.action_layer.weight.T, explorer.action_layer.weight.T]
        ).contiguous()
        self.action_bias = torch.stack(
            [chaser.action_layer.bias, explorer.action_layer.bias]
        ).unsqueeze(1)
        self.value_weight_t = torch.stack(
            [chaser.value_layer.weight.T, explorer.value_layer.weight.T]
        ).contiguous()
        self.value_bias = torch.stack(
            [chaser.value_layer.bias, explorer.value_layer.bias]
        ).unsqueeze(1)
        self.hidden_size = chaser.hidden_size
        self.hidden = torch.zeros(2, batch_size, self.hidden_size, device=device)

    def step(
        self,
        chaser_flat_index: torch.Tensor,
        explorer_flat_index: torch.Tensor,
        partner_visible: torch.Tensor,
    ) -> tuple[PolicyOutput, PolicyOutput]:
        observation_size = self.observation_size
        half = observation_size // 2
        own_index = torch.cat(
            [chaser_flat_index, observation_size + explorer_flat_index]
        )
        other_index = torch.cat(
            [
                half + explorer_flat_index,
                observation_size + half + chaser_flat_index,
            ]
        )
        visible = partner_visible.to(self.input_table.dtype).unsqueeze(1)
        input_projection = (
            self.input_table[own_index]
            + self.input_table[other_index] * visible.repeat(2, 1)
        ).view(2, -1, self.hidden_size)
        next_hidden = torch.relu(
            input_projection
            + self.bias_ih
            + torch.bmm(self.hidden, self.weight_hh_t)
            + self.bias_hh
        )
        logits = torch.baddbmm(self.action_bias, next_hidden, self.action_weight_t)
        values = torch.baddbmm(self.value_bias, next_hidden, self.value_weight_t).squeeze(-1)
        self.hidden = next_hidden
        return (
            PolicyOutput(
                logits=logits[0],
                value=values[0],
                hidden=next_hidden[0],
                state=next_hidden[0],
            ),
            PolicyOutput(
                logits=logits[1],
                value=values[1],
                hidden=next_hidden[1],
                state=next_hidden[1],
            ),
        )


def _build_grid_observations(
    *,
    own_positions: torch.Tensor,
    other_positions: torch.Tensor,
    other_visible: torch.Tensor,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    steps, batch_size = own_positions.shape[:2]
    observations = torch.zeros(
        steps,
        batch_size,
        2,
        grid_size,
        grid_size,
        device=device,
    )
    step_index = torch.arange(steps, device=device)[:, None].expand(steps, batch_size)
    batch_index = torch.arange(batch_size, device=device)[None, :].expand(steps, batch_size)
    observations[
        step_index,
        batch_index,
        0,
        own_positions[..., 0],
        own_positions[..., 1],
    ] = 1.0
    visible_step = step_index[other_visible]
    visible_batch = batch_index[other_visible]
    visible_other_positions = other_positions[other_visible]
    observations[
        visible_step,
        visible_batch,
        1,
        visible_other_positions[:, 0],
        visible_other_positions[:, 1],
    ] = 1.0
    return observations.flatten(start_dim=2)

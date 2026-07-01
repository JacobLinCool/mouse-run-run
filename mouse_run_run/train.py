from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig
from mouse_run_run.health import (
    NonFiniteTrainingError,
    assert_finite_metrics,
    assert_finite_module,
    assert_finite_tensor,
    assert_finite_tensors,
    module_max_abs,
    require_healthy_checkpoint,
)
from mouse_run_run.observability import TrainingObserver
from mouse_run_run.policy import RNNActorCritic
from mouse_run_run.serialization import save_checkpoint


DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


@dataclass(frozen=True)
class TrainConfig:
    updates: int = 200
    batch_size: int = 40
    hidden_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    recurrent_l2_coef: float = 0.0
    grad_clip: float = 1.0
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
    finite_guard: bool = True
    experiment_id: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    env: GridWorldConfig = GridWorldConfig()


@dataclass(frozen=True)
class RolloutMetrics:
    collisions_per_episode: float
    chaser_return: float
    explorer_return: float
    chaser_partner_vision: float
    explorer_partner_vision: float
    chaser_new_fields: float
    explorer_new_fields: float
    final_distance: float
    chaser_subspace_norm: float
    explorer_subspace_norm: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    grad_norm: float
    max_param_abs: float
    nonfinite_count: float


@dataclass(frozen=True)
class AgentRollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True)
class Rollout:
    chaser: AgentRollout
    explorer: AgentRollout
    metrics: RolloutMetrics


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


StatusCallback = Callable[[dict[str, object]], None]


def train(
    config: TrainConfig,
    *,
    on_status: StatusCallback | None = None,
) -> RolloutMetrics:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    _configure_device_math(device, config)

    env_steps_per_update = config.batch_size * config.env.max_steps
    observer = TrainingObserver(
        run_dir=config.run_dir,
        metrics_path=config.metrics_path,
        status_path=config.status_path,
        tensorboard_dir=config.tensorboard_dir,
        config=_checkpoint_config(config),
        total_updates=config.updates,
        env_steps_per_update=env_steps_per_update,
        status_every_seconds=config.status_every_seconds,
        cost_per_hour=config.cost_per_hour,
        device=str(device),
    )
    env = BatchedChaseEnv(config.env, config.batch_size, device)
    chaser = RNNActorCritic(
        config.env.observation_size,
        hidden_size=config.hidden_size,
    ).to(device)
    explorer = RNNActorCritic(
        config.env.observation_size,
        hidden_size=config.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(
        [*chaser.parameters(), *explorer.parameters()],
        lr=config.learning_rate,
    )
    if config.finite_guard:
        assert_finite_module(chaser, location="initialization", prefix="chaser")
        assert_finite_module(explorer, location="initialization", prefix="explorer")

    latest_metrics = _empty_metrics()
    latest_update = 0
    last_checkpoint_at = perf_counter()
    try:
        for update in range(1, config.updates + 1):
            latest_update = update
            should_log = _should_log(config, update)
            should_checkpoint = _should_checkpoint(config, update, last_checkpoint_at)
            should_capture_metrics = should_log or should_checkpoint or observer.due()
            rollout = _collect_rollout(
                config=config,
                env=env,
                chaser=chaser,
                explorer=explorer,
                device=device,
                capture_metrics=should_capture_metrics,
            )
            if config.finite_guard:
                _assert_rollout_finite(rollout)
            captured_metrics = _ppo_update(
                config=config,
                rollout=rollout,
                chaser=chaser,
                explorer=explorer,
                optimizer=optimizer,
                capture_metrics=should_capture_metrics,
            )
            if captured_metrics is not None:
                latest_metrics = captured_metrics
                if config.finite_guard:
                    assert_finite_metrics(asdict(latest_metrics), location="metrics")

            should_record = should_capture_metrics
            checkpoint_path = None
            if should_checkpoint:
                checkpoint_path = _checkpoint_path_for_update(config, update)
                _save_checkpoint(config, chaser, explorer, latest_metrics, path=checkpoint_path)
                last_checkpoint_at = perf_counter()
                should_record = True

            if should_log:
                _print_metrics(update, latest_metrics)
            if should_record:
                snapshot = observer.record(
                    update=update,
                    metrics=asdict(latest_metrics),
                    checkpoint_path=checkpoint_path,
                )
                if snapshot and on_status:
                    on_status(snapshot)

        final_checkpoint = _save_checkpoint(config, chaser, explorer, latest_metrics)
        snapshot = observer.record(
            update=latest_update,
            metrics=asdict(latest_metrics),
            state="completed",
            checkpoint_path=final_checkpoint,
        )
        if snapshot and on_status:
            on_status(snapshot)
        return latest_metrics
    except BaseException as exc:
        error_name = "nonfinite" if isinstance(exc, NonFiniteTrainingError) else "exception"
        snapshot = observer.record(
            update=latest_update,
            metrics=asdict(latest_metrics),
            state="failed",
            error=f"{error_name}: {repr(exc)}",
        )
        if snapshot and on_status:
            on_status(snapshot)
        raise
    finally:
        observer.close()


def _should_log(config: TrainConfig, update: int) -> bool:
    return update == 1 or update % config.log_every == 0 or update == config.updates


def _print_metrics(update: int, metrics: RolloutMetrics) -> None:
    print(
        f"update={update:04d} "
        f"collisions={metrics.collisions_per_episode:.2f} "
        f"chaser_return={metrics.chaser_return:.2f} "
        f"explorer_return={metrics.explorer_return:.2f} "
        f"vision=({metrics.chaser_partner_vision:.3f},"
        f"{metrics.explorer_partner_vision:.3f}) "
        f"new_fields=({metrics.chaser_new_fields:.1f},"
        f"{metrics.explorer_new_fields:.1f}) "
        f"distance={metrics.final_distance:.2f} "
        f"loss=({metrics.policy_loss:.3f},"
        f"{metrics.value_loss:.3f}) "
        f"kl={metrics.approx_kl:.4f}",
        flush=True,
    )


def _should_checkpoint(
    config: TrainConfig,
    update: int,
    last_checkpoint_at: float,
) -> bool:
    if update == config.updates:
        return False
    if config.checkpoint_every > 0 and update % config.checkpoint_every == 0:
        return True
    return (
        config.checkpoint_every_seconds > 0.0
        and perf_counter() - last_checkpoint_at >= config.checkpoint_every_seconds
    )


def _checkpoint_path_for_update(config: TrainConfig, update: int) -> Path:
    if config.run_dir:
        return config.run_dir / "checkpoints" / f"update_{update:06d}.safetensors"
    return config.checkpoint.with_name(
        f"{config.checkpoint.stem}_update_{update:06d}{config.checkpoint.suffix}"
    )


def _configure_device_math(device: torch.device, config: TrainConfig) -> None:
    if device.type != "cuda":
        return
    if config.cuda_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def _collect_rollout(
    *,
    config: TrainConfig,
    env: BatchedChaseEnv,
    chaser: RNNActorCritic,
    explorer: RNNActorCritic,
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
    chaser_value_tensor = torch.empty(max_steps, batch_size, device=device)
    explorer_value_tensor = torch.empty_like(chaser_value_tensor)
    chaser_reward_tensor = torch.empty(max_steps, batch_size, device=device)
    explorer_reward_tensor = torch.empty_like(chaser_reward_tensor)
    done_tensor = torch.empty(max_steps, batch_size, dtype=torch.bool, device=device)

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

    for step_index in range(max_steps):
        chaser_flat_index = _flat_position(env.chaser_position, grid_size)
        explorer_flat_index = _flat_position(env.explorer_position, grid_size)
        partner_visible = env.partner_visible()

        chaser_position_tensor[step_index].copy_(env.chaser_position)
        explorer_position_tensor[step_index].copy_(env.explorer_position)
        partner_visible_tensor[step_index].copy_(partner_visible)

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
        if config.finite_guard:
            assert_finite_tensors(
                {
                    "chaser_logits": chaser_output.logits,
                    "chaser_value": chaser_output.value,
                    "chaser_hidden": chaser_output.hidden,
                    "explorer_logits": explorer_output.logits,
                    "explorer_value": explorer_output.value,
                    "explorer_hidden": explorer_output.hidden,
                },
                location=f"rollout_step_{step_index}",
            )
        chaser_action, chaser_log_prob = _sample_categorical(chaser_output.logits)
        explorer_action, explorer_log_prob = _sample_categorical(explorer_output.logits)

        if config.triton_env_step:
            result = env.step_training_fused(chaser_action, explorer_action)
        else:
            result = env.step_training(chaser_action, explorer_action)

        chaser_action_tensor[step_index].copy_(chaser_action)
        explorer_action_tensor[step_index].copy_(explorer_action)
        chaser_log_prob_tensor[step_index].copy_(chaser_log_prob)
        explorer_log_prob_tensor[step_index].copy_(explorer_log_prob)
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

        chaser_hidden = chaser_output.hidden
        explorer_hidden = explorer_output.hidden

    chaser_advantage, chaser_return = _gae(
        rewards=chaser_reward_tensor,
        values=chaser_value_tensor,
        dones=done_tensor,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    explorer_advantage, explorer_return = _gae(
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
        old_values=chaser_value_tensor.detach(),
        rewards=chaser_reward_tensor.detach(),
        advantages=_normalize(chaser_advantage).detach(),
        returns=chaser_return.detach(),
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
        old_values=explorer_value_tensor.detach(),
        rewards=explorer_reward_tensor.detach(),
        advantages=_normalize(explorer_advantage).detach(),
        returns=explorer_return.detach(),
    )
    metrics = _empty_metrics()
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
            chaser_subspace_norm=_mean_or_zero(chaser_subspace_norms),
            explorer_subspace_norm=_mean_or_zero(explorer_subspace_norms),
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            grad_norm=0.0,
            max_param_abs=module_max_abs(chaser, explorer),
            nonfinite_count=0.0,
        )
    return Rollout(chaser=chaser_agent, explorer=explorer_agent, metrics=metrics)


def _flat_position(position: torch.Tensor, grid_size: int) -> torch.Tensor:
    return position[:, 0] * grid_size + position[:, 1]


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


def _mean_or_zero(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    return torch.stack(values).mean().item()


def _ppo_update(
    *,
    config: TrainConfig,
    rollout: Rollout,
    chaser: RNNActorCritic,
    explorer: RNNActorCritic,
    optimizer: torch.optim.Optimizer,
    capture_metrics: bool,
) -> RolloutMetrics | None:
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    grad_norms: list[float] = []
    parameters = [*chaser.parameters(), *explorer.parameters()]

    for ppo_epoch in range(config.ppo_epochs):
        chaser_log_prob, chaser_value, chaser_entropy = _evaluate_actions(
            chaser,
            rollout.chaser.observations,
            rollout.chaser.actions,
        )
        explorer_log_prob, explorer_value, explorer_entropy = _evaluate_actions(
            explorer,
            rollout.explorer.observations,
            rollout.explorer.actions,
        )

        chaser_policy_loss = _clipped_policy_loss(
            log_prob=chaser_log_prob,
            old_log_prob=rollout.chaser.old_log_probs,
            advantage=rollout.chaser.advantages,
            clip_epsilon=config.clip_epsilon,
        )
        explorer_policy_loss = _clipped_policy_loss(
            log_prob=explorer_log_prob,
            old_log_prob=rollout.explorer.old_log_probs,
            advantage=rollout.explorer.advantages,
            clip_epsilon=config.clip_epsilon,
        )
        value_loss = 0.5 * (
            (chaser_value - rollout.chaser.returns).square().mean()
            + (explorer_value - rollout.explorer.returns).square().mean()
        )
        entropy = chaser_entropy.mean() + explorer_entropy.mean()
        policy_loss = chaser_policy_loss + explorer_policy_loss
        recurrent_l2 = config.recurrent_l2_coef * (
            chaser.recurrent_weight_norm()
            + explorer.recurrent_weight_norm()
        )
        loss = (
            policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy
            + recurrent_l2
        )
        if config.finite_guard:
            assert_finite_tensors(
                {
                    "chaser_log_prob": chaser_log_prob,
                    "explorer_log_prob": explorer_log_prob,
                    "chaser_value": chaser_value,
                    "explorer_value": explorer_value,
                    "chaser_entropy": chaser_entropy,
                    "explorer_entropy": explorer_entropy,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "recurrent_l2": recurrent_l2,
                    "loss": loss,
                },
                location=f"ppo_epoch_{ppo_epoch}",
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            parameters,
            config.grad_clip,
            error_if_nonfinite=config.finite_guard,
        )
        if config.finite_guard:
            assert_finite_tensor(grad_norm, location=f"ppo_epoch_{ppo_epoch}", name="grad_norm")
        optimizer.step()
        if config.finite_guard:
            assert_finite_module(chaser, location=f"ppo_epoch_{ppo_epoch}", prefix="chaser")
            assert_finite_module(explorer, location=f"ppo_epoch_{ppo_epoch}", prefix="explorer")

        if capture_metrics:
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            grad_norms.append(float(grad_norm.item()))
            with torch.no_grad():
                chaser_kl = rollout.chaser.old_log_probs - chaser_log_prob
                explorer_kl = rollout.explorer.old_log_probs - explorer_log_prob
                approx_kls.append(0.5 * (chaser_kl.mean().item() + explorer_kl.mean().item()))

    if not capture_metrics:
        return None
    return RolloutMetrics(
        collisions_per_episode=rollout.metrics.collisions_per_episode,
        chaser_return=rollout.metrics.chaser_return,
        explorer_return=rollout.metrics.explorer_return,
        chaser_partner_vision=rollout.metrics.chaser_partner_vision,
        explorer_partner_vision=rollout.metrics.explorer_partner_vision,
        chaser_new_fields=rollout.metrics.chaser_new_fields,
        explorer_new_fields=rollout.metrics.explorer_new_fields,
        final_distance=rollout.metrics.final_distance,
        chaser_subspace_norm=rollout.metrics.chaser_subspace_norm,
        explorer_subspace_norm=rollout.metrics.explorer_subspace_norm,
        policy_loss=sum(policy_losses) / len(policy_losses),
        value_loss=sum(value_losses) / len(value_losses),
        entropy=sum(entropies) / len(entropies),
        approx_kl=sum(approx_kls) / len(approx_kls),
        grad_norm=sum(grad_norms) / len(grad_norms),
        max_param_abs=module_max_abs(chaser, explorer),
        nonfinite_count=0.0,
    )


def _evaluate_actions(
    policy: RNNActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values, _ = policy.sequence(observations)
    log_probs = logits.log_softmax(dim=-1)
    action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return action_log_probs, values, entropy


def _sample_categorical(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gumbel = -torch.empty_like(logits).exponential_().log()
    action = (logits + gumbel).argmax(dim=-1)
    log_prob = logits.log_softmax(dim=-1).gather(
        dim=-1,
        index=action.unsqueeze(-1),
    ).squeeze(-1)
    return action, log_prob


def _clipped_policy_loss(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    ratio = (log_prob - old_log_prob).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()


def _gae(
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


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)


def _empty_metrics() -> RolloutMetrics:
    return RolloutMetrics(
        collisions_per_episode=0.0,
        chaser_return=0.0,
        explorer_return=0.0,
        chaser_partner_vision=0.0,
        explorer_partner_vision=0.0,
        chaser_new_fields=0.0,
        explorer_new_fields=0.0,
        final_distance=0.0,
        chaser_subspace_norm=0.0,
        explorer_subspace_norm=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        approx_kl=0.0,
        grad_norm=0.0,
        max_param_abs=0.0,
        nonfinite_count=0.0,
    )


def _save_checkpoint(
    config: TrainConfig,
    chaser: RNNActorCritic,
    explorer: RNNActorCritic,
    metrics: RolloutMetrics,
    *,
    path: Path | None = None,
) -> Path | None:
    checkpoint = path or config.checkpoint
    if not checkpoint:
        return None
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        checkpoint,
        config=_checkpoint_config(config),
        metrics=asdict(metrics),
        chaser_state=chaser.state_dict(),
        explorer_state=explorer.state_dict(),
    )
    if config.finite_guard:
        require_healthy_checkpoint(checkpoint)
    print(f"saved_checkpoint={checkpoint}", flush=True)
    return checkpoint


def _assert_rollout_finite(rollout: Rollout) -> None:
    assert_finite_tensors(
        {
            "chaser_old_log_probs": rollout.chaser.old_log_probs,
            "chaser_old_values": rollout.chaser.old_values,
            "chaser_rewards": rollout.chaser.rewards,
            "chaser_advantages": rollout.chaser.advantages,
            "chaser_returns": rollout.chaser.returns,
            "explorer_old_log_probs": rollout.explorer.old_log_probs,
            "explorer_old_values": rollout.explorer.old_values,
            "explorer_rewards": rollout.explorer.rewards,
            "explorer_advantages": rollout.explorer.advantages,
            "explorer_returns": rollout.explorer.returns,
        },
        location="rollout",
    )


def _checkpoint_config(config: TrainConfig) -> dict[str, object]:
    return _json_ready(asdict(config))


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value

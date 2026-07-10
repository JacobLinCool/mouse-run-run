from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from mouse_run_run.env import BatchedChaseEnv
from mouse_run_run.health import (
    NonFiniteTrainingError,
    assert_finite_metrics,
    assert_finite_module,
    assert_finite_tensors,
    require_healthy_checkpoint,
)
from mouse_run_run.observability import TrainingObserver
from mouse_run_run.policy import PolicyBase, build_policy
from mouse_run_run.ppo import ppo_update
from mouse_run_run.serialization import TrainingState, load_checkpoint, load_training_state, save_checkpoint
from mouse_run_run.training_config import (
    TrainConfig,
    checkpoint_config,
    configure_device_math,
    select_device,
    validate_train_config,
)
from mouse_run_run.training_rollout import collect_rollout
from mouse_run_run.training_types import LearnerState, Rollout, RolloutMetrics, empty_metrics


StatusCallback = Callable[[dict[str, object]], None]


def train(
    config: TrainConfig,
    *,
    on_status: StatusCallback | None = None,
) -> RolloutMetrics:
    validate_train_config(config)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    configure_device_math(device, config)

    env_steps_per_update = config.batch_size * config.env.max_steps
    observer = TrainingObserver(
        run_dir=config.run_dir,
        metrics_path=config.metrics_path,
        status_path=config.status_path,
        tensorboard_dir=config.tensorboard_dir,
        config=checkpoint_config(config),
        total_updates=config.updates,
        env_steps_per_update=env_steps_per_update,
        status_every_seconds=config.status_every_seconds,
        cost_per_hour=config.cost_per_hour,
        device=str(device),
    )
    env = BatchedChaseEnv(config.env, config.batch_size, device)
    def make_policy() -> PolicyBase:
        return build_policy(
            config.architecture,
            config.env.observation_size,
            hidden_size=config.hidden_size,
            rnn_initialization=config.rnn_initialization,
        ).to(device)

    if config.learner_mode == "rllib_2_2":
        # [OFFICIAL-TRAIN] inserts policy1/explorer before policy2/chaser. The
        # released model uses PyTorch defaults, so this order determines which
        # seeded parameter draw belongs to each role. Source IDs resolve in the
        # official-dynamics SPEC's "Implementation Source Registry".
        explorer = make_policy()
        chaser = make_policy()
    else:
        chaser = make_policy()
        explorer = make_policy()
    # [OFFICIAL-TRAIN] defines two PPOTorchPolicy instances. RLlib policy
    # ownership means separate parameters, Adam optimizer moments, and
    # [RAY-KL-ADAPT] state; no weights or optimizer state are shared here.
    chaser_optimizer = torch.optim.Adam(chaser.parameters(), lr=config.learning_rate)
    explorer_optimizer = torch.optim.Adam(explorer.parameters(), lr=config.learning_rate)
    learner_state = LearnerState(
        chaser_kl_coeff=config.kl_coeff,
        explorer_kl_coeff=config.kl_coeff,
    )
    minibatch_generator = torch.Generator(device="cpu")
    minibatch_generator.manual_seed(config.seed + 1_000_003)
    if config.finite_guard:
        assert_finite_module(chaser, location="initialization", prefix="chaser")
        assert_finite_module(explorer, location="initialization", prefix="explorer")

    start_update = 1
    if config.resume_from is not None:
        start_update = 1 + _restore_training_state(
            config=config,
            chaser=chaser,
            explorer=explorer,
            chaser_optimizer=chaser_optimizer,
            explorer_optimizer=explorer_optimizer,
            learner_state=learner_state,
            minibatch_generator=minibatch_generator,
            device=device,
        )

    latest_metrics = empty_metrics()
    latest_update = start_update - 1
    last_checkpoint_at = perf_counter()
    try:
        for update in range(start_update, config.updates + 1):
            latest_update = update
            should_log = _should_log(config, update)
            should_checkpoint = _should_checkpoint(config, update, last_checkpoint_at)
            should_capture_metrics = should_log or should_checkpoint or observer.due()
            rollout = collect_rollout(
                config=config,
                env=env,
                chaser=chaser,
                explorer=explorer,
                device=device,
                capture_metrics=should_capture_metrics,
            )
            if config.finite_guard:
                _assert_rollout_finite(rollout)
            captured_metrics = ppo_update(
                config=config,
                rollout=rollout,
                chaser=chaser,
                explorer=explorer,
                chaser_optimizer=chaser_optimizer,
                explorer_optimizer=explorer_optimizer,
                learner_state=learner_state,
                minibatch_generator=minibatch_generator,
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
                _save_checkpoint(
                    config,
                    chaser,
                    explorer,
                    latest_metrics,
                    path=checkpoint_path,
                    chaser_optimizer=chaser_optimizer,
                    explorer_optimizer=explorer_optimizer,
                    learner_state=learner_state,
                    minibatch_generator=minibatch_generator,
                    update=update,
                    device=device,
                )
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

        final_checkpoint = _save_checkpoint(
            config,
            chaser,
            explorer,
            latest_metrics,
            chaser_optimizer=chaser_optimizer,
            explorer_optimizer=explorer_optimizer,
            learner_state=learner_state,
            minibatch_generator=minibatch_generator,
            update=latest_update,
            device=device,
        )
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
        f"kl_coeff=({metrics.chaser_kl_coeff:.4g},"
        f"{metrics.explorer_kl_coeff:.4g})",
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


def _save_checkpoint(
    config: TrainConfig,
    chaser: PolicyBase,
    explorer: PolicyBase,
    metrics: RolloutMetrics,
    *,
    path: Path | None = None,
    chaser_optimizer: torch.optim.Optimizer | None = None,
    explorer_optimizer: torch.optim.Optimizer | None = None,
    learner_state: LearnerState | None = None,
    minibatch_generator: torch.Generator | None = None,
    update: int | None = None,
    device: torch.device | None = None,
) -> Path | None:
    checkpoint = path or config.checkpoint
    if not checkpoint:
        return None
    training_state = None
    if (
        chaser_optimizer is not None
        and explorer_optimizer is not None
        and learner_state is not None
        and minibatch_generator is not None
        and update is not None
    ):
        training_state = TrainingState(
            update=update,
            optimizer_states={
                "chaser": chaser_optimizer.state_dict(),
                "explorer": explorer_optimizer.state_dict(),
            },
            learner_state=asdict(learner_state),
            cpu_rng_state=torch.get_rng_state(),
            minibatch_rng_state=minibatch_generator.get_state(),
            cuda_rng_state=(
                torch.cuda.get_rng_state(device)
                if device is not None and device.type == "cuda"
                else None
            ),
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        checkpoint,
        config=checkpoint_config(config),
        metrics=asdict(metrics),
        chaser_state=chaser.state_dict(),
        explorer_state=explorer.state_dict(),
        training_state=training_state,
    )
    if config.finite_guard:
        require_healthy_checkpoint(checkpoint)
    print(f"saved_checkpoint={checkpoint}", flush=True)
    return checkpoint


def _restore_training_state(
    *,
    config: TrainConfig,
    chaser: PolicyBase,
    explorer: PolicyBase,
    chaser_optimizer: torch.optim.Optimizer,
    explorer_optimizer: torch.optim.Optimizer,
    learner_state: LearnerState,
    minibatch_generator: torch.Generator,
    device: torch.device,
) -> int:
    """Load model/optimizer/RNG state from a checkpoint; return its update index."""
    path = Path(config.resume_from)  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    saved_config, _, chaser_state, explorer_state = load_checkpoint(path)
    current_config = checkpoint_config(config)
    resume_keys = (
        "env",
        "batch_size",
        "architecture",
        "hidden_size",
        "seed",
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
    )
    for key in resume_keys:
        if saved_config.get(key) != current_config.get(key):
            raise ValueError(
                f"resume checkpoint {path} was trained with {key}="
                f"{saved_config.get(key)!r}, current config has "
                f"{current_config.get(key)!r}"
            )
    training_state = load_training_state(path)
    if training_state is None:
        raise ValueError(f"checkpoint {path} has no training state; cannot resume")
    if training_state.update >= config.updates:
        raise ValueError(
            f"checkpoint {path} is already at update {training_state.update}"
            f" >= configured updates {config.updates}"
        )
    chaser.load_state_dict(chaser_state)
    explorer.load_state_dict(explorer_state)
    chaser_optimizer.load_state_dict(training_state.optimizer_states["chaser"])
    explorer_optimizer.load_state_dict(training_state.optimizer_states["explorer"])
    learner_state.chaser_kl_coeff = training_state.learner_state["chaser_kl_coeff"]
    learner_state.explorer_kl_coeff = training_state.learner_state["explorer_kl_coeff"]
    torch.set_rng_state(training_state.cpu_rng_state)
    minibatch_generator.set_state(training_state.minibatch_rng_state)
    if device.type == "cuda" and training_state.cuda_rng_state is not None:
        torch.cuda.set_rng_state(training_state.cuda_rng_state, device)
    print(f"resumed_from={path} update={training_state.update}", flush=True)
    return training_state.update


def _assert_rollout_finite(rollout: Rollout) -> None:
    assert_finite_tensors(
        {
            "chaser_old_log_probs": rollout.chaser.old_log_probs,
            "chaser_old_logits": rollout.chaser.old_logits,
            "chaser_old_values": rollout.chaser.old_values,
            "chaser_rewards": rollout.chaser.rewards,
            "chaser_advantages": rollout.chaser.advantages,
            "chaser_returns": rollout.chaser.returns,
            "explorer_old_log_probs": rollout.explorer.old_log_probs,
            "explorer_old_logits": rollout.explorer.old_logits,
            "explorer_old_values": rollout.explorer.old_values,
            "explorer_rewards": rollout.explorer.rewards,
            "explorer_advantages": rollout.explorer.advantages,
            "explorer_returns": rollout.explorer.returns,
        },
        location="rollout",
    )
    if rollout.chaser.state_inputs is not None:
        assert rollout.explorer.state_inputs is not None
        assert_finite_tensors(
            {
                "chaser_state_inputs": rollout.chaser.state_inputs,
                "explorer_state_inputs": rollout.explorer.state_inputs,
            },
            location="rollout_state_inputs",
        )

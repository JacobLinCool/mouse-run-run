import argparse

import pytest

from mouse_run_run.training_config import (
    TrainConfig,
    add_training_arguments,
    apply_preset_defaults,
    config_payload,
    train_config_from_payload,
)


def _parsed_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_training_arguments(parser, updates=200, preset="modern_fast", device="cpu")
    args = parser.parse_args(argv or [])
    apply_preset_defaults(args)
    return args


def test_worker_payload_round_trip_matches_direct_construction() -> None:
    args = _parsed_args(["--preset", "official_code"])
    payload = {**config_payload(args), "task": "social", "seed": 5}

    config = train_config_from_payload(payload)

    assert config == TrainConfig(
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
        sgd_minibatch_size=args.sgd_minibatch_size,
        max_seq_len=args.max_seq_len,
        kl_coeff=args.kl_coeff,
        kl_target=args.kl_target,
        learner_mode=args.learner_mode,
        rnn_initialization=args.rnn_initialization,
        seed=5,
        device=args.device,
        cuda_tf32=args.cuda_tf32,
        subspace_metric_period=args.subspace_metric_period,
        triton_env_step=args.triton_env_step,
        fused_agent_rollout=args.fused_agent_rollout,
        finite_guard=args.finite_guard,
        env=config.env,
    )
    assert config.env.task == "social"
    assert config.env.max_steps == args.max_steps
    assert config.env.spawn_mode == args.spawn_mode


def test_payload_architecture_defaults_to_rnn() -> None:
    payload = {"task": "social", "max_steps": 100, "spawn_mode": "full_grid"}

    config = train_config_from_payload(payload)

    assert config.architecture == "rnn"


def test_unknown_payload_keys_raise_with_key_names() -> None:
    payload = {
        "task": "social",
        "max_steps": 100,
        "spawn_mode": "full_grid",
        "leraning_rate": 3e-4,
        "banana": 1,
    }

    with pytest.raises(ValueError, match="banana, leraning_rate"):
        train_config_from_payload(payload)


def test_unknown_overrides_raise_with_key_names() -> None:
    payload = {"task": "social", "max_steps": 100, "spawn_mode": "full_grid"}

    with pytest.raises(ValueError, match="learnin_rate"):
        train_config_from_payload(payload, learnin_rate=1e-4)

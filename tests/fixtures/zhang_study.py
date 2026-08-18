"""Tiny study fixtures: a behavior sweep and a gate-emitting confirmatory run."""

from experiments.zhang_2025_rnn.study import _make_definition
from mouse_run_run.core.config import PPOConfig


development_smoke_definition = _make_definition(
    name="zhang-2025-rnn-development-smoke",
    purpose="development",
    enabled_stages=("train", "behavior"),
    seeds=(0,),
    updates=2,
    checkpoint_every=1,
    behavior_checkpoints=(1, 2),
    neural_checkpoints=(2,),
    ppo=PPOConfig(
        epochs=1,
        minibatch_size=16,
        sequence_length=4,
        recurrent_l2_coefficient=0.0,
    ),
    smoke=True,
)


# Confirmatory purpose is what turns the scientific gates on, so this fixture is
# the only one that exercises the gate readers.
confirmatory_smoke_definition = _make_definition(
    name="zhang-2025-rnn-confirmatory-smoke",
    purpose="confirmatory",
    enabled_stages=("train", "behavior", "neural", "causal"),
    seeds=(0, 1),
    updates=1,
    checkpoint_every=1,
    behavior_checkpoints=(1,),
    neural_checkpoints=(1,),
    ppo=PPOConfig(
        epochs=1,
        minibatch_size=16,
        sequence_length=4,
        recurrent_l2_coefficient=0.0,
    ),
    smoke=True,
)

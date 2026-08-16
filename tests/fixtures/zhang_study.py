"""Tiny behavior-checkpoint sweep fixture."""

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

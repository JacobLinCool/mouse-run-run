"""Goal-directed Zhang et al. (2025) RNN study definitions."""

from __future__ import annotations

from dataclasses import asdict

from experiments.chase_grid.environment import (
    AGENT_IDS,
    CHASER,
    EXPLORER,
    ChaseGridConfig,
    GridRenderer,
    make_environment,
)
from experiments.chase_grid.model import ModelConfig, build_policy
from experiments.chase_grid.reward import RewardConfig
from mouse_run_run.core.config import PPOConfig, TrainingConfig
from mouse_run_run.core.experiment import Experiment
from mouse_run_run.studies.types import (
    BehaviorRecipe,
    CausalRecipe,
    Condition,
    DecoderRecipe,
    NeuralRecipe,
    PLSCRecipe,
    StudyDefinition,
    StudyPurpose,
    StudyStageName,
    Visibility,
)


CONFIRMATORY_UPDATE = 2_000

GOAL_DIRECTED_PPO = PPOConfig(
    learning_rate=5e-5,
    gamma=0.99,
    gae_lambda=1.0,
    epochs=1,
    minibatch_size=4_000,
    sequence_length=20,
    clip_epsilon=0.3,
    value_coefficient=1.0,
    value_clip=10.0,
    entropy_coefficient=0.0,
    initial_kl_coefficient=0.2,
    kl_target=0.01,
    recurrent_l2_coefficient=0.3,
    max_gradient_norm=None,
)

PROTOCOL_PPO = PPOConfig(
    learning_rate=5e-5,
    gamma=0.99,
    gae_lambda=1.0,
    epochs=30,
    minibatch_size=128,
    sequence_length=20,
    clip_epsilon=0.3,
    value_coefficient=1.0,
    value_clip=10.0,
    entropy_coefficient=0.0,
    initial_kl_coefficient=0.2,
    kl_target=0.01,
    recurrent_l2_coefficient=3.0,
    max_gradient_norm=None,
)


def _make_definition(
    *,
    name: str,
    purpose: StudyPurpose,
    enabled_stages: tuple[StudyStageName, ...],
    seeds: tuple[int, ...],
    updates: int,
    checkpoint_every: int,
    behavior_checkpoints: tuple[int, ...],
    neural_checkpoints: tuple[int, ...],
    ppo: PPOConfig,
    smoke: bool = False,
) -> StudyDefinition:
    hidden_size = 8 if smoke else 256
    episodes_per_update = 2 if smoke else 40
    steps_per_episode = 8 if smoke else 100

    def build(
        condition: Condition,
        seed: int,
        horizon: int,
        visibility: Visibility,
    ) -> Experiment:
        environment_config = ChaseGridConfig(
            grid_size=10,
            vision_radius=3,
            max_steps=horizon,
            partner_visibility=visibility,
            spawn_mode="exclude_last",
            reward=RewardConfig(task=condition),
        )
        model = ModelConfig(
            kind="rnn",
            hidden_size=hidden_size,
            initialization="pytorch_default",
        )
        training = TrainingConfig(
            updates=updates,
            horizon=steps_per_episode,
            checkpoint_every=checkpoint_every,
            progress_every=1 if smoke else max(1, checkpoint_every // 5),
            ppo=ppo,
        )
        experiment = Experiment(
            name=f"{name}_{condition}",
            agent_ids=AGENT_IDS,
            observation_shapes={CHASER: (2, 10, 10), EXPLORER: (2, 10, 10)},
            action_counts={CHASER: 4, EXPLORER: 4},
            make_environment=lambda runtime, device: make_environment(
                environment_config, runtime, device
            ),
            policy_factories={
                CHASER: lambda context: build_policy(model, context),
                EXPLORER: lambda context: build_policy(model, context),
            },
            training=training,
            renderer=GridRenderer(environment_config),
            metadata={
                "study": name,
                "purpose": purpose,
                "condition": condition,
                "seed": seed,
                "evaluation_horizon": horizon,
                "visibility": visibility,
                "model": asdict(model),
                "environment": asdict(environment_config),
            },
        )
        experiment.validate()
        return experiment

    definition = StudyDefinition(
        name=name,
        purpose=purpose,
        enabled_stages=enabled_stages,
        conditions=("social", "non_social"),
        seeds=seeds,
        updates=updates,
        episodes_per_update=episodes_per_update,
        steps_per_episode=steps_per_episode,
        checkpoint_every=checkpoint_every,
        hidden_size=hidden_size,
        ppo=ppo,
        behavior=BehaviorRecipe(
            checkpoint_updates=behavior_checkpoints,
            episodes=2 if smoke else 100,
            horizon=steps_per_episode,
            batch_size=2 if smoke else 100,
        ),
        neural=NeuralRecipe(
            checkpoint_updates=neural_checkpoints,
            episodes=4 if smoke else 25,
            horizon=8 if smoke else 500,
            batch_size=2 if smoke else 25,
            non_social_visibility_controls=("none", "partial", "full"),
            plsc=PLSCRecipe(
                permutations=4 if smoke else 2_000,
                percentile=0.75 if smoke else 0.975,
                min_shift=2 if smoke else 60,
                null_batch_size=2 if smoke else 4,
            ),
            decoder=DecoderRecipe(
                folds=2 if smoke else 5,
                shuffled_controls=2 if smoke else 200,
            ),
        ),
        causal=CausalRecipe(
            episodes=2 if smoke else 100,
            horizon=8 if smoke else 100,
            batch_size=2 if smoke else 100,
            shared_rank=2 if smoke else 10,
            random_control_rank=2 if smoke else 25,
        ),
        build_experiment=build,
        source=(
            "Goal-directed artificial-agent replication of Zhang et al. Nature "
            "2025; DOI 10.1038/s41586-025-09196-4"
        ),
    )
    definition.validate()
    return definition


# Development uses behavior only. A checkpoint is chosen globally from this
# sweep, recorded in CONFIRMATORY_UPDATE, and then frozen before confirmation.
development_definition = _make_definition(
    name="zhang-2025-rnn-development",
    purpose="development",
    enabled_stages=("train", "behavior"),
    seeds=(0, 1, 2),
    updates=5_000,
    checkpoint_every=500,
    behavior_checkpoints=tuple(range(500, 5_001, 500)),
    neural_checkpoints=(5_000,),
    ppo=GOAL_DIRECTED_PPO,
)


# The canonical study uses fresh seeds and one predeclared checkpoint for every
# inferential stage. Hyperparameters are selected for the target phenomena,
# not for optimizer-level fidelity to the released RLlib program.
definition = _make_definition(
    name="zhang-2025-rnn-confirmatory",
    purpose="confirmatory",
    enabled_stages=("train", "behavior", "neural", "causal"),
    seeds=tuple(range(100, 110)),
    updates=CONFIRMATORY_UPDATE,
    checkpoint_every=500,
    behavior_checkpoints=(CONFIRMATORY_UPDATE,),
    neural_checkpoints=(CONFIRMATORY_UPDATE,),
    ppo=GOAL_DIRECTED_PPO,
)


# This expensive definition answers a different question: sensitivity to the
# paper/released-code optimization protocol. It is not the default study.
protocol_definition = _make_definition(
    name="zhang-2025-rnn-protocol-sensitivity",
    purpose="protocol_sensitivity",
    enabled_stages=("train", "behavior", "neural", "causal"),
    seeds=tuple(range(10)),
    updates=20_000,
    checkpoint_every=1_000,
    behavior_checkpoints=(20_000,),
    neural_checkpoints=tuple(range(15_000, 20_001, 1_000)),
    ppo=PROTOCOL_PPO,
)


smoke_definition = _make_definition(
    name="zhang-2025-rnn-smoke",
    purpose="smoke",
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

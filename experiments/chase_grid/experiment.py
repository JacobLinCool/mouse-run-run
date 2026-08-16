"""Composition root for the default chase-grid research experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from experiments.chase_grid.environment import (
    AGENT_IDS,
    CHASER,
    EXPLORER,
    ChaseGridConfig,
    GridRenderer,
    make_environment,
)
from experiments.chase_grid.model import ModelConfig, build_policy
from mouse_run_run.core.config import PPOConfig, TrainingConfig
from mouse_run_run.core.experiment import Experiment, ExperimentDefinition


@dataclass(frozen=True)
class ChaseExperimentConfig:
    environment: ChaseGridConfig = field(default_factory=ChaseGridConfig)
    chaser_model: ModelConfig = field(default_factory=ModelConfig)
    explorer_model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            updates=200,
            horizon=100,
            checkpoint_every=20,
            progress_every=1,
            ppo=PPOConfig(),
        )
    )


def build_experiment(config: ChaseExperimentConfig = ChaseExperimentConfig()) -> Experiment:
    grid = config.environment.grid_size
    experiment = Experiment(
        name="chase_grid",
        agent_ids=AGENT_IDS,
        observation_shapes={CHASER: (2, grid, grid), EXPLORER: (2, grid, grid)},
        action_counts={CHASER: 4, EXPLORER: 4},
        make_environment=lambda runtime, device: make_environment(
            config.environment, runtime, device
        ),
        policy_factories={
            CHASER: lambda context: build_policy(config.chaser_model, context),
            EXPLORER: lambda context: build_policy(config.explorer_model, context),
        },
        training=config.training,
        renderer=GridRenderer(config.environment),
        metadata={
            "definition": "experiments.chase_grid.experiment:definition",
            "config": asdict(config),
        },
    )
    experiment.validate()
    return experiment


definition = ExperimentDefinition(config=ChaseExperimentConfig(), build=build_experiment)
experiment = definition
default_experiment = definition.resolve()

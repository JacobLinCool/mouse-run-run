"""Renderer-only experiment values for strict saved study rollouts."""

from experiments.zhang_2025_rnn.study import definition
from mouse_run_run.studies.types import StudyUnit


social_experiment = definition.experiment(
    StudyUnit("social", definition.seeds[0]),
    horizon=definition.causal.horizon,
)
non_social_experiment = definition.experiment(
    StudyUnit("non_social", definition.seeds[0]),
    horizon=definition.causal.horizon,
)

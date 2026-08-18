"""Validated contracts for a reproducible train-to-causal study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from mouse_run_run.core.config import PPOConfig
from mouse_run_run.core.experiment import Experiment


Condition = Literal["social", "non_social"]
Visibility = Literal["none", "partial", "full"]
StudyPurpose = Literal[
    "development",
    "confirmatory",
    "protocol_sensitivity",
    "smoke",
]
StudyStageName = Literal["train", "behavior", "neural", "causal"]


@dataclass(frozen=True)
class BehaviorRecipe:
    checkpoint_updates: tuple[int, ...]
    episodes: int
    horizon: int
    batch_size: int


@dataclass(frozen=True)
class PLSCRecipe:
    permutations: int
    percentile: float
    min_shift: int
    null_batch_size: int


@dataclass(frozen=True)
class DecoderRecipe:
    folds: int
    shuffled_controls: int


@dataclass(frozen=True)
class NeuralRecipe:
    checkpoint_updates: tuple[int, ...]
    episodes: int
    horizon: int
    batch_size: int
    non_social_visibility_controls: tuple[Visibility, ...]
    plsc: PLSCRecipe
    decoder: DecoderRecipe


@dataclass(frozen=True)
class CausalRecipe:
    episodes: int
    horizon: int
    batch_size: int
    shared_rank: int
    random_control_max_rank: int
    """Upper bound on the control subspace; its rank is matched on removed variance."""


@dataclass(frozen=True)
class StudyUnit:
    condition: Condition
    seed: int

    @property
    def unit_id(self) -> str:
        return f"{self.condition}.seed_{self.seed:04d}"


ExperimentBuilder = Callable[[Condition, int, int, Visibility], Experiment]


@dataclass(frozen=True)
class StudyDefinition:
    name: str
    purpose: StudyPurpose
    enabled_stages: tuple[StudyStageName, ...]
    conditions: tuple[Condition, ...]
    seeds: tuple[int, ...]
    updates: int
    episodes_per_update: int
    steps_per_episode: int
    checkpoint_every: int
    hidden_size: int
    ppo: PPOConfig
    behavior: BehaviorRecipe
    neural: NeuralRecipe
    causal: CausalRecipe
    build_experiment: ExperimentBuilder
    source: str

    @property
    def units(self) -> tuple[StudyUnit, ...]:
        return tuple(
            StudyUnit(condition, seed)
            for condition in self.conditions
            for seed in self.seeds
        )

    @property
    def environment_steps_per_unit(self) -> int:
        return self.updates * self.episodes_per_update * self.steps_per_episode

    @property
    def total_environment_steps(self) -> int:
        return len(self.units) * self.environment_steps_per_unit

    @property
    def emits_scientific_gates(self) -> bool:
        return self.purpose in ("confirmatory", "protocol_sensitivity")

    @property
    def optimizer_minibatches_per_agent_update(self) -> int:
        segments = (
            self.episodes_per_update
            * self.steps_per_episode
            // self.ppo.sequence_length
        )
        segments_per_minibatch = self.ppo.minibatch_size // self.ppo.sequence_length
        return (segments + segments_per_minibatch - 1) // segments_per_minibatch

    @property
    def optimizer_steps_per_update(self) -> int:
        return self.optimizer_minibatches_per_agent_update * self.ppo.epochs * 2

    @property
    def optimizer_steps_per_unit(self) -> int:
        return self.optimizer_steps_per_update * self.updates

    @property
    def total_optimizer_steps(self) -> int:
        return self.optimizer_steps_per_unit * len(self.units)

    def experiment(
        self,
        unit: StudyUnit,
        *,
        horizon: int | None = None,
        visibility: Visibility | None = None,
    ) -> Experiment:
        selected_visibility: Visibility = visibility or (
            "partial" if unit.condition == "social" else "none"
        )
        experiment = self.build_experiment(
            unit.condition,
            unit.seed,
            self.steps_per_episode if horizon is None else horizon,
            selected_visibility,
        )
        experiment.validate()
        return experiment

    def validate(self) -> None:
        if not self.name or not self.name.replace("-", "_").isidentifier():
            raise ValueError("study name must be a non-empty identifier")
        if self.purpose not in (
            "development",
            "confirmatory",
            "protocol_sensitivity",
            "smoke",
        ):
            raise ValueError("study purpose is invalid")
        if not self.source:
            raise ValueError("study source must be non-empty")
        if self.conditions != ("social", "non_social"):
            raise ValueError("Zhang study conditions must be social then non_social")
        if not self.enabled_stages or len(set(self.enabled_stages)) != len(
            self.enabled_stages
        ):
            raise ValueError("study enabled stages must be unique and non-empty")
        canonical_stages = ("train", "behavior", "neural", "causal")
        if any(stage not in canonical_stages for stage in self.enabled_stages):
            raise ValueError("study has an unknown enabled stage")
        if tuple(
            stage for stage in canonical_stages if stage in self.enabled_stages
        ) != self.enabled_stages:
            raise ValueError("study enabled stages must preserve canonical order")
        if self.enabled_stages[0] != "train":
            raise ValueError("study must enable training before downstream stages")
        if self.purpose == "development" and self.enabled_stages != (
            "train",
            "behavior",
        ):
            raise ValueError("development studies may only train and evaluate behavior")
        if self.emits_scientific_gates and self.enabled_stages != canonical_stages:
            raise ValueError("scientific studies require every evidence stage")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("study seeds must be unique non-negative integers")
        for value, label in (
            (self.updates, "updates"),
            (self.episodes_per_update, "episodes_per_update"),
            (self.steps_per_episode, "steps_per_episode"),
            (self.checkpoint_every, "checkpoint_every"),
            (self.hidden_size, "hidden_size"),
        ):
            if value < 1:
                raise ValueError(f"study {label} must be positive")
        if self.updates % self.checkpoint_every:
            raise ValueError("checkpoint_every must divide study updates")
        self.ppo.validate()
        if self.steps_per_episode % self.ppo.sequence_length:
            raise ValueError(
                "study horizon must divide exactly into recurrent sequence lengths"
            )
        if not self.behavior.checkpoint_updates:
            raise ValueError("behavior recipe requires checkpoint updates")
        if not self.neural.checkpoint_updates:
            raise ValueError("neural recipe requires checkpoint updates")
        for checkpoints, label in (
            (self.behavior.checkpoint_updates, "behavior"),
            (self.neural.checkpoint_updates, "neural"),
        ):
            if tuple(sorted(set(checkpoints))) != checkpoints:
                raise ValueError(f"{label} checkpoints must be sorted and unique")
            if any(
                update < 1
                or update > self.updates
                or update % self.checkpoint_every
                for update in checkpoints
            ):
                raise ValueError(
                    f"{label} checkpoints must exist in the training schedule"
                )
        if self.emits_scientific_gates and len(self.behavior.checkpoint_updates) != 1:
            raise ValueError(
                "scientific behavior gates require one predeclared checkpoint"
            )
        if self.purpose == "confirmatory" and (
            self.behavior.checkpoint_updates != self.neural.checkpoint_updates
            or len(self.neural.checkpoint_updates) != 1
            or self.neural.checkpoint_updates[0] != self.updates
        ):
            raise ValueError(
                "confirmatory analyses require the same fixed final checkpoint"
            )
        if self.emits_scientific_gates and (
            self.neural.checkpoint_updates[-1]
            != self.behavior.checkpoint_updates[0]
        ):
            raise ValueError(
                "scientific behavior and causal evidence must use the same final checkpoint"
            )
        for recipe in (self.behavior, self.neural, self.causal):
            if recipe.episodes < 1 or recipe.horizon < 1 or recipe.batch_size < 1:
                raise ValueError("study rollout recipes require positive sizes")
        if self.neural.plsc.min_shift * 2 >= self.neural.horizon:
            raise ValueError("neural horizon must exceed twice the PLSC minimum shift")
        if self.causal.shared_rank + self.causal.random_control_max_rank > self.hidden_size:
            raise ValueError("causal shared and control ranks must fit hidden width")
        if self.neural.non_social_visibility_controls != ("none", "partial", "full"):
            raise ValueError("non-social controls must be none, partial, full")
        for unit in self.units:
            experiment = self.experiment(unit)
            if experiment.training.updates != self.updates:
                raise ValueError("study builder changed the declared update count")
            if experiment.training.horizon != self.steps_per_episode:
                raise ValueError("study builder changed the declared training horizon")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "enabled_stages": list(self.enabled_stages),
            "conditions": list(self.conditions),
            "seeds": list(self.seeds),
            "updates": self.updates,
            "episodes_per_update": self.episodes_per_update,
            "steps_per_episode": self.steps_per_episode,
            "checkpoint_every": self.checkpoint_every,
            "hidden_size": self.hidden_size,
            "ppo": asdict(self.ppo),
            "behavior": asdict(self.behavior),
            "neural": asdict(self.neural),
            "causal": asdict(self.causal),
            "source": self.source,
        }

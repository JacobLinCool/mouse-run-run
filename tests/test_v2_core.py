from __future__ import annotations

from dataclasses import dataclass

import torch

from mouse_run_run.core.environment import EnvReset, EnvTransition
from mouse_run_run.core.intervention import InterventionContext, InterventionPipeline
from mouse_run_run.core.policy import (
    ActivationSite,
    PolicyFeatures,
    PolicyModule,
    PolicyReadout,
    SequenceEvaluation,
)
from mouse_run_run.core.simulation import SimulationConfig, SimulationEngine
from mouse_run_run.core.types import AgentId, TensorMap, TensorTree


AGENTS = (AgentId("alpha"), AgentId("beta"), AgentId("gamma"))


class DummyEnvironment:
    def __init__(self, batch_size: int = 2) -> None:
        self._batch_size = batch_size
        self._device = torch.device("cpu")
        self.step_index = 0

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        return AGENTS

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self, *, generator: torch.Generator) -> EnvReset:
        self.step_index = 0
        return EnvReset(
            observations={
                agent: torch.ones(self.batch_size, index + 1)
                for index, agent in enumerate(AGENTS)
            },
            world={"step": torch.zeros(self.batch_size, dtype=torch.long)},
        )

    def step(self, actions: dict[AgentId, torch.Tensor]) -> EnvTransition:
        self.step_index += 1
        done = torch.full((self.batch_size,), self.step_index >= 3)
        return EnvTransition(
            observations={
                agent: torch.ones(self.batch_size, index + 1)
                for index, agent in enumerate(AGENTS)
            },
            rewards={agent: actions[agent].float() for agent in AGENTS},
            terminated=torch.zeros_like(done),
            truncated=done,
            active=torch.ones_like(done),
            world={"step": torch.full((self.batch_size,), self.step_index)},
            events={"tick": torch.ones(self.batch_size, dtype=torch.bool)},
        )


class DummyPolicy(PolicyModule):
    def __init__(self, observation_width: int, hidden_width: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(observation_width, hidden_width, bias=False)
        torch.nn.init.ones_(self.projection.weight)
        self.head = torch.nn.Linear(hidden_width, 2, bias=False)
        with torch.no_grad():
            self.head.weight[0].fill_(1.0)
            self.head.weight[1].fill_(-1.0)
        self._sites = {
            "hidden": ActivationSite("hidden", hidden_width, readout=True, recurrent=True)
        }

    @property
    def activation_sites(self) -> dict[str, ActivationSite]:
        return self._sites

    def initial_state(self, batch_size: int, device: torch.device) -> TensorTree:
        return {"hidden": torch.zeros(batch_size, self.head.in_features)}

    def advance(self, observation: torch.Tensor, state: TensorTree) -> PolicyFeatures:
        hidden = state["hidden"] + self.projection(observation)
        return PolicyFeatures({"hidden": hidden}, {"hidden": hidden})

    def readout(
        self, features: PolicyFeatures, activation_overrides: TensorMap | None = None
    ) -> PolicyReadout:
        hidden = (
            features.activations["hidden"]
            if activation_overrides is None
            else activation_overrides["hidden"]
        )
        return PolicyReadout(self.head(hidden), hidden.mean(dim=-1))

    def carry_state(
        self, features: PolicyFeatures, activation_overrides: TensorMap | None = None
    ) -> TensorTree:
        return {
            "hidden": (
                features.activations["hidden"]
                if activation_overrides is None
                else activation_overrides["hidden"]
            )
        }

    def evaluate_sequence(
        self, observations: torch.Tensor, *, initial_state: TensorTree | None = None
    ) -> SequenceEvaluation:
        state = initial_state or self.initial_state(observations.shape[1], observations.device)
        logits, values, hidden = [], [], []
        for observation in observations:
            features = self.advance(observation, state)
            readout = self.readout(features)
            state = self.carry_state(features)
            logits.append(readout.logits)
            values.append(readout.value)
            hidden.append(features.activations["hidden"])
        return SequenceEvaluation(
            torch.stack(logits), torch.stack(values), {"hidden": torch.stack(hidden)}
        )


@dataclass(frozen=True)
class NegateIntervention:
    target: str
    name: str = "negate"
    agent_ids: frozenset[AgentId] = frozenset({AGENTS[0]})
    site: str = "hidden"

    def apply(self, value: torch.Tensor, context: InterventionContext) -> torch.Tensor:
        return -value


def _engine() -> SimulationEngine:
    policies = {
        agent: DummyPolicy(index + 1, index + 2)
        for index, agent in enumerate(AGENTS)
    }
    return SimulationEngine(DummyEnvironment(), policies)


def test_simulation_supports_three_heterogeneous_agents_and_t_plus_one_world() -> None:
    batch = _engine().collect(
        SimulationConfig(horizon=3, deterministic=True, record_activations=True, record_world=True)
    )
    assert batch.agent_ids == AGENTS
    assert batch.world["step"].shape == (4, 2)
    assert batch.events["tick"].shape == (3, 2)
    assert [batch.agents[agent].activations["hidden"].shape[-1] for agent in AGENTS] == [
        2,
        3,
        4,
    ]


def test_readout_and_recurrent_interventions_have_distinct_causal_semantics() -> None:
    baseline = _engine().collect(
        SimulationConfig(horizon=3, deterministic=True, record_activations=True)
    )
    readout = _engine().collect(
        SimulationConfig(horizon=3, deterministic=True, record_activations=True),
        interventions=InterventionPipeline((NegateIntervention("readout"),)),
    )
    recurrent = _engine().collect(
        SimulationConfig(horizon=3, deterministic=True, record_activations=True),
        interventions=InterventionPipeline((NegateIntervention("recurrent"),)),
    )
    agent = AGENTS[0]
    assert torch.equal(
        readout.agents[agent].activations["hidden"],
        baseline.agents[agent].activations["hidden"],
    )
    assert not torch.equal(readout.agents[agent].actions, baseline.agents[agent].actions)
    assert not torch.equal(
        recurrent.agents[agent].activations["hidden"],
        baseline.agents[agent].activations["hidden"],
    )
    assert "raw.hidden" in recurrent.agents[agent].activations
    assert "readout.hidden" in readout.agents[agent].activations

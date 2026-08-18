"""The one stepping loop used by training, rollout, and interventions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mouse_run_run.core.environment import MultiAgentEnvironment
from mouse_run_run.core.intervention import InterventionPipeline
from mouse_run_run.core.policy import PolicyModule
from mouse_run_run.core.types import AgentId, TensorMap, require_agent_keys, require_tensor_tree


@dataclass(frozen=True)
class SimulationConfig:
    horizon: int
    deterministic: bool = False
    seed: int = 0
    action_seed: int | None = None
    record_activations: bool = False
    record_world: bool = False


@dataclass(frozen=True)
class AgentTrajectory:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    bootstrap_value: torch.Tensor
    activations: TensorMap = field(default_factory=dict)
    state_inputs: TensorMap = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryBatch:
    agent_ids: tuple[AgentId, ...]
    agents: dict[AgentId, AgentTrajectory]
    terminated: torch.Tensor
    truncated: torch.Tensor
    active: torch.Tensor
    world: TensorMap
    events: TensorMap
    seed: int
    episode_seeds: torch.Tensor

    @property
    def horizon(self) -> int:
        return self.terminated.shape[0]

    @property
    def batch_size(self) -> int:
        return self.terminated.shape[1]


def episode_behavior_summary(trajectory: TrajectoryBatch) -> dict[str, torch.Tensor]:
    """Per-episode behaviour columns shared by rollout tables and training metrics.

    Event counts are summed over the horizon, boolean events also become the
    fraction of active steps they held, and a recorded ``distance`` world signal
    contributes its active-step mean and final value.
    """

    columns: dict[str, torch.Tensor] = {}
    active_count = trajectory.active.sum(dim=0).clamp_min(1)
    for key, value in trajectory.events.items():
        if value.ndim != 2:
            continue
        columns[f"event.{key}"] = value.to(torch.float64).sum(dim=0)
        if value.dtype == torch.bool:
            columns[f"event_fraction.{key}"] = (
                (value & trajectory.active).sum(dim=0) / active_count
            )
    if "distance" in trajectory.world:
        distance = trajectory.world["distance"][:-1]
        columns["world.distance_mean"] = (
            (distance * trajectory.active).sum(dim=0) / active_count
        )
        columns["world.distance_final"] = trajectory.world["distance"][-1]
    return columns


class SimulationEngine:
    def __init__(
        self,
        environment: MultiAgentEnvironment,
        policies: dict[AgentId, PolicyModule],
    ) -> None:
        self.environment = environment
        self.policies = policies

    @torch.no_grad()
    def collect(
        self,
        config: SimulationConfig,
        *,
        interventions: InterventionPipeline | None = None,
    ) -> TrajectoryBatch:
        if config.horizon < 1:
            raise ValueError("simulation horizon must be positive")
        if tuple(self.policies) != self.environment.agent_ids:
            raise ValueError("policy keys must exactly match environment agent_ids")
        if len({id(policy) for policy in self.policies.values()}) != len(self.policies):
            raise ValueError("v2 requires an independent policy instance per agent")
        pipeline = interventions or InterventionPipeline()
        pipeline.validate(self.policies)
        device = self.environment.device
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device).manual_seed(config.seed)
        action_generator = torch.Generator(device=generator_device).manual_seed(
            config.seed if config.action_seed is None else config.action_seed
        )
        reset = self.environment.reset(generator=generator)
        require_agent_keys(reset.observations, self.environment.agent_ids, label="reset observations")
        _validate_observations(reset.observations, batch_size=self.environment.batch_size)
        _validate_tensor_map(reset.world, batch_size=self.environment.batch_size, label="reset world")
        observations = reset.observations
        batch_size = self.environment.batch_size
        episode_indices = torch.arange(batch_size, device=device)
        states = {
            agent_id: policy.initial_state(batch_size, device)
            for agent_id, policy in self.policies.items()
        }
        for agent_id, state in states.items():
            require_tensor_tree(state, label=f"initial_state[{agent_id}]")

        agent_records: dict[AgentId, dict[str, object]] = {
            agent_id: {
                "observations": [],
                "actions": [],
                "rewards": [],
                "logits": [],
                "values": [],
                "log_probs": [],
                "activations": {},
                "state_inputs": {},
            }
            for agent_id in self.environment.agent_ids
        }
        terminated: list[torch.Tensor] = []
        truncated: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        world: dict[str, list[torch.Tensor]] = (
            {key: [value] for key, value in reset.world.items()}
            if config.record_world
            else {}
        )
        events: dict[str, list[torch.Tensor]] = {}

        for step in range(config.horizon):
            actions: dict[AgentId, torch.Tensor] = {}
            next_states: dict[AgentId, dict[str, torch.Tensor]] = {}
            for agent_id in self.environment.agent_ids:
                policy = self.policies[agent_id]
                state_records = agent_records[agent_id]["state_inputs"]
                assert isinstance(state_records, dict)
                for key, value in states[agent_id].items():
                    state_records.setdefault(key, []).append(value)
                features = policy.advance(observations[agent_id], states[agent_id])
                require_tensor_tree(features.next_state, label=f"next_state[{agent_id}]")
                missing = set(policy.activation_sites) - features.activations.keys()
                unknown = features.activations.keys() - set(policy.activation_sites)
                if missing or unknown:
                    raise ValueError(
                        f"policy {agent_id} activation mismatch; "
                        f"missing={sorted(missing)}, unknown={sorted(unknown)}"
                    )
                for name, value in features.activations.items():
                    site = policy.activation_sites[name]
                    if value.ndim != 2 or value.shape != (batch_size, site.width):
                        raise ValueError(
                            f"policy {agent_id} activation {name!r} must have shape "
                            f"{(batch_size, site.width)}, got {tuple(value.shape)}"
                        )
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(
                            f"policy {agent_id} activation {name!r} is non-finite"
                        )

                recurrent_overrides: TensorMap = {}
                readout_overrides: TensorMap = {}
                for name, site in policy.activation_sites.items():
                    raw = features.activations[name]
                    recurrent_value = raw
                    if site.recurrent:
                        recurrent_value = pipeline.apply(
                            agent_id=agent_id,
                            site=site,
                            target="recurrent",
                            step=step,
                            episode_indices=episode_indices,
                            value=raw,
                        )
                        if recurrent_value is not raw:
                            recurrent_overrides[name] = recurrent_value
                    readout_value = recurrent_value
                    if site.readout:
                        readout_value = pipeline.apply(
                            agent_id=agent_id,
                            site=site,
                            target="readout",
                            step=step,
                            episode_indices=episode_indices,
                            value=recurrent_value,
                        )
                        if readout_value is not recurrent_value:
                            readout_overrides[name] = readout_value

                    if config.record_activations and site.analysis:
                        activation_records = agent_records[agent_id]["activations"]
                        assert isinstance(activation_records, dict)
                        activation_records.setdefault(name, []).append(recurrent_value)
                        if pipeline.matches(agent_id=agent_id, site=name, target="recurrent"):
                            activation_records.setdefault(f"raw.{name}", []).append(raw)
                        if pipeline.matches(agent_id=agent_id, site=name, target="readout"):
                            activation_records.setdefault(f"readout.{name}", []).append(readout_value)

                readout = policy.readout(
                    features,
                    {**recurrent_overrides, **readout_overrides} or None,
                )
                if readout.logits.ndim != 2 or readout.logits.shape[0] != batch_size:
                    raise ValueError(f"policy {agent_id} returned invalid logits shape")
                if readout.value.shape != (batch_size,):
                    raise ValueError(f"policy {agent_id} returned invalid value shape")
                if not torch.isfinite(readout.logits).all() or not torch.isfinite(
                    readout.value
                ).all():
                    raise FloatingPointError(f"policy {agent_id} readout is non-finite")
                next_state = policy.carry_state(features, recurrent_overrides or None)
                require_tensor_tree(next_state, label=f"carry_state[{agent_id}]")
                next_states[agent_id] = next_state
                action = _select_action(
                    readout.logits,
                    config.deterministic,
                    action_generator,
                )
                log_prob = torch.log_softmax(readout.logits, dim=-1).gather(
                    -1, action.unsqueeze(-1)
                ).squeeze(-1)
                actions[agent_id] = action

                record = agent_records[agent_id]
                for key, value in (
                    ("observations", observations[agent_id]),
                    ("actions", action),
                    ("logits", readout.logits),
                    ("values", readout.value),
                    ("log_probs", log_prob),
                ):
                    series = record[key]
                    assert isinstance(series, list)
                    series.append(value)

            transition = self.environment.step(actions)
            require_agent_keys(transition.observations, self.environment.agent_ids, label="observations")
            require_agent_keys(transition.rewards, self.environment.agent_ids, label="rewards")
            _validate_transition(
                transition,
                batch_size=batch_size,
                agent_ids=self.environment.agent_ids,
            )
            for agent_id in self.environment.agent_ids:
                rewards = agent_records[agent_id]["rewards"]
                assert isinstance(rewards, list)
                rewards.append(transition.rewards[agent_id])
            terminated.append(transition.terminated)
            truncated.append(transition.truncated)
            active.append(transition.active)
            for key, value in transition.events.items():
                events.setdefault(key, []).append(value)
            if config.record_world:
                if transition.world.keys() != reset.world.keys():
                    raise ValueError("environment world keys changed during simulation")
                for key, value in transition.world.items():
                    world[key].append(value)
            observations = transition.observations
            states = next_states

        agents: dict[AgentId, AgentTrajectory] = {}
        for agent_id, record in agent_records.items():
            final_features = self.policies[agent_id].advance(
                observations[agent_id], states[agent_id]
            )
            final_readout = self.policies[agent_id].readout(final_features)
            activation_records = record["activations"]
            assert isinstance(activation_records, dict)
            state_records = record["state_inputs"]
            assert isinstance(state_records, dict)
            agents[agent_id] = AgentTrajectory(
                observations=torch.stack(record["observations"]),
                actions=torch.stack(record["actions"]),
                rewards=torch.stack(record["rewards"]),
                logits=torch.stack(record["logits"]),
                values=torch.stack(record["values"]),
                log_probs=torch.stack(record["log_probs"]),
                bootstrap_value=final_readout.value,
                activations={
                    key: torch.stack(values) for key, values in activation_records.items()
                },
                state_inputs={
                    key: torch.stack(values) for key, values in state_records.items()
                },
            )
        return TrajectoryBatch(
            agent_ids=self.environment.agent_ids,
            agents=agents,
            terminated=torch.stack(terminated),
            truncated=torch.stack(truncated),
            active=torch.stack(active),
            world={key: torch.stack(values) for key, values in world.items()},
            events={key: torch.stack(values) for key, values in events.items()},
            seed=config.seed,
            episode_seeds=torch.full(
                (batch_size,), config.seed, dtype=torch.long, device=device
            ),
        )


def _select_action(
    logits: torch.Tensor,
    deterministic: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError(f"policy logits must have shape (batch, actions), got {tuple(logits.shape)}")
    if deterministic:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    uniforms = torch.rand(
        logits.shape[0],
        generator=generator,
        device=generator.device,
        dtype=probabilities.dtype,
    ).to(logits.device)
    # Explicit inverse-CDF sampling makes the action random-number stream a
    # stable paired-rollout input, independent of the probability values.
    cumulative = probabilities.cumsum(dim=-1)
    return (uniforms.unsqueeze(-1) > cumulative).sum(dim=-1).clamp_max(
        logits.shape[-1] - 1
    )


def _validate_observations(
    observations: dict[AgentId, torch.Tensor],
    *,
    batch_size: int,
) -> None:
    for agent_id, value in observations.items():
        if value.ndim < 2 or value.shape[0] != batch_size:
            raise ValueError(f"observation for {agent_id} has invalid batch axis")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"observation for {agent_id} is non-finite")


def _validate_tensor_map(values: TensorMap, *, batch_size: int, label: str) -> None:
    for key, value in values.items():
        if value.ndim < 1 or value.shape[0] != batch_size:
            raise ValueError(f"{label} {key!r} has invalid batch axis")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} {key!r} is non-finite")


def _validate_transition(
    transition: object,
    *,
    batch_size: int,
    agent_ids: tuple[AgentId, ...],
) -> None:
    from mouse_run_run.core.environment import EnvTransition

    if not isinstance(transition, EnvTransition):
        raise TypeError(f"environment returned {type(transition)!r}, expected EnvTransition")
    _validate_observations(transition.observations, batch_size=batch_size)
    for agent_id in agent_ids:
        reward = transition.rewards[agent_id]
        if reward.shape != (batch_size,) or not reward.is_floating_point():
            raise ValueError(f"reward for {agent_id} must be floating shape ({batch_size},)")
        if not torch.isfinite(reward).all():
            raise FloatingPointError(f"reward for {agent_id} is non-finite")
    for name in ("terminated", "truncated", "active"):
        value = getattr(transition, name)
        if value.shape != (batch_size,) or value.dtype != torch.bool:
            raise ValueError(f"transition {name} must be bool shape ({batch_size},)")
    _validate_tensor_map(transition.world, batch_size=batch_size, label="world")
    _validate_tensor_map(transition.events, batch_size=batch_size, label="event")

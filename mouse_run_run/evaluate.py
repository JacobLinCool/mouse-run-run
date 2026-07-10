from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from torch.distributions import Categorical

from mouse_run_run.degenerate import episode_degeneracy
from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig
from mouse_run_run.policy import PolicyBase, build_policy
from mouse_run_run.serialization import load_checkpoint
from mouse_run_run.training_config import select_device


OpponentMode = Literal["self_play", "random_chaser", "random_explorer"]


@dataclass(frozen=True)
class EvaluationMetrics:
    episodes: int
    collisions_per_episode: float
    chaser_return: float
    explorer_return: float
    chaser_partner_vision: float
    explorer_partner_vision: float
    chaser_new_fields: float
    explorer_new_fields: float
    final_distance: float
    average_distance: float
    degenerate_episodes: int
    degenerate_fraction: float
    same_state_run_steps: float
    chaser_stuck_run_steps: float
    explorer_stuck_run_steps: float


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    *,
    episodes: int = 512,
    batch_size: int = 128,
    device_name: str = "auto",
    deterministic: bool = True,
    opponent_mode: OpponentMode = "self_play",
    max_steps: int | None = None,
    degenerate_threshold_fraction: float = 0.01,
    seed: int | None = None,
) -> EvaluationMetrics:
    if opponent_mode not in ("self_play", "random_chaser", "random_explorer"):
        raise ValueError(f"Unsupported opponent mode: {opponent_mode}")
    if seed is not None:
        torch.manual_seed(seed)
    device = select_device(device_name)
    config, _, chaser_state, explorer_state = load_checkpoint(checkpoint)
    env_config = GridWorldConfig(**config["env"])
    if max_steps is not None:
        env_config = replace(env_config, max_steps=max_steps)

    architecture = config.get("architecture", "rnn")
    chaser = build_policy(
        architecture,
        env_config.observation_size,
        hidden_size=config["hidden_size"],
    ).to(device)
    explorer = build_policy(
        architecture,
        env_config.observation_size,
        hidden_size=config["hidden_size"],
    ).to(device)
    chaser.load_state_dict(chaser_state)
    explorer.load_state_dict(explorer_state)
    chaser.eval()
    explorer.eval()

    totals = {
        "collisions_per_episode": 0.0,
        "chaser_return": 0.0,
        "explorer_return": 0.0,
        "chaser_partner_vision": 0.0,
        "explorer_partner_vision": 0.0,
        "chaser_new_fields": 0.0,
        "explorer_new_fields": 0.0,
        "final_distance": 0.0,
        "average_distance": 0.0,
        "degenerate_episodes": 0.0,
        "same_state_run_steps": 0.0,
        "chaser_stuck_run_steps": 0.0,
        "explorer_stuck_run_steps": 0.0,
    }
    completed = 0
    while completed < episodes:
        current_batch_size = min(batch_size, episodes - completed)
        metrics = _evaluate_batch(
            env_config=env_config,
            batch_size=current_batch_size,
            chaser=chaser,
            explorer=explorer,
            device=device,
            deterministic=deterministic,
            opponent_mode=opponent_mode,
            degenerate_threshold_fraction=degenerate_threshold_fraction,
        )
        for key in totals:
            if key == "degenerate_episodes":
                totals[key] += getattr(metrics, key)
            else:
                totals[key] += getattr(metrics, key) * current_batch_size
        completed += current_batch_size

    return EvaluationMetrics(
        episodes=episodes,
        collisions_per_episode=totals["collisions_per_episode"] / episodes,
        chaser_return=totals["chaser_return"] / episodes,
        explorer_return=totals["explorer_return"] / episodes,
        chaser_partner_vision=totals["chaser_partner_vision"] / episodes,
        explorer_partner_vision=totals["explorer_partner_vision"] / episodes,
        chaser_new_fields=totals["chaser_new_fields"] / episodes,
        explorer_new_fields=totals["explorer_new_fields"] / episodes,
        final_distance=totals["final_distance"] / episodes,
        average_distance=totals["average_distance"] / episodes,
        degenerate_episodes=int(totals["degenerate_episodes"]),
        degenerate_fraction=totals["degenerate_episodes"] / episodes,
        same_state_run_steps=totals["same_state_run_steps"] / episodes,
        chaser_stuck_run_steps=totals["chaser_stuck_run_steps"] / episodes,
        explorer_stuck_run_steps=totals["explorer_stuck_run_steps"] / episodes,
    )


@torch.no_grad()
def _evaluate_batch(
    *,
    env_config: GridWorldConfig,
    batch_size: int,
    chaser: PolicyBase,
    explorer: PolicyBase,
    device: torch.device,
    deterministic: bool,
    opponent_mode: OpponentMode,
    degenerate_threshold_fraction: float,
) -> EvaluationMetrics:
    env = BatchedChaseEnv(env_config, batch_size, device)
    chaser_observation, explorer_observation = env.reset()
    chaser_hidden = chaser.initial_hidden(batch_size, device)
    explorer_hidden = explorer.initial_hidden(batch_size, device)

    chaser_return = torch.zeros(batch_size, device=device)
    explorer_return = torch.zeros(batch_size, device=device)
    collisions = torch.zeros(batch_size, device=device)
    chaser_partner_vision = torch.zeros(batch_size, device=device)
    explorer_partner_vision = torch.zeros(batch_size, device=device)
    chaser_new_fields = torch.zeros(batch_size, device=device)
    explorer_new_fields = torch.zeros(batch_size, device=device)
    final_distance = env.distance()
    distance_sum = torch.zeros(batch_size, device=device)
    chaser_positions = [env.chaser_position.clone()]
    explorer_positions = [env.explorer_position.clone()]
    chaser_actions = []
    explorer_actions = []
    chaser_collisions = []
    explorer_collisions = []

    for _ in range(env_config.max_steps):
        # The randomized agent's policy network is never queried: its forward
        # pass would be discarded, and sampling from it would perturb the RNG
        # stream shared with the random opponent.
        if opponent_mode == "random_chaser":
            chaser_action = torch.randint(4, (batch_size,), device=device)
        else:
            chaser_output = chaser(chaser_observation, chaser_hidden)
            chaser_action = _select_action(chaser_output.logits, deterministic)
            chaser_hidden = chaser_output.state
        if opponent_mode == "random_explorer":
            explorer_action = torch.randint(4, (batch_size,), device=device)
        else:
            explorer_output = explorer(explorer_observation, explorer_hidden)
            explorer_action = _select_action(explorer_output.logits, deterministic)
            explorer_hidden = explorer_output.state

        result = env.step(chaser_action, explorer_action)
        chaser_actions.append(chaser_action)
        explorer_actions.append(explorer_action)
        chaser_positions.append(result.chaser_position)
        explorer_positions.append(result.explorer_position)
        chaser_collisions.append(result.chaser_collision)
        explorer_collisions.append(result.explorer_collision)
        chaser_return += result.chaser_reward
        explorer_return += result.explorer_reward
        collisions += result.collision.float()
        chaser_partner_vision += result.chaser_partner_visible.float()
        explorer_partner_vision += result.explorer_partner_visible.float()
        chaser_new_fields += result.chaser_new_field.float()
        explorer_new_fields += result.explorer_new_field.float()
        final_distance = result.distance
        distance_sum += result.distance

        chaser_observation = result.chaser_observation
        explorer_observation = result.explorer_observation

    horizon = float(env_config.max_steps)
    degeneracy = episode_degeneracy(
        torch.stack(chaser_positions),
        torch.stack(explorer_positions),
        torch.stack(chaser_actions),
        torch.stack(explorer_actions),
        chaser_collisions=torch.stack(chaser_collisions),
        explorer_collisions=torch.stack(explorer_collisions),
        threshold_fraction=degenerate_threshold_fraction,
    )
    return EvaluationMetrics(
        episodes=batch_size,
        collisions_per_episode=collisions.mean().item(),
        chaser_return=chaser_return.mean().item(),
        explorer_return=explorer_return.mean().item(),
        chaser_partner_vision=(chaser_partner_vision / horizon).mean().item(),
        explorer_partner_vision=(explorer_partner_vision / horizon).mean().item(),
        chaser_new_fields=chaser_new_fields.mean().item(),
        explorer_new_fields=explorer_new_fields.mean().item(),
        final_distance=final_distance.mean().item(),
        average_distance=(distance_sum / horizon).mean().item(),
        degenerate_episodes=int(degeneracy["degenerate"].sum().item()),
        degenerate_fraction=degeneracy["degenerate"].float().mean().item(),
        same_state_run_steps=degeneracy["same_state_run_steps"].float().mean().item(),
        chaser_stuck_run_steps=degeneracy["chaser_stuck_run_steps"].float().mean().item(),
        explorer_stuck_run_steps=degeneracy["explorer_stuck_run_steps"].float().mean().item(),
    )


def _select_action(logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return logits.argmax(dim=-1)
    return Categorical(logits=logits).sample()

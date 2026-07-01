import shlex
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.distributions import Categorical

from mouse_run_run.degenerate import episode_degeneracy
from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig
from mouse_run_run.evaluate import OpponentMode
from mouse_run_run.policy import RNNActorCritic
from mouse_run_run.serialization import load_checkpoint, save_rollout
from mouse_run_run.train import select_device


@torch.no_grad()
def collect_rollouts(
    checkpoint: Path,
    output: Path,
    *,
    episodes: int = 128,
    batch_size: int = 32,
    device_name: str = "auto",
    deterministic: bool = False,
    opponent_mode: OpponentMode = "self_play",
    max_steps: int | None = None,
    degenerate_threshold_fraction: float = 0.01,
    analysis_protocol: str = "custom_rollout_v2",
    command: str | None = None,
) -> None:
    device = select_device(device_name)
    config, _, chaser_state, explorer_state = load_checkpoint(checkpoint)
    env_config = GridWorldConfig(**config["env"])
    if max_steps is not None:
        env_config = replace(env_config, max_steps=max_steps)

    chaser = RNNActorCritic(
        env_config.observation_size,
        hidden_size=config["hidden_size"],
    ).to(device)
    explorer = RNNActorCritic(
        env_config.observation_size,
        hidden_size=config["hidden_size"],
    ).to(device)
    chaser.load_state_dict(chaser_state)
    explorer.load_state_dict(explorer_state)
    chaser.eval()
    explorer.eval()

    chunks = []
    completed = 0
    while completed < episodes:
        current_batch_size = min(batch_size, episodes - completed)
        chunks.append(
            _collect_batch(
                env_config=env_config,
                batch_size=current_batch_size,
                chaser=chaser,
                explorer=explorer,
                device=device,
                deterministic=deterministic,
                opponent_mode=opponent_mode,
                degenerate_threshold_fraction=degenerate_threshold_fraction,
            )
        )
        completed += current_batch_size

    output.parent.mkdir(parents=True, exist_ok=True)
    save_rollout(
        output,
        metadata={
            "schema_version": 2,
            "analysis_protocol": analysis_protocol,
            "checkpoint": str(checkpoint),
            "checkpoint_config": config,
            "env_config": asdict(env_config),
            "episodes": episodes,
            "batch_size": batch_size,
            "deterministic": deterministic,
            "opponent_mode": opponent_mode,
            "max_steps": env_config.max_steps,
            "degenerate_threshold_fraction": degenerate_threshold_fraction,
            "command": command or shlex.join(sys.argv),
        },
        rollout=_concat_chunks(chunks),
    )
    print(f"saved_rollouts={output}", flush=True)


@torch.no_grad()
def _collect_batch(
    *,
    env_config: GridWorldConfig,
    batch_size: int,
    chaser: RNNActorCritic,
    explorer: RNNActorCritic,
    device: torch.device,
    deterministic: bool,
    opponent_mode: OpponentMode,
    degenerate_threshold_fraction: float,
) -> dict[str, torch.Tensor]:
    env = BatchedChaseEnv(env_config, batch_size, device)
    chaser_observation, explorer_observation = env.reset()
    chaser_hidden = chaser.initial_hidden(batch_size, device)
    explorer_hidden = explorer.initial_hidden(batch_size, device)
    initial_chaser_position = env.chaser_position.clone()
    initial_explorer_position = env.explorer_position.clone()
    records: dict[str, list[torch.Tensor]] = {
        "chaser_observation": [],
        "explorer_observation": [],
        "chaser_hidden": [],
        "explorer_hidden": [],
        "chaser_action": [],
        "explorer_action": [],
        "chaser_reward": [],
        "explorer_reward": [],
        "collision": [],
        "chaser_new_field": [],
        "explorer_new_field": [],
        "chaser_partner_visible": [],
        "explorer_partner_visible": [],
        "chaser_approach": [],
        "explorer_escape": [],
        "explorer_escape_close": [],
        "explorer_escape_near": [],
        "explorer_escape_far": [],
        "distance": [],
        "chaser_position": [],
        "explorer_position": [],
    }

    for _ in range(env_config.max_steps):
        chaser_output = chaser(chaser_observation, chaser_hidden)
        explorer_output = explorer(explorer_observation, explorer_hidden)
        chaser_action = _select_action(chaser_output.logits, deterministic)
        explorer_action = _select_action(explorer_output.logits, deterministic)
        if opponent_mode == "random_chaser":
            chaser_action = torch.randint(4, (batch_size,), device=device)
        elif opponent_mode == "random_explorer":
            explorer_action = torch.randint(4, (batch_size,), device=device)
        elif opponent_mode != "self_play":
            raise ValueError(f"Unsupported opponent mode: {opponent_mode}")

        result = env.step(chaser_action, explorer_action)
        records["chaser_observation"].append(chaser_observation)
        records["explorer_observation"].append(explorer_observation)
        records["chaser_hidden"].append(chaser_output.hidden)
        records["explorer_hidden"].append(explorer_output.hidden)
        records["chaser_action"].append(chaser_action)
        records["explorer_action"].append(explorer_action)
        records["chaser_reward"].append(result.chaser_reward)
        records["explorer_reward"].append(result.explorer_reward)
        records["collision"].append(result.collision)
        records["chaser_new_field"].append(result.chaser_new_field)
        records["explorer_new_field"].append(result.explorer_new_field)
        records["chaser_partner_visible"].append(result.chaser_partner_visible)
        records["explorer_partner_visible"].append(result.explorer_partner_visible)
        records["chaser_approach"].append(result.chaser_approach)
        records["explorer_escape"].append(result.explorer_escape)
        records["explorer_escape_close"].append(result.explorer_escape_close)
        records["explorer_escape_near"].append(result.explorer_escape_near)
        records["explorer_escape_far"].append(result.explorer_escape_far)
        records["distance"].append(result.distance)
        records["chaser_position"].append(result.chaser_position)
        records["explorer_position"].append(result.explorer_position)

        chaser_hidden = chaser_output.hidden
        explorer_hidden = explorer_output.hidden
        chaser_observation = result.chaser_observation
        explorer_observation = result.explorer_observation

    tensors = {key: torch.stack(value).cpu() for key, value in records.items()}
    degeneracy = episode_degeneracy(
        torch.cat([initial_chaser_position[None], torch.stack(records["chaser_position"])]),
        torch.cat([initial_explorer_position[None], torch.stack(records["explorer_position"])]),
        torch.stack(records["chaser_action"]),
        torch.stack(records["explorer_action"]),
        threshold_fraction=degenerate_threshold_fraction,
    )
    tensors["episode_initial_chaser_position"] = initial_chaser_position.cpu()
    tensors["episode_initial_explorer_position"] = initial_explorer_position.cpu()
    for key, value in degeneracy.items():
        tensors[f"episode_{key}"] = value.cpu()
    return tensors


def _concat_chunks(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = chunks[0].keys()
    return {
        key: torch.cat(
            [chunk[key] for chunk in chunks],
            dim=0 if key.startswith("episode_") else 1,
        )
        for key in keys
    }


def _select_action(logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return logits.argmax(dim=-1)
    return Categorical(logits=logits).sample()

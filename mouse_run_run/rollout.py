import shlex
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.distributions import Categorical

from mouse_run_run.degenerate import episode_degeneracy
from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig
from mouse_run_run.evaluate import OpponentMode
from mouse_run_run.policy import PolicyBase, build_policy
from mouse_run_run.serialization import load_checkpoint, save_rollout
from mouse_run_run.train import select_device


# Tensors aligned with states s_0..s_T (length max_steps + 1 along dim 0).
STATE_SERIES_KEYS = (
    "chaser_position",
    "explorer_position",
    "distance",
    "chaser_partner_visible",
    "explorer_partner_visible",
)


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
    analysis_protocol: str = "custom_rollout_v3",
    command: str | None = None,
    seed: int | None = None,
) -> None:
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
            "schema_version": 3,
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
            "seed": seed,
            "alignment": {
                "state_series": STATE_SERIES_KEYS,
                "convention": (
                    "state-series tensors have length max_steps + 1 along dim 0 and"
                    " index t is the state s_t before action t;"
                    " observations/hidden/actions at index t are computed from s_t"
                    " (hidden[t] produced action[t]);"
                    " rewards and event flags at index t describe the transition"
                    " s_t -> s_{t+1}"
                ),
            },
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
    chaser: PolicyBase,
    explorer: PolicyBase,
    device: torch.device,
    deterministic: bool,
    opponent_mode: OpponentMode,
    degenerate_threshold_fraction: float,
) -> dict[str, torch.Tensor]:
    env = BatchedChaseEnv(env_config, batch_size, device)
    chaser_observation, explorer_observation = env.reset()
    chaser_hidden = chaser.initial_hidden(batch_size, device)
    explorer_hidden = explorer.initial_hidden(batch_size, device)
    initial_chaser_visible, initial_explorer_visible = env.partner_visibilities()
    # State-series keys are seeded with the s_0 value and get one entry per
    # step afterwards, so they are (max_steps + 1, episodes, ...) and index t
    # is the state before action t. All other per-step keys are
    # (max_steps, episodes, ...) and index t aligns with state s_t
    # (observations/hidden/actions) or transition s_t -> s_{t+1}
    # (rewards/events).
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
        "chaser_collision": [],
        "explorer_collision": [],
        "chaser_new_field": [],
        "explorer_new_field": [],
        "chaser_approach": [],
        "explorer_escape": [],
        "explorer_escape_close": [],
        "explorer_escape_near": [],
        "explorer_escape_far": [],
        "chaser_partner_visible": [initial_chaser_visible],
        "explorer_partner_visible": [initial_explorer_visible],
        "distance": [env.distance()],
        "chaser_position": [env.chaser_position.clone()],
        "explorer_position": [env.explorer_position.clone()],
    }

    for _ in range(env_config.max_steps):
        chaser_output = chaser(chaser_observation, chaser_hidden)
        explorer_output = explorer(explorer_observation, explorer_hidden)
        # The randomized agent's hidden trajectory is still recorded (the
        # trained network keeps observing), but its policy is never sampled so
        # the RNG stream only feeds the random opponent.
        if opponent_mode == "random_chaser":
            chaser_action = torch.randint(4, (batch_size,), device=device)
        else:
            chaser_action = _select_action(chaser_output.logits, deterministic)
        if opponent_mode == "random_explorer":
            explorer_action = torch.randint(4, (batch_size,), device=device)
        else:
            explorer_action = _select_action(explorer_output.logits, deterministic)

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
        records["chaser_collision"].append(result.chaser_collision)
        records["explorer_collision"].append(result.explorer_collision)
        records["chaser_new_field"].append(result.chaser_new_field)
        records["explorer_new_field"].append(result.explorer_new_field)
        records["chaser_approach"].append(result.chaser_approach)
        records["explorer_escape"].append(result.explorer_escape)
        records["explorer_escape_close"].append(result.explorer_escape_close)
        records["explorer_escape_near"].append(result.explorer_escape_near)
        records["explorer_escape_far"].append(result.explorer_escape_far)
        records["chaser_partner_visible"].append(result.chaser_partner_visible)
        records["explorer_partner_visible"].append(result.explorer_partner_visible)
        records["distance"].append(result.distance)
        records["chaser_position"].append(result.chaser_position)
        records["explorer_position"].append(result.explorer_position)

        chaser_hidden = chaser_output.state
        explorer_hidden = explorer_output.state
        chaser_observation = result.chaser_observation
        explorer_observation = result.explorer_observation

    stacked = {key: torch.stack(value) for key, value in records.items()}
    degeneracy = episode_degeneracy(
        stacked["chaser_position"],
        stacked["explorer_position"],
        stacked["chaser_action"],
        stacked["explorer_action"],
        chaser_collisions=stacked["chaser_collision"],
        explorer_collisions=stacked["explorer_collision"],
        threshold_fraction=degenerate_threshold_fraction,
    )
    tensors = {key: value.cpu() for key, value in stacked.items()}
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

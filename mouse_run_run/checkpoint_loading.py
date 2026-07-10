"""Rebuild live policy pairs from checkpoints.

serialization.py stays a pure format module; this is the one module that may
import policy/env to turn a checkpoint's config + state dicts back into
networks.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from mouse_run_run.env import GridWorldConfig
from mouse_run_run.policy import PolicyBase, build_policy
from mouse_run_run.serialization import load_checkpoint


@dataclass(frozen=True)
class LoadedPolicyPair:
    """A checkpoint's chaser/explorer pair rebuilt on a device, in eval mode."""

    env_config: GridWorldConfig
    config: dict[str, Any]
    metrics: dict[str, Any]
    chaser: PolicyBase
    explorer: PolicyBase


def load_policy_pair(
    path: Path,
    *,
    device: torch.device,
    max_steps: int | None = None,
) -> LoadedPolicyPair:
    """Load a checkpoint and rebuild both policies on ``device`` in eval mode.

    ``max_steps`` overrides the environment horizon stored in the checkpoint
    (analyses often roll out a different horizon than training used).
    """
    config, metrics, chaser_state, explorer_state = load_checkpoint(path)
    env_config = GridWorldConfig(**config["env"])
    if max_steps is not None:
        env_config = replace(env_config, max_steps=max_steps)

    if "hidden_size" not in config:
        raise ValueError(
            f"Checkpoint {path} has no 'hidden_size' in its config metadata;"
            " cannot rebuild its policies."
        )

    def build() -> PolicyBase:
        return build_policy(
            # Checkpoints from before the cross-architecture study predate the
            # "architecture" config key and are all vanilla RNNs, so default
            # silently rather than raising.
            config.get("architecture", "rnn"),
            env_config.observation_size,
            hidden_size=config["hidden_size"],
            # Pre-preset checkpoints predate this key; "modern" matches both
            # their training-time initialization and build_policy's default.
            # The initial weights are overwritten by load_state_dict below,
            # but the constructor draws from torch's global RNG, so honoring
            # the saved key keeps downstream seeded sampling consistent with
            # the checkpoint's training regime.
            rnn_initialization=config.get("rnn_initialization", "modern"),
        ).to(device)

    chaser = build()
    explorer = build()
    chaser.load_state_dict(chaser_state)
    explorer.load_state_dict(explorer_state)
    chaser.eval()
    explorer.eval()
    return LoadedPolicyPair(
        env_config=env_config,
        config=config,
        metrics=metrics,
        chaser=chaser,
        explorer=explorer,
    )

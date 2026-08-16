from __future__ import annotations

import pytest
import torch

from experiments.chase_grid.environment import CHASER, EXPLORER, ChaseGridConfig, ChaseGridEnvironment
from mouse_run_run.core.experiment import RuntimeConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton parity")
def test_triton_transition_matches_torch_for_seeded_state_and_actions() -> None:
    device = torch.device("cuda")
    config = ChaseGridConfig(grid_size=7, vision_radius=2, max_steps=8)
    torch_env = ChaseGridEnvironment(
        config, RuntimeConfig(batch_size=33, environment_backend="torch"), device
    )
    triton_env = ChaseGridEnvironment(
        config, RuntimeConfig(batch_size=33, environment_backend="triton"), device
    )
    torch_env.reset(generator=torch.Generator(device=device).manual_seed(19))
    triton_env.reset(generator=torch.Generator(device=device).manual_seed(19))
    generator = torch.Generator(device=device).manual_seed(21)
    for _ in range(config.max_steps):
        actions = {
            CHASER: torch.randint(4, (33,), device=device, generator=generator),
            EXPLORER: torch.randint(4, (33,), device=device, generator=generator),
        }
        reference = torch_env.step(actions)
        accelerated = triton_env.step(actions)
        for agent_id in (CHASER, EXPLORER):
            assert torch.equal(reference.observations[agent_id], accelerated.observations[agent_id])
            assert torch.equal(reference.rewards[agent_id], accelerated.rewards[agent_id])
        for key in reference.events:
            assert torch.equal(reference.events[key], accelerated.events[key])
        for key in reference.world:
            assert torch.equal(reference.world[key], accelerated.world[key])

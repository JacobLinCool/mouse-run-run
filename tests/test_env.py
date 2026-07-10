import torch

from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig


def test_reset_returns_flattened_two_channel_observations() -> None:
    torch.manual_seed(0)
    config = GridWorldConfig(grid_size=5, max_steps=3)
    env = BatchedChaseEnv(config, batch_size=4, device=torch.device("cpu"))

    chaser_obs, explorer_obs = env.reset()

    assert chaser_obs.shape == (4, 2 * 5 * 5)
    assert explorer_obs.shape == (4, 2 * 5 * 5)
    assert (env.chaser_position != env.explorer_position).any(dim=1).all()
    assert chaser_obs[:, :25].sum(dim=1).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_partial_full_and_none_partner_visibility() -> None:
    device = torch.device("cpu")
    env = BatchedChaseEnv(
        GridWorldConfig(grid_size=5, vision_radius=1, partner_visibility="partial"),
        batch_size=2,
        device=device,
    )
    env.reset_state()
    env.chaser_position = torch.tensor([[0, 0], [0, 0]], device=device)
    env.explorer_position = torch.tensor([[0, 1], [2, 2]], device=device)

    assert env.partner_visible().tolist() == [True, False]

    full = BatchedChaseEnv(
        GridWorldConfig(grid_size=5, partner_visibility="full"),
        batch_size=2,
        device=device,
    )
    full.reset_state()
    assert full.partner_visible().tolist() == [True, True]

    none = BatchedChaseEnv(
        GridWorldConfig(grid_size=5, partner_visibility="none"),
        batch_size=2,
        device=device,
    )
    none.reset_state()
    assert none.partner_visible().tolist() == [False, False]


def test_social_collision_blocks_chaser_and_rewards_collision() -> None:
    device = torch.device("cpu")
    env = BatchedChaseEnv(GridWorldConfig(grid_size=5, task="social"), 1, device)
    env.reset_state()
    env.chaser_position = torch.tensor([[1, 1]], device=device)
    env.explorer_position = torch.tensor([[1, 2]], device=device)
    env.chaser_visited.zero_()
    env.explorer_visited.zero_()

    result = env.step(
        torch.tensor([1], device=device),  # chaser moves right into explorer
        torch.tensor([1], device=device),  # explorer moves right
    )

    assert result.chaser_collision.tolist() == [True]
    assert result.collision.tolist() == [True]
    assert result.chaser_position.tolist() == [[1, 1]]
    assert result.chaser_reward.tolist() == [1.0]
    assert result.explorer_reward.tolist() == [-1.0]


def test_non_social_observation_hides_partner_channel() -> None:
    device = torch.device("cpu")
    config = GridWorldConfig(grid_size=5, task="non_social", partner_visibility="none")
    env = BatchedChaseEnv(config, batch_size=1, device=device)
    env.reset_state()

    chaser_obs, _ = env.observations()

    assert chaser_obs[:, :25].sum().item() == 1.0
    assert chaser_obs[:, 25:].sum().item() == 0.0


def test_non_social_keeps_physical_partner_in_fov_event() -> None:
    device = torch.device("cpu")
    env = BatchedChaseEnv(
        GridWorldConfig(
            grid_size=5,
            vision_radius=1,
            task="non_social",
            partner_visibility="none",
        ),
        batch_size=1,
        device=device,
    )
    env.reset_state()
    env.chaser_position = torch.tensor([[1, 1]], device=device)
    env.explorer_position = torch.tensor([[1, 2]], device=device)

    assert env.partner_visible().tolist() == [False]
    assert env.partner_visibilities()[0].tolist() == [True]


def test_official_spawn_mode_excludes_last_row_and_column() -> None:
    torch.manual_seed(0)
    env = BatchedChaseEnv(
        GridWorldConfig(grid_size=10, spawn_mode="official_exclude_last"),
        batch_size=256,
        device=torch.device("cpu"),
    )

    env.reset_state()

    assert int(env.chaser_position.max()) <= 8
    assert int(env.explorer_position.max()) <= 8

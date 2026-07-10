"""Alignment contract of the rollout schema (v3).

State-series tensors (STATE_SERIES_KEYS) have length max_steps + 1 along
dim 0 and index t is the pre-action state s_t; observations/hidden/actions at
index t are computed from s_t (hidden[t] produced action[t]); rewards and
event flags at index t describe the transition s_t -> s_{t+1}. analysis.py
slices against this contract, so an off-by-one here shifts every decoding
label.
"""

from dataclasses import asdict

import torch

from mouse_run_run.analysis import load_rollout
from mouse_run_run.env import (
    ACTION_DELTA_VALUES,
    CHASER_NEW_FIELD_REWARD,
    CHASER_STEP_REWARD,
    EXPLORER_NEW_FIELD_REWARD,
    EXPLORER_STEP_REWARD,
    GridWorldConfig,
    SOCIAL_CHASER_COLLISION_REWARD,
    SOCIAL_EXPLORER_COLLISION_REWARD,
)
from mouse_run_run.policy import build_policy
from mouse_run_run.rollout import (
    STATE_SERIES_KEYS,
    _concat_chunks,
    collect_batch,
    collect_rollouts,
)
from mouse_run_run.serialization import save_checkpoint, save_rollout


CONFIG = GridWorldConfig(grid_size=5, vision_radius=1, max_steps=8)
BATCH_SIZE = 8
HIDDEN_SIZE = 8
# Seed 2 was chosen so the tiny batch exercises every recorded event type:
# chaser and explorer collisions, new-field steps, approach, all escape
# distance bands, and mixed partner visibility.
SEED = 2

PER_STEP_KEYS = (
    "chaser_observation",
    "explorer_observation",
    "chaser_hidden",
    "explorer_hidden",
    "chaser_action",
    "explorer_action",
    "chaser_reward",
    "explorer_reward",
    "collision",
    "chaser_collision",
    "explorer_collision",
    "chaser_new_field",
    "explorer_new_field",
    "chaser_approach",
    "explorer_escape",
    "explorer_escape_close",
    "explorer_escape_near",
    "explorer_escape_far",
)

EPISODE_KEYS = (
    "episode_degenerate",
    "episode_same_state_run_steps",
    "episode_chaser_stuck_run_steps",
    "episode_explorer_stuck_run_steps",
    "episode_threshold_steps",
)


def _collect(*, deterministic: bool) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, torch.Tensor]]:
    torch.manual_seed(SEED)
    chaser = build_policy("rnn", CONFIG.observation_size, hidden_size=HIDDEN_SIZE)
    explorer = build_policy("rnn", CONFIG.observation_size, hidden_size=HIDDEN_SIZE)
    chaser.eval()
    explorer.eval()
    tensors = collect_batch(
        env_config=CONFIG,
        batch_size=BATCH_SIZE,
        chaser=chaser,
        explorer=explorer,
        device=torch.device("cpu"),
        deterministic=deterministic,
        opponent_mode="self_play",
        degenerate_threshold_fraction=0.01,
    )
    return chaser, explorer, tensors


def _clamp(coordinate: int) -> int:
    return max(0, min(CONFIG.grid_size - 1, coordinate))


def _candidate(position: tuple[int, int], action: int) -> tuple[int, int]:
    delta = ACTION_DELTA_VALUES[action]
    return (_clamp(position[0] + delta[0]), _clamp(position[1] + delta[1]))


def _squared_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def test_collect_batch_shapes_follow_state_vs_step_convention() -> None:
    _, _, tensors = _collect(deterministic=False)
    steps = CONFIG.max_steps

    assert set(tensors) == set(PER_STEP_KEYS) | set(STATE_SERIES_KEYS) | set(EPISODE_KEYS)

    # State-series keys: length max_steps + 1 along dim 0 (s_0 .. s_T).
    for key in STATE_SERIES_KEYS:
        assert tensors[key].shape[0] == steps + 1, key
        assert tensors[key].shape[1] == BATCH_SIZE, key
    assert tensors["chaser_position"].shape == (steps + 1, BATCH_SIZE, 2)
    assert tensors["explorer_position"].shape == (steps + 1, BATCH_SIZE, 2)
    assert tensors["chaser_position"].dtype == torch.long
    assert tensors["distance"].shape == (steps + 1, BATCH_SIZE)
    assert tensors["distance"].dtype == torch.float32
    assert tensors["chaser_partner_visible"].shape == (steps + 1, BATCH_SIZE)
    assert tensors["chaser_partner_visible"].dtype == torch.bool
    assert tensors["explorer_partner_visible"].dtype == torch.bool

    # Per-step keys: length max_steps along dim 0.
    for key in PER_STEP_KEYS:
        assert tensors[key].shape[0] == steps, key
        assert tensors[key].shape[1] == BATCH_SIZE, key
    assert tensors["chaser_observation"].shape == (steps, BATCH_SIZE, CONFIG.observation_size)
    assert tensors["chaser_hidden"].shape == (steps, BATCH_SIZE, HIDDEN_SIZE)
    assert tensors["chaser_action"].shape == (steps, BATCH_SIZE)
    assert tensors["chaser_action"].dtype == torch.long
    assert tensors["chaser_reward"].dtype == torch.float32
    for key in PER_STEP_KEYS[8:]:
        assert tensors[key].dtype == torch.bool, key

    # Episode keys: one entry per episode along dim 0.
    for key in EPISODE_KEYS:
        assert tensors[key].shape == (BATCH_SIZE,), key
    assert tensors["episode_degenerate"].dtype == torch.bool


def test_state_series_index_t_describes_state_before_action_t() -> None:
    _, _, tensors = _collect(deterministic=False)
    chaser_position = tensors["chaser_position"]
    explorer_position = tensors["explorer_position"]

    # distance[t] and visibility[t] are recomputable from positions[t] for
    # every t in 0..T, including the seeded s_0 entry and the final state.
    expected_distance = (
        (chaser_position.float() - explorer_position.float()).square().sum(dim=-1).sqrt()
    )
    assert torch.equal(tensors["distance"], expected_distance)

    chebyshev = (chaser_position - explorer_position).abs().max(dim=-1).values
    expected_visible = chebyshev <= CONFIG.vision_radius
    assert torch.equal(tensors["chaser_partner_visible"], expected_visible)
    assert torch.equal(tensors["explorer_partner_visible"], expected_visible)


def test_event_flags_at_t_describe_transition_to_t_plus_1() -> None:
    _, _, tensors = _collect(deterministic=False)
    steps = CONFIG.max_steps

    # The chosen seed exercises every branch of the replay below.
    assert bool(tensors["chaser_collision"].any())
    assert bool(tensors["explorer_collision"].any())
    assert bool(tensors["chaser_new_field"].any())
    assert bool(tensors["explorer_new_field"].any())
    assert bool(tensors["chaser_approach"].any())
    assert bool(tensors["explorer_escape_far"].any())

    for episode in range(BATCH_SIZE):
        chaser_visited = {tuple(tensors["chaser_position"][0, episode].tolist())}
        explorer_visited = {tuple(tensors["explorer_position"][0, episode].tolist())}
        for t in range(steps):
            old_chaser = tuple(tensors["chaser_position"][t, episode].tolist())
            old_explorer = tuple(tensors["explorer_position"][t, episode].tolist())

            # Chaser moves first and resolves against the explorer's old
            # position; the explorer then resolves against the chaser's new
            # position (env._advance order).
            chaser_candidate = _candidate(old_chaser, int(tensors["chaser_action"][t, episode]))
            chaser_collision = chaser_candidate == old_explorer
            new_chaser = old_chaser if chaser_collision else chaser_candidate

            explorer_candidate = _candidate(
                old_explorer, int(tensors["explorer_action"][t, episode])
            )
            explorer_collision = explorer_candidate == new_chaser
            new_explorer = old_explorer if explorer_collision else explorer_candidate

            assert tuple(tensors["chaser_position"][t + 1, episode].tolist()) == new_chaser
            assert tuple(tensors["explorer_position"][t + 1, episode].tolist()) == new_explorer
            assert bool(tensors["chaser_collision"][t, episode]) == chaser_collision
            assert bool(tensors["explorer_collision"][t, episode]) == explorer_collision
            collision = chaser_collision or explorer_collision
            assert bool(tensors["collision"][t, episode]) == collision

            chaser_new_field = not chaser_collision and new_chaser not in chaser_visited
            explorer_new_field = not explorer_collision and new_explorer not in explorer_visited
            if not chaser_collision:
                chaser_visited.add(new_chaser)
            if not explorer_collision:
                explorer_visited.add(new_explorer)
            assert bool(tensors["chaser_new_field"][t, episode]) == chaser_new_field
            assert bool(tensors["explorer_new_field"][t, episode]) == explorer_new_field

            # Approach/escape compare each agent's post-move distance to its
            # partner's pre-move position (squared forms are exact integers).
            old_squared = _squared_distance(old_chaser, old_explorer)
            chaser_partner_squared = _squared_distance(new_chaser, old_explorer)
            explorer_partner_squared = _squared_distance(new_explorer, old_chaser)
            approach = (
                not chaser_collision
                and not chaser_new_field
                and chaser_partner_squared < old_squared
            )
            escape = (
                not explorer_collision
                and not explorer_new_field
                and explorer_partner_squared > old_squared
            )
            assert bool(tensors["chaser_approach"][t, episode]) == approach
            assert bool(tensors["explorer_escape"][t, episode]) == escape
            assert bool(tensors["explorer_escape_far"][t, episode]) == (
                escape and explorer_partner_squared >= 25
            )
            assert bool(tensors["explorer_escape_near"][t, episode]) == (
                escape and 9 <= explorer_partner_squared < 25
            )
            assert bool(tensors["explorer_escape_close"][t, episode]) == (
                escape and explorer_partner_squared < 9
            )

            # Rewards at t score the same transition (social task table).
            chaser_reward = CHASER_STEP_REWARD
            explorer_reward = EXPLORER_STEP_REWARD
            if chaser_new_field:
                chaser_reward = CHASER_NEW_FIELD_REWARD
            if explorer_new_field:
                explorer_reward = EXPLORER_NEW_FIELD_REWARD
            if collision:
                chaser_reward = SOCIAL_CHASER_COLLISION_REWARD
                explorer_reward = SOCIAL_EXPLORER_COLLISION_REWARD
            # Recorded rewards are float32, so compare against the float32
            # rendering of the reward-table constants.
            assert tensors["chaser_reward"][t, episode] == torch.tensor(
                chaser_reward, dtype=torch.float32
            )
            assert tensors["explorer_reward"][t, episode] == torch.tensor(
                explorer_reward, dtype=torch.float32
            )


def test_hidden_t_is_computed_from_s_t_and_produces_action_t() -> None:
    chaser, explorer, tensors = _collect(deterministic=True)
    steps = CONFIG.max_steps
    grid = CONFIG.grid_size

    # Observation t encodes state s_t: own one-hot channel at position[t],
    # partner channel gated by visibility in s_t.
    for agent, own_key, other_key, visible_key in (
        ("chaser", "chaser_position", "explorer_position", "chaser_partner_visible"),
        ("explorer", "explorer_position", "chaser_position", "explorer_partner_visible"),
    ):
        observation = tensors[f"{agent}_observation"].view(steps, BATCH_SIZE, 2, grid, grid)
        for t in range(steps):
            for episode in range(BATCH_SIZE):
                own = tensors[own_key][t, episode]
                other = tensors[other_key][t, episode]
                visible = float(tensors[visible_key][t, episode])
                assert float(observation[t, episode, 0, own[0], own[1]]) == 1.0
                assert float(observation[t, episode, 0].sum()) == 1.0
                assert float(observation[t, episode, 1, other[0], other[1]]) == visible
                assert float(observation[t, episode, 1].sum()) == visible

    # hidden[t] is the state threaded from hidden[t-1] and observation[t],
    # and (deterministic rollout) action[t] is the argmax readout of
    # hidden[t] — the "hidden[t] produced action[t]" half of the contract.
    with torch.no_grad():
        for policy, agent in ((chaser, "chaser"), (explorer, "explorer")):
            state = policy.initial_hidden(BATCH_SIZE, torch.device("cpu"))
            for t in range(steps):
                output = policy(tensors[f"{agent}_observation"][t], state)
                assert torch.equal(output.hidden, tensors[f"{agent}_hidden"][t])
                assert torch.equal(
                    output.logits.argmax(dim=-1), tensors[f"{agent}_action"][t]
                )
                state = output.state


def test_concat_chunks_stacks_episode_keys_on_dim_0_and_time_series_on_dim_1() -> None:
    chunk_a = {
        "chaser_hidden": torch.zeros(4, 2, 3),
        "chaser_position": torch.zeros(5, 2, 2, dtype=torch.long),
        "episode_degenerate": torch.zeros(2, dtype=torch.bool),
    }
    chunk_b = {
        "chaser_hidden": torch.ones(4, 5, 3),
        "chaser_position": torch.ones(5, 5, 2, dtype=torch.long),
        "episode_degenerate": torch.ones(5, dtype=torch.bool),
    }

    merged = _concat_chunks([chunk_a, chunk_b])

    # Time-series keys keep time on dim 0 and gain episodes on dim 1;
    # episode_* keys are per-episode and concatenate on dim 0.
    assert merged["chaser_hidden"].shape == (4, 7, 3)
    assert merged["chaser_position"].shape == (5, 7, 2)
    assert merged["episode_degenerate"].shape == (7,)
    assert torch.equal(merged["chaser_hidden"][:, :2], chunk_a["chaser_hidden"])
    assert torch.equal(merged["chaser_hidden"][:, 2:], chunk_b["chaser_hidden"])
    assert torch.equal(merged["chaser_position"][:, :2], chunk_a["chaser_position"])
    assert torch.equal(merged["chaser_position"][:, 2:], chunk_b["chaser_position"])
    assert merged["episode_degenerate"].tolist() == [False, False, True, True, True, True, True]


def test_save_load_rollout_round_trip_preserves_shapes_dtypes_values(tmp_path) -> None:
    _, _, tensors = _collect(deterministic=False)
    path = tmp_path / "rollout.safetensors"

    save_rollout(
        path,
        metadata={"schema_version": 3, "max_steps": CONFIG.max_steps},
        rollout=tensors,
    )
    metadata, loaded = load_rollout(path)

    assert metadata == {"schema_version": 3, "max_steps": CONFIG.max_steps}
    assert set(loaded) == set(tensors)
    for key, value in tensors.items():
        assert loaded[key].shape == value.shape, key
        assert loaded[key].dtype == value.dtype, key
        assert torch.equal(loaded[key], value), key


def test_collect_rollouts_concatenates_chunks_and_records_alignment(tmp_path) -> None:
    torch.manual_seed(0)
    config = GridWorldConfig(grid_size=5, vision_radius=1, max_steps=4)
    chaser = build_policy("rnn", config.observation_size, hidden_size=HIDDEN_SIZE)
    explorer = build_policy("rnn", config.observation_size, hidden_size=HIDDEN_SIZE)
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_checkpoint(
        checkpoint,
        config={
            "env": asdict(config),
            "architecture": "rnn",
            "hidden_size": HIDDEN_SIZE,
            "seed": 0,
        },
        metrics={},
        chaser_state=chaser.state_dict(),
        explorer_state=explorer.state_dict(),
    )
    output = tmp_path / "rollout.safetensors"

    # episodes=5 with batch_size=2 forces three chunks (2, 2, 1).
    collect_rollouts(
        checkpoint,
        output,
        episodes=5,
        batch_size=2,
        device_name="cpu",
        deterministic=True,
        seed=0,
        command="test",
    )
    metadata, tensors = load_rollout(output)

    assert metadata["schema_version"] == 3
    assert metadata["max_steps"] == config.max_steps
    assert metadata["episodes"] == 5
    assert metadata["alignment"]["state_series"] == list(STATE_SERIES_KEYS)
    assert "hidden[t] produced action[t]" in metadata["alignment"]["convention"]
    for key in STATE_SERIES_KEYS:
        assert tensors[key].shape[0] == config.max_steps + 1, key
        assert tensors[key].shape[1] == 5, key
    assert tensors["chaser_hidden"].shape == (config.max_steps, 5, HIDDEN_SIZE)
    assert tensors["chaser_action"].shape == (config.max_steps, 5)
    assert tensors["episode_degenerate"].shape == (5,)

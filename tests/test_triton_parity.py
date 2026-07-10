import importlib.util
from pathlib import Path

import pytest
import torch

from mouse_run_run import env

_VERIFY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_triton_env.py"


def test_reward_constants_match_paper_table() -> None:
    # Tripwire: these exact literals ([PAPER-METHODS] Supplementary Table 3,
    # [OFFICIAL-ARENAS]) feed both the CPU path and the Triton kernel. Any
    # accidental edit changes trained-environment semantics.
    assert env.CHASER_STEP_REWARD == -0.1
    assert env.EXPLORER_STEP_REWARD == -0.5
    assert env.CHASER_NEW_FIELD_REWARD == 0.1
    assert env.EXPLORER_NEW_FIELD_REWARD == 1.0
    assert env.SOCIAL_CHASER_COLLISION_REWARD == 1.0
    assert env.SOCIAL_EXPLORER_COLLISION_REWARD == -1.0
    assert env.NON_SOCIAL_CHASER_COLLISION_REWARD == -0.1
    assert env.NON_SOCIAL_EXPLORER_COLLISION_REWARD == -0.5


def test_action_delta_constants_match_official_arenas() -> None:
    # [OFFICIAL-ARENAS]: 0=up, 1=right, 2=down, 3=left.
    assert env.ACTION_DELTA_VALUES == ((-1, 0), (0, 1), (1, 0), (0, -1))
    assert env.ACTION_DELTAS.dtype == torch.long
    assert env.ACTION_DELTAS.tolist() == [[-1, 0], [0, 1], [1, 0], [0, -1]]


def _load_verify_triton_env():
    spec = importlib.util.spec_from_file_location("verify_triton_env", _VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_triton_step_matches_reference_env() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton env parity check")
    verify = _load_verify_triton_env()
    device = torch.device("cuda")
    steps = 100
    for task, partner_visibility in (
        ("social", "partial"),
        ("non_social", "none"),
        ("social", "full"),
    ):
        config = env.GridWorldConfig(
            max_steps=steps,
            task=task,
            partner_visibility=partner_visibility,
        )
        for seed in range(3):
            verify.verify_case(
                config=config,
                seed=seed,
                batch_size=40,
                steps=steps,
                device=device,
            )

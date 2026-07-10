from __future__ import annotations

import json
from pathlib import Path

from mouse_run_run.viewer_payloads import _rnn_social_example


def _checkpoint(seed: int) -> str:
    return (
        f"runs/mouse-run-run-1/social/seed_{seed:04d}/attempt_01/"
        "checkpoints/latest.safetensors"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _neural_record(seed: int, *, complete: bool = True) -> dict:
    decoding = {
        key: {
            "balanced_accuracy": 0.9 + seed / 100,
            "shuffled_accuracy": 0.5,
            "n_positive": 100,
            "n_samples": 1_000,
        }
        for key in (
            "chaser_collision",
            "chaser_partner_escape",
            "explorer_collision",
            "explorer_partner_approach",
        )
    }
    if not complete:
        decoding["explorer_partner_approach"]["shuffled_accuracy"] = None
    return {
        "architecture": "rnn",
        "task": "social",
        "seed": seed,
        "checkpoint": _checkpoint(seed),
        "n_valid_episodes": 23,
        "decoding": decoding,
        "plsc": {
            "headline_null": "episode_shuffle",
            "n_episodes": 23,
            "n_samples": 11_500,
            "permutations": 200,
            "n_significant": 16,
            "top_dim_correlation": 0.94,
            "singular_values": [float(20 - index) for index in range(16)],
            "nulls": {
                "episode_shuffle": {
                    "thresholds": [float(10 - index / 2) for index in range(16)],
                    "significant": [True] * 16,
                }
            },
        },
    }


def _evaluation(seed: int, opponent_mode: str, collisions: float) -> dict:
    return {
        "checkpoint": _checkpoint(seed),
        "opponent_mode": opponent_mode,
        "evaluation_protocol": "paper_random_opponent_v1",
        "episodes": 100,
        "max_steps": 100,
        "metrics": {
            "collisions_per_episode": collisions,
            "chaser_partner_vision": 0.9,
            "average_distance": 2.0 if opponent_mode == "random_explorer" else 6.0,
            "chaser_new_fields": 25.0,
            "explorer_new_fields": 15.0,
            "degenerate_fraction": 0.9,
        },
    }


def _write_example_artifacts(runs_root: Path) -> None:
    seeds = (1, 2, 4)
    table_root = runs_root / "tables" / "mouse-run-run-1"
    analysis_root = runs_root / "analyses"
    _write_jsonl(
        table_root / "runs.jsonl",
        [
            {
                "checkpoint": _checkpoint(seed),
                "task": "social",
                "seed": seed,
                "update": 20_000,
            }
            for seed in seeds
        ],
    )
    _write_jsonl(
        table_root / "rollouts.jsonl",
        [
            {
                "checkpoint": _checkpoint(seed),
                "episode_count": 25,
                "degenerate_episode_count": 2,
                "max_steps": 500,
                "degenerate_threshold_fraction": 0.01,
            }
            for seed in seeds
        ],
    )
    scores = {1: 30.0, 2: 10.0, 4: 20.0}
    _write_jsonl(
        table_root / "evaluations.jsonl",
        [
            row
            for seed in seeds
            for row in (
                _evaluation(seed, "random_explorer", scores[seed]),
                _evaluation(seed, "random_chaser", 0.3),
            )
        ],
    )
    _write_jsonl(
        analysis_root / "mouse-run-run-1" / "neural_records.jsonl",
        [_neural_record(seed, complete=seed != 1) for seed in seeds],
    )

    trend = [
        {
            "update": 1_000,
            "partner_unique": 0.05,
            "collisions_per_episode": 80.0,
            "n_valid_episodes": 40,
            "is_latest": False,
        },
        {
            "update": 1_000_000_000,
            "partner_unique": 0.1,
            "collisions_per_episode": 120.0,
            "n_valid_episodes": 38,
            "is_latest": True,
        },
    ]
    c4_path = analysis_root / "c4_mouse-run-run-1.json"
    c4_path.parent.mkdir(parents=True, exist_ok=True)
    c4_path.write_text(
        json.dumps({"results": {"social": [{"seed": seed, "trend": trend} for seed in seeds]}}),
        encoding="utf-8",
    )

    conditions = {
        "unperturbed": {
            "collisions_per_episode": 25.0,
            "partner_in_vision": 0.95,
            "average_distance": 1.8,
        },
        "shared_removed": {
            "collisions_per_episode": 10.0,
            "partner_in_vision": 0.87,
            "average_distance": 2.4,
        },
        "random_removed": {
            "collisions_per_episode": 0.2,
            "partner_in_vision": 0.48,
            "average_distance": 4.2,
        },
    }
    (analysis_root / "c5_mouse-run-run-1.json").write_text(
        json.dumps(
            {
                "episodes": 100,
                "max_steps": 100,
                "units": [{"seed": seed, **conditions} for seed in seeds],
            }
        ),
        encoding="utf-8",
    )


def test_rnn_social_example_selects_highest_complete_checkpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_example_artifacts(runs_root)

    example = _rnn_social_example(runs_root)

    assert example is not None
    assert example["seed"] == 4
    assert example["selection"] == {
        "kind": "post_hoc_visual_exemplar",
        "rule": (
            "Among RNN/social final checkpoints with at least 20 valid 500-step "
            "self-play episodes and complete C2-C5 records, select the highest "
            "collisions/episode against the standardized random explorer; ties "
            "resolve to the lowest seed."
        ),
        "candidate_count": 3,
        "eligible_count": 2,
        "eligible_seeds": [2, 4],
        "score": 20.0,
    }
    assert example["protocols"]["self_play"]["valid_episodes"] == 23
    assert example["behavior"]["chaser_partner_vision_percent"] == 90.0
    assert len(example["decoding"]) == 4
    assert len(example["plsc"]["singular_values"]) == 16
    assert example["partner_representation"]["trend"][-1]["update"] == 20_000
    assert example["perturbation"]["conditions"]["random_removed"][
        "collisions_per_episode"
    ] == 0.2


def test_rnn_social_example_is_absent_without_source_artifacts(tmp_path: Path) -> None:
    assert _rnn_social_example(tmp_path / "runs") is None

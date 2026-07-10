"""Back-compat of the resumable-record skip keys in the paper CLIs.

Records written before the resume key included evaluation parameters must be
treated as produced under the then-current defaults, so completed work under
unchanged settings is never re-run, while changed settings never collide.
"""

import json

from mouse_run_run import paper_eval_cli, paper_rollout_cli


EVAL_DEFAULTS = {
    "episodes": 100,
    "max_steps": 100,
    "batch_size": 128,
    "deterministic": False,
    "seed": 0,
    "degenerate_threshold_fraction": 0.01,
}

ROLLOUT_DEFAULTS = {
    "episodes": 25,
    "max_steps": 500,
    "batch_size": 25,
    "deterministic": False,
    "seed": 0,
    "degenerate_threshold_fraction": 0.01,
}


def test_eval_legacy_record_matches_default_settings() -> None:
    legacy_record = {"status": "ok", "checkpoint": "a.safetensors"}
    assert paper_eval_cli._settings_key(legacy_record) == paper_eval_cli._settings_key(
        EVAL_DEFAULTS
    )


def test_eval_explicit_false_is_not_treated_as_missing() -> None:
    assert paper_eval_cli._settings_key(
        {"deterministic": False}
    ) == paper_eval_cli._settings_key({})


def test_eval_changed_settings_do_not_collide() -> None:
    baseline = paper_eval_cli._settings_key(EVAL_DEFAULTS)
    for field, changed in (
        ("episodes", 200),
        ("max_steps", 50),
        ("batch_size", 64),
        ("deterministic", True),
        ("seed", 1),
        ("degenerate_threshold_fraction", 0.5),
    ):
        assert paper_eval_cli._settings_key({**EVAL_DEFAULTS, field: changed}) != baseline


def test_evaluated_keys_carry_settings(tmp_path) -> None:
    output = tmp_path / "records.jsonl"
    legacy = {"status": "ok", "checkpoint": "c.safetensors", "opponent_mode": "random_explorer"}
    reseeded = {**legacy, "seed": 3, **{k: v for k, v in EVAL_DEFAULTS.items() if k != "seed"}}
    output.write_text(
        json.dumps(legacy) + "\n" + json.dumps(reseeded) + "\n",
        encoding="utf-8",
    )

    keys = paper_eval_cli._evaluated_keys(output)

    default_key = paper_eval_cli._settings_key(EVAL_DEFAULTS)
    assert ("c.safetensors", "random_explorer", default_key) in keys
    assert (
        "c.safetensors",
        "random_explorer",
        paper_eval_cli._settings_key({**EVAL_DEFAULTS, "seed": 3}),
    ) in keys
    assert (
        "c.safetensors",
        "random_explorer",
        paper_eval_cli._settings_key({**EVAL_DEFAULTS, "seed": 4}),
    ) not in keys


def test_rollout_legacy_record_matches_default_settings() -> None:
    legacy_record = {"status": "ok", "checkpoint": "a.safetensors"}
    assert paper_rollout_cli._settings_key(
        legacy_record
    ) == paper_rollout_cli._settings_key(ROLLOUT_DEFAULTS)


def test_rollout_changed_settings_do_not_collide(tmp_path) -> None:
    records = tmp_path / "paper_rollout_records.jsonl"
    records.write_text(
        json.dumps({"status": "ok", "checkpoint": "c.safetensors"}) + "\n",
        encoding="utf-8",
    )

    collected = paper_rollout_cli._collected_checkpoints(records)

    default_key = paper_rollout_cli._settings_key(ROLLOUT_DEFAULTS)
    assert ("c.safetensors", default_key) in collected
    assert (
        "c.safetensors",
        paper_rollout_cli._settings_key({**ROLLOUT_DEFAULTS, "episodes": 50}),
    ) not in collected

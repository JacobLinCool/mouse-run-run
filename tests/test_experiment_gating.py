"""Checkpoint selection, health gating, and runner attempt accounting.

Builds fake experiment trees (``<task>/seed_*/attempt_*`` with the worker's
status.json ``state`` contract and real safetensors checkpoints) in the states
the gating code distinguishes: completed-healthy, completed-with-NaN, failed,
interrupted (no status.json), and corrupt status. Asserts the SPEC selection
rule (latest successful finite attempt per unit), the failed-attempt budget,
resume-checkpoint selection, completed-attempt revalidation, and shutdown exit
classification.
"""

import argparse
import json
import time
from pathlib import Path

import pytest
import torch

from mouse_run_run.checkpoint_select import (
    checkpoint_format_files,
    checkpoint_identity,
    latest_successful_unit_checkpoints,
    select_checkpoints,
)
from mouse_run_run.experiment_runner import ExperimentRunner
from mouse_run_run.health import (
    NonFiniteTrainingError,
    assert_finite_scalar,
    assert_finite_tensor,
    checkpoint_health,
    require_healthy_checkpoint,
)
from mouse_run_run.provenance import read_jsonl
from mouse_run_run.serialization import (
    TrainingState,
    read_checkpoint_metadata,
    save_checkpoint,
    save_rollout,
)


UNIT_ID = "social/seed_0000"


def write_checkpoint(path, *, nan=False, training_update=None, config=None) -> Path:
    """Real checkpoint-format file with tiny tensors; optionally non-finite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    training_state = None
    if training_update is not None:
        training_state = TrainingState(
            update=training_update,
            optimizer_states={},
            learner_state={"chaser_kl_coeff": 0.2},
            cpu_rng_state=torch.get_rng_state(),
            minibatch_rng_state=torch.Generator().manual_seed(123).get_state(),
        )
    save_checkpoint(
        path,
        config=config or {},
        metrics={"reward": 1.0},
        chaser_state={"w": torch.full((2, 2), float("nan") if nan else 0.5)},
        explorer_state={"w": torch.full((2, 2), 0.25)},
        training_state=training_state,
    )
    return path


def write_attempt(unit_dir, attempt, *, state, nan=False, checkpoint=True) -> Path:
    """Attempt directory in one of the states the gating code distinguishes.

    ``state=None`` means no status.json (interrupted mid-run); ``state`` set to
    a non-JSON string simulates a torn status write.
    """
    attempt_dir = unit_dir / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if state is not None:
        payload = state if state.startswith("{") else json.dumps({"state": state, "update": 1})
        (attempt_dir / "status.json").write_text(payload, encoding="utf-8")
    if checkpoint:
        write_checkpoint(attempt_dir / "checkpoints" / "latest.safetensors", nan=nan)
    return attempt_dir


def runner_args(tmp_path, **overrides) -> argparse.Namespace:
    defaults = dict(
        runs_root=tmp_path / "runs",
        experiment="exp",
        tasks="social",
        seeds=1,
        seed_start=0,
        resume=True,
        max_attempts=2,
        max_total_attempts=10,
        attempt_timeout_hours=0.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_runner(tmp_path, **overrides) -> ExperimentRunner:
    args = runner_args(tmp_path, **overrides)
    return ExperimentRunner(args, worker_script=tmp_path / "worker.py")


def unit_dir(tmp_path) -> Path:
    return tmp_path / "runs" / "exp" / "social" / "seed_0000"


class FakeProcess:
    """Finished subprocess stand-in for _poll_running classification."""

    def __init__(self, returncode: int) -> None:
        self._returncode = returncode
        self.pid = 4242

    def poll(self) -> int:
        return self._returncode


def finish_job(runner, job, returncode):
    """Register a job as running with an already-exited process, then poll."""
    process = FakeProcess(returncode)
    job["started_monotonic"] = time.monotonic()
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runner.running[process] = job
    runner.log_handles[process] = log_path.open("a", encoding="utf-8")
    runner._poll_running()


def raw_events(runner) -> list:
    return [record["event"] for record in read_jsonl(runner.root / "raw_records.jsonl")]


# ---------------------------------------------------------------------------
# checkpoint_select: latest successful finite attempt per unit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def selection_tree(tmp_path_factory):
    """Experiment layout covering every attempt state the selector must gate."""
    root = tmp_path_factory.mktemp("selection_experiment")

    unit_a = root / "social" / "seed_0000"
    write_attempt(unit_a, 1, state="completed")
    write_attempt(unit_a, 2, state="completed", nan=True)
    write_attempt(unit_a, 3, state="failed")
    write_attempt(unit_a, 4, state=None)
    write_attempt(unit_a, 5, state="{not json")
    write_attempt(unit_a, 6, state="completed", checkpoint=False)

    unit_b = root / "social" / "seed_0001"
    write_attempt(unit_b, 1, state="completed")
    attempt = write_attempt(unit_b, 2, state="completed")
    write_checkpoint(
        attempt / "checkpoints" / "update_000005.safetensors", training_update=5
    )

    unit_c = root / "non_social" / "seed_0000"
    write_attempt(unit_c, 1, state="failed")

    save_rollout(
        root / "rollout.safetensors",
        metadata={"schema_version": 3},
        rollout={"chaser_hidden": torch.zeros(1, 1, 2)},
    )
    (root / "garbage.safetensors").write_bytes(b"not a safetensors file")

    expected = [
        unit_a / "attempt_01" / "checkpoints" / "latest.safetensors",
        unit_b / "attempt_02" / "checkpoints" / "latest.safetensors",
    ]
    return root, expected


def test_latest_successful_unit_checkpoints_skips_unhealthy_and_incomplete(selection_tree):
    root, expected = selection_tree

    selected = latest_successful_unit_checkpoints(root)

    # unit_a: attempts 6..2 are completed-without-checkpoint, corrupt status,
    # interrupted, failed, and completed-with-NaN; only attempt_01 qualifies.
    # unit_b: the newest completed healthy attempt wins. unit_c: nothing.
    assert selected == expected


def test_select_checkpoints_directory_defaults_to_per_unit_selection(selection_tree):
    root, expected = selection_tree

    assert select_checkpoints([root]) == sorted(expected)


def test_select_checkpoints_all_checkpoints_keeps_only_checkpoint_format(selection_tree):
    root, _ = selection_tree

    everything = select_checkpoints([root], all_checkpoints=True)

    # 5 unit_a latests + 2 unit_b latests + 1 mid-training + 1 unit_c latest.
    assert len(everything) == 9
    assert everything == checkpoint_format_files(root)
    nan_checkpoint = (
        root / "social" / "seed_0000" / "attempt_02" / "checkpoints" / "latest.safetensors"
    )
    mid_training = (
        root / "social" / "seed_0001" / "attempt_02" / "checkpoints" / "update_000005.safetensors"
    )
    assert nan_checkpoint in everything
    assert mid_training in everything
    assert root / "rollout.safetensors" not in everything
    assert root / "garbage.safetensors" not in everything


def test_select_checkpoints_flat_directory_excludes_mid_training(tmp_path) -> None:
    final = write_checkpoint(tmp_path / "final.safetensors")
    write_checkpoint(tmp_path / "final_update_000100.safetensors", training_update=100)
    write_checkpoint(tmp_path / "update_000050.safetensors", training_update=50)

    assert select_checkpoints([tmp_path]) == [final]
    assert len(select_checkpoints([tmp_path], all_checkpoints=True)) == 3


def test_select_checkpoints_file_arguments_dedup_and_format_gate(tmp_path) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint.safetensors")
    rollout = tmp_path / "rollout.safetensors"
    save_rollout(
        rollout,
        metadata={"schema_version": 3},
        rollout={"chaser_hidden": torch.zeros(1, 1, 2)},
    )

    assert select_checkpoints([checkpoint, checkpoint]) == [checkpoint]
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        select_checkpoints([rollout])
    with pytest.raises(FileNotFoundError):
        select_checkpoints([tmp_path / "missing"])


def test_checkpoint_identity_is_stable_and_distinct(tmp_path) -> None:
    config = {
        "env": {"task": "social"},
        "seed": 3,
        "experiment_id": "exp",
        "run_id": "exp.social.seed_0003.attempt_01",
        "attempt_id": "attempt_01",
    }

    identity = checkpoint_identity(config)

    assert identity == {
        "task": "social",
        "seed": 3,
        "unit_id": "social/seed_0003",
        "experiment_id": "exp",
        "run_id": "exp.social.seed_0003.attempt_01",
        "attempt_id": "attempt_01",
    }
    # Stable through checkpoint metadata round-trip.
    checkpoint = write_checkpoint(tmp_path / "checkpoint.safetensors", config=config)
    loaded_config, _ = read_checkpoint_metadata(checkpoint)
    assert checkpoint_identity(loaded_config) == identity
    # Distinct across units, degrades to None fields when identity is absent.
    assert checkpoint_identity({**config, "seed": 4})["unit_id"] == "social/seed_0004"
    assert checkpoint_identity({"env": {"task": "social"}})["unit_id"] is None
    assert checkpoint_identity(None) == {
        "task": None,
        "seed": None,
        "unit_id": None,
        "experiment_id": None,
        "run_id": None,
        "attempt_id": None,
    }


# ---------------------------------------------------------------------------
# health: checkpoint_health and the finite guards
# ---------------------------------------------------------------------------


def test_checkpoint_health_reports_healthy_checkpoint(tmp_path) -> None:
    checkpoint = write_checkpoint(tmp_path / "checkpoint.safetensors")

    health = checkpoint_health(checkpoint)

    assert health.ok
    assert health.format_ok
    assert health.tensor_count == 2
    assert health.nonfinite_tensor_count == 0
    assert health.max_abs == 0.5
    assert health.failures == ()
    assert health.to_dict()["path"] == str(checkpoint)
    assert "path" not in health.to_dict(include_path=False)
    assert require_healthy_checkpoint(checkpoint) == health


def test_checkpoint_health_flags_nonfinite_tensors(tmp_path) -> None:
    checkpoint = write_checkpoint(tmp_path / "nan.safetensors", nan=True)

    health = checkpoint_health(checkpoint)

    assert not health.ok
    assert health.format_ok
    assert health.nonfinite_tensor_count == 1
    assert health.failures == ({"name": "chaser.w", "nonfinite_count": 4, "element_count": 4},)
    with pytest.raises(ValueError, match=str(checkpoint)):
        require_healthy_checkpoint(checkpoint)


def test_checkpoint_health_rejects_wrong_format_and_unreadable_files(tmp_path) -> None:
    rollout = tmp_path / "rollout.safetensors"
    save_rollout(
        rollout,
        metadata={"schema_version": 3},
        rollout={"chaser_hidden": torch.zeros(1, 1, 2)},
    )
    wrong_format = checkpoint_health(rollout)
    assert not wrong_format.ok
    assert not wrong_format.format_ok
    assert wrong_format.failures[0]["name"] == "__format__"

    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"not a safetensors file")
    unreadable = checkpoint_health(garbage)
    assert not unreadable.ok
    assert not unreadable.format_ok
    assert unreadable.failures[0]["name"] == "__metadata__"

    missing = checkpoint_health(tmp_path / "missing.safetensors")
    assert not missing.ok


def test_finite_guards_raise_with_location_and_name() -> None:
    assert_finite_scalar(1.0, location="test", name="loss")
    with pytest.raises(NonFiniteTrainingError) as excinfo:
        assert_finite_scalar(float("nan"), location="update_3", name="loss")
    assert excinfo.value.location == "update_3"
    assert excinfo.value.name == "loss"

    assert_finite_tensor(torch.tensor([1, 2]), location="test", name="ints")
    with pytest.raises(NonFiniteTrainingError, match="nonfinite=1/4"):
        assert_finite_tensor(
            torch.tensor([1.0, float("inf"), 2.0, 3.0]), location="test", name="values"
        )


# ---------------------------------------------------------------------------
# ExperimentRunner: attempt accounting and resume policy at initialization
# ---------------------------------------------------------------------------


def test_failed_attempt_count_counts_failed_and_unhealthy_completed_only(tmp_path) -> None:
    unit = unit_dir(tmp_path)
    write_attempt(unit, 1, state="failed", checkpoint=False)
    write_attempt(unit, 2, state="completed", nan=True)
    write_attempt(unit, 3, state=None, checkpoint=False)  # interrupted: free retry

    runner = make_runner(tmp_path, max_attempts=3)

    assert runner.unit_failure_counts[UNIT_ID] == 2
    assert runner.failed_units == []
    assert len(runner.pending) == 1
    job = runner.pending[0]
    assert job["attempt"] == 4
    assert job["resume_from"] is None


def test_runner_exhausts_unit_after_max_attempts_genuine_failures(tmp_path) -> None:
    unit = unit_dir(tmp_path)
    write_attempt(unit, 1, state="failed", checkpoint=False)
    write_attempt(unit, 2, state="completed", nan=True)

    runner = make_runner(tmp_path, max_attempts=2)

    assert runner.pending == []
    assert runner.jobs == []
    assert [failure["reason"] for failure in runner.failed_units] == ["max_attempts_exhausted"]


def test_runner_bounds_interrupted_retries_by_max_total_attempts(tmp_path) -> None:
    unit = unit_dir(tmp_path)
    for attempt in (1, 2, 3):
        write_attempt(unit, attempt, state=None, checkpoint=False)

    runner = make_runner(tmp_path, max_total_attempts=3, max_attempts=3)

    assert runner.unit_failure_counts[UNIT_ID] == 0
    assert runner.pending == []
    assert [failure["reason"] for failure in runner.failed_units] == [
        "max_total_attempts_exhausted"
    ]


def test_resume_checkpoint_prefers_newest_healthy_training_state(tmp_path) -> None:
    unit = unit_dir(tmp_path)
    # Failed attempt with a resumable checkpoint: must never donate state.
    failed = write_attempt(unit, 1, state="failed", checkpoint=False)
    write_checkpoint(
        failed / "checkpoints" / "update_000009.safetensors", training_update=9
    )
    # Interrupted attempt: exploded and stateless checkpoints are skipped.
    interrupted = write_attempt(unit, 2, state=None)  # latest has no training state
    good = write_checkpoint(
        interrupted / "checkpoints" / "update_000005.safetensors", training_update=5
    )
    write_checkpoint(
        interrupted / "checkpoints" / "update_000010.safetensors",
        training_update=10,
        nan=True,
    )

    runner = make_runner(tmp_path, max_attempts=3)

    assert runner.unit_failure_counts[UNIT_ID] == 1
    assert len(runner.pending) == 1
    job = runner.pending[0]
    assert job["attempt"] == 3
    assert job["resume_from"] == str(good)


def test_resume_checkpoint_skips_failed_attempts_entirely(tmp_path) -> None:
    unit = unit_dir(tmp_path)
    failed = write_attempt(unit, 1, state="failed")
    write_checkpoint(
        failed / "checkpoints" / "update_000009.safetensors", training_update=9
    )

    runner = make_runner(tmp_path)

    assert len(runner.pending) == 1
    assert runner.pending[0]["resume_from"] is None


def test_runner_revalidates_completed_attempts_and_resumes_newest_healthy(tmp_path) -> None:
    unit = unit_dir(tmp_path)
    write_attempt(unit, 1, state="completed")
    write_attempt(unit, 2, state="completed")
    write_attempt(unit, 3, state="completed", nan=True)

    runner = make_runner(tmp_path)

    # attempt_03 says "completed" but its checkpoint exploded; the newest
    # completed-and-healthy attempt (02) satisfies the unit without rerunning.
    assert runner.pending == []
    assert runner.completed_unit_ids == {UNIT_ID}
    assert len(runner.completed) == 1
    job = runner.completed[0]
    assert job["attempt"] == 2
    assert job["checkpoint_health"]["ok"] is True


# ---------------------------------------------------------------------------
# ExperimentRunner: exit classification in _poll_running
# ---------------------------------------------------------------------------


def test_clean_exit_with_healthy_checkpoint_completes_unit(tmp_path) -> None:
    runner = make_runner(tmp_path)
    job = runner.pending.pop(0)
    write_checkpoint(Path(job["checkpoint"]))
    (Path(job["run_dir"]) / "status.json").write_text(
        json.dumps({"state": "completed", "update": 1}), encoding="utf-8"
    )

    finish_job(runner, job, returncode=0)

    assert runner.completed_unit_ids == {UNIT_ID}
    assert runner.failed_attempts == []
    assert job["checkpoint_health"]["ok"] is True
    assert raw_events(runner) == ["attempt_completed"]


def test_clean_exit_with_nan_checkpoint_is_failed_and_retried_from_scratch(tmp_path) -> None:
    runner = make_runner(tmp_path)
    job = runner.pending.pop(0)
    # The exploded checkpoint still carries training state: health gating, not
    # missing state, must be what blocks resuming from it.
    write_checkpoint(Path(job["checkpoint"]), nan=True, training_update=3)
    (Path(job["run_dir"]) / "status.json").write_text(
        json.dumps({"state": "completed", "update": 1}), encoding="utf-8"
    )

    finish_job(runner, job, returncode=0)

    assert runner.completed_unit_ids == set()
    assert [failure["reason"] for failure in runner.failed_attempts] == [
        "invalid_completed_checkpoint"
    ]
    assert runner.failed_attempts[0]["checkpoint_health"]["ok"] is False
    assert runner.unit_failure_counts[UNIT_ID] == 1
    retry = runner.pending[0]
    assert retry["attempt"] == 2
    assert retry["resume_from"] is None
    assert raw_events(runner) == ["attempt_failed"]


def test_nonzero_exit_during_shutdown_is_interrupted_not_failed(tmp_path) -> None:
    runner = make_runner(tmp_path)
    job = runner.pending.pop(0)
    runner.interrupted = True

    finish_job(runner, job, returncode=3)

    assert runner.failed_attempts == []
    assert runner.failed_units == []
    assert runner.pending == []
    assert runner.unit_failure_counts[UNIT_ID] == 0
    assert [attempt["returncode"] for attempt in runner.interrupted_attempts] == [3]
    assert raw_events(runner) == ["attempt_interrupted"]


def test_nonzero_exits_retry_then_exhaust_max_attempts(tmp_path) -> None:
    runner = make_runner(tmp_path, max_attempts=2)

    first = runner.pending.pop(0)
    finish_job(runner, first, returncode=1)
    assert [failure["reason"] for failure in runner.failed_attempts] == ["process_returncode"]
    assert runner.unit_failure_counts[UNIT_ID] == 1
    assert runner.failed_units == []
    assert len(runner.pending) == 1

    second = runner.pending.pop(0)
    assert second["attempt"] == 2
    finish_job(runner, second, returncode=1)
    assert runner.unit_failure_counts[UNIT_ID] == 2
    assert runner.pending == []
    assert len(runner.failed_units) == 1
    assert runner.failed_units[0]["attempt"] == 2
    assert raw_events(runner) == ["attempt_failed", "attempt_failed"]

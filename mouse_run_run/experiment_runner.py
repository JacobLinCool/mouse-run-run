"""Local multi-seed experiment orchestration.

Hosts the process-multiplexing runner behind scripts/run_local_experiment.py:
attempt accounting, resume/retry policy, shutdown classification, and the
append-only manifest/status evidence written for each experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors import SafetensorError

from mouse_run_run.health import checkpoint_health
from mouse_run_run.provenance import (
    append_jsonl,
    collect_provenance,
    hash_file,
    json_hash,
    write_json_atomic,
)
from mouse_run_run.serialization import read_metadata
from mouse_run_run.training_config import config_payload


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace, *, worker_script: Path) -> None:
        self.args = args
        self.worker_script = worker_script
        self.root = args.runs_root / args.experiment
        self.logs_dir = self.root / "logs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.units = self._build_units()
        self.jobs: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []
        self.completed_unit_ids: set[str] = set()
        self.failed_units: list[dict[str, Any]] = []
        self.failed_attempts: list[dict[str, Any]] = []
        self.interrupted_attempts: list[dict[str, Any]] = []
        self.unit_failure_counts: dict[str, int] = {}
        self.running: dict[subprocess.Popen[str], dict[str, Any]] = {}
        self.log_handles: dict[subprocess.Popen[str], Any] = {}
        self.interrupted = False
        self.started_at = time.perf_counter()
        self._initialize_jobs()

    def run(self) -> None:
        self._write_manifest()
        self._install_signal_handlers()
        self._write_status("running")
        try:
            while self.pending or self.running:
                while (
                    not self.interrupted
                    and self.pending
                    and len(self.running) < self.args.concurrency
                ):
                    self._start_job(self.pending.pop(0))
                self._poll_running()
                self._write_status("interrupted" if self.interrupted else "running")
                if self.interrupted:
                    self._terminate_running()
                    break
                time.sleep(10.0)
        finally:
            self._close_log_handles()

        state = (
            "completed"
            if (
                not self.failed_units
                and not self.pending
                and not self.running
                and len(self.completed_unit_ids) == len(self.units)
            )
            else "failed"
        )
        if self.interrupted:
            state = "interrupted"
        self._write_status(state)
        if self.failed_units:
            raise SystemExit(1)

    def _build_units(self) -> list[dict[str, Any]]:
        task_names = _task_names(self.args.tasks)
        units = []
        for seed_offset in range(self.args.seeds):
            seed = self.args.seed_start + seed_offset
            for task in task_names:
                units.append(
                    {
                        "task": task,
                        "seed": seed,
                        "unit_id": _unit_id(task, seed),
                    }
                )
        return units

    def _initialize_jobs(self) -> None:
        for unit in self.units:
            completed = self._resume_completed_attempt(unit)
            if completed is not None:
                self.jobs.append(completed)
                self.completed.append(completed)
                self.completed_unit_ids.add(completed["unit_id"])
                continue

            # Only genuinely failed attempts (NaN/Inf, crash, invalid
            # checkpoint) count toward max_attempts. Interrupted attempts
            # (runner shutdown, host reboot) are retried with a resume
            # checkpoint and only bounded by max_total_attempts.
            failure_count = self._failed_attempt_count(unit)
            self.unit_failure_counts[unit["unit_id"]] = failure_count
            next_attempt = self._next_attempt(unit)
            if failure_count >= self.args.max_attempts:
                self._record_exhausted_unit(unit, next_attempt - 1, "max_attempts_exhausted")
                continue
            if next_attempt > self.args.max_total_attempts:
                self._record_exhausted_unit(
                    unit, next_attempt - 1, "max_total_attempts_exhausted"
                )
                continue
            job = self._make_job(unit, attempt=next_attempt)
            job["resume_from"] = self._find_resume_checkpoint(unit)
            self.jobs.append(job)
            self.pending.append(job)

    def _record_exhausted_unit(self, unit: dict[str, Any], attempt: int, reason: str) -> None:
        failure = {
            **unit,
            "attempt": attempt,
            "failed_at": datetime.now(UTC).isoformat(),
            "reason": reason,
        }
        self.failed_units.append(failure)

    def _failed_attempt_count(self, unit: dict[str, Any]) -> int:
        count = 0
        for attempt_dir in self._attempt_dirs(unit):
            status = self._read_status_dir(attempt_dir)
            state = status.get("state")
            if state == "failed":
                count += 1
            elif state == "completed":
                checkpoint = attempt_dir / "checkpoints" / "latest.safetensors"
                if not checkpoint_health(checkpoint).ok:
                    count += 1
        return count

    def _find_resume_checkpoint(self, unit: dict[str, Any]) -> str | None:
        """Best resumable checkpoint from the most recent non-failed attempt.

        Failed attempts (NaN/Inf and friends) are skipped so a genuine
        failure retries from scratch; interrupted or timed-out attempts
        donate their newest healthy checkpoint with training state.
        """
        for attempt_dir in reversed(self._attempt_dirs(unit)):
            if self._read_status_dir(attempt_dir).get("state") == "failed":
                continue
            checkpoint_dir = attempt_dir / "checkpoints"
            if not checkpoint_dir.is_dir():
                continue
            best: Path | None = None
            best_update = -1
            for path in checkpoint_dir.glob("*.safetensors"):
                update = _training_state_update(path)
                if update is None or update <= best_update:
                    continue
                if not checkpoint_health(path).ok:
                    continue
                best = path
                best_update = update
            if best is not None:
                return str(best)
        return None

    def _resume_completed_attempt(self, unit: dict[str, Any]) -> dict[str, Any] | None:
        if not self.args.resume:
            return None
        completed: list[dict[str, Any]] = []
        for attempt_dir in self._attempt_dirs(unit):
            job = self._make_job(unit, attempt=_attempt_number(attempt_dir))
            status = self._read_job_status(job)
            health = checkpoint_health(Path(job["checkpoint"]))
            if status.get("state") == "completed" and health.ok:
                job["checkpoint_health"] = health.to_dict()
                completed.append(job)
        return completed[-1] if completed else None

    def _next_attempt(self, unit: dict[str, Any]) -> int:
        attempts = [_attempt_number(path) for path in self._attempt_dirs(unit)]
        return max(attempts, default=0) + 1

    def _attempt_dirs(self, unit: dict[str, Any]) -> list[Path]:
        seed_dir = self.root / unit["task"] / f"seed_{unit['seed']:04d}"
        return sorted(path for path in seed_dir.glob("attempt_*") if path.is_dir())

    def _make_job(self, unit: dict[str, Any], *, attempt: int) -> dict[str, Any]:
        # Copy only the unit identity: `unit` may be a failed job dict, and
        # spreading it whole would leak stale pid/returncode/health fields
        # into the retry record.
        unit = {"task": unit["task"], "seed": unit["seed"], "unit_id": unit["unit_id"]}
        attempt_id = f"attempt_{attempt:02d}"
        run_dir = self.root / unit["task"] / f"seed_{unit['seed']:04d}" / attempt_id
        run_id = f"{self.args.experiment}.{unit['task']}.seed_{unit['seed']:04d}.{attempt_id}"
        return {
            **unit,
            "attempt": attempt,
            "attempt_id": attempt_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "checkpoint": str(run_dir / "checkpoints" / "latest.safetensors"),
            "log": str(self.logs_dir / f"{unit['task']}_seed_{unit['seed']:04d}_{attempt_id}.log"),
        }

    def _start_job(self, job: dict[str, Any]) -> None:
        payload = {
            **job,
            "experiment": self.args.experiment,
            **config_payload(self.args),
            "log_every": self.args.log_every,
            "checkpoint_every": self.args.checkpoint_every,
            "checkpoint_every_seconds": self.args.checkpoint_every_seconds,
            "status_every_seconds": self.args.status_every_seconds,
            "cost_per_hour": self.args.cost_per_hour,
        }
        env = os.environ.copy()
        env["MRR_WORKER_CONFIG"] = json.dumps(payload)
        env["PYTHONUNBUFFERED"] = "1"
        Path(job["run_dir"]).mkdir(parents=True, exist_ok=True)
        log_handle = Path(job["log"]).open("a", encoding="utf-8")
        log_handle.write(
            f"\n=== started {datetime.now(UTC).isoformat()} "
            f"task={job['task']} seed={job['seed']} attempt={job['attempt']} ===\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            [sys.executable, str(self.worker_script), "--worker"],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        job["pid"] = process.pid
        job["started_at"] = datetime.now(UTC).isoformat()
        job["started_monotonic"] = time.monotonic()
        self.running[process] = job
        self.log_handles[process] = log_handle
        self._append_raw_record("attempt_started", job)
        print(
            f"started task={job['task']} seed={job['seed']} attempt={job['attempt']} "
            f"pid={process.pid} log={job['log']}",
            flush=True,
        )

    def _poll_running(self) -> None:
        now = time.monotonic()
        timeout_seconds = self.args.attempt_timeout_hours * 3600.0
        for process, job in list(self.running.items()):
            returncode = process.poll()
            if returncode is None:
                if (
                    timeout_seconds > 0.0
                    and not job.get("timed_out")
                    and now - job["started_monotonic"] > timeout_seconds
                ):
                    job["timed_out"] = True
                    job["kill_deadline"] = now + 30.0
                    print(
                        f"timeout task={job['task']} seed={job['seed']} "
                        f"attempt={job['attempt']} pid={job.get('pid')}",
                        flush=True,
                    )
                    process.terminate()
                elif job.get("timed_out") and now > job.get("kill_deadline", now):
                    process.kill()
                continue
            log_handle = self.log_handles.pop(process)
            log_handle.write(
                f"=== exited {datetime.now(UTC).isoformat()} "
                f"returncode={returncode} ===\n"
            )
            log_handle.close()
            del self.running[process]
            if returncode == 0 and self._completed_job_is_valid(job):
                self.completed.append(job)
                self.completed_unit_ids.add(job["unit_id"])
                self._append_raw_record("attempt_completed", job)
                print(
                    f"completed task={job['task']} seed={job['seed']} "
                    f"attempt={job['attempt']}",
                    flush=True,
                )
                continue
            if job.get("timed_out"):
                self._handle_failed_attempt(job, returncode=returncode, reason="timeout")
                continue
            if self.interrupted and returncode != 0:
                # Shutdown-induced exit: raw evidence is recorded, but the
                # attempt does not count toward max_attempts and the next
                # invocation resumes from its newest checkpoint.
                interruption = {
                    **job,
                    "returncode": returncode,
                    "interrupted_at": datetime.now(UTC).isoformat(),
                }
                self.interrupted_attempts.append(interruption)
                self._append_raw_record("attempt_interrupted", interruption)
                print(
                    f"interrupted task={job['task']} seed={job['seed']} "
                    f"attempt={job['attempt']}",
                    flush=True,
                )
                continue
            reason = "process_returncode" if returncode != 0 else "invalid_completed_checkpoint"
            self._handle_failed_attempt(job, returncode=returncode, reason=reason)

    def _completed_job_is_valid(self, job: dict[str, Any]) -> bool:
        status = self._read_job_status(job)
        health = checkpoint_health(Path(job["checkpoint"]))
        job["checkpoint_health"] = health.to_dict()
        return status.get("state") == "completed" and health.ok

    def _handle_failed_attempt(
        self,
        job: dict[str, Any],
        *,
        returncode: int | None,
        reason: str,
    ) -> None:
        health = checkpoint_health(Path(job["checkpoint"]))
        failure = {
            **job,
            "returncode": returncode,
            "failed_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "checkpoint_health": health.to_dict(),
        }
        self.failed_attempts.append(failure)
        self._append_raw_record("attempt_failed", failure)
        unit_id = job["unit_id"]
        self.unit_failure_counts[unit_id] = self.unit_failure_counts.get(unit_id, 0) + 1
        if (
            not self.interrupted
            and self.unit_failure_counts[unit_id] < self.args.max_attempts
            and job["attempt"] < self.args.max_total_attempts
            and unit_id not in self.completed_unit_ids
        ):
            retry = self._make_job(job, attempt=job["attempt"] + 1)
            retry["resume_from"] = self._find_resume_checkpoint(retry)
            self.jobs.append(retry)
            self.pending.append(retry)
            print(
                f"retrying task={job['task']} seed={job['seed']} "
                f"next_attempt={retry['attempt']} reason={reason}",
                flush=True,
            )
            return
        self.failed_units.append(failure)
        print(
            f"failed task={job['task']} seed={job['seed']} attempt={job['attempt']} "
            f"reason={reason} returncode={returncode} log={job['log']}",
            flush=True,
        )

    def _terminate_running(self) -> None:
        for process in self.running:
            process.terminate()
        deadline = time.monotonic() + 30.0
        while self.running and time.monotonic() < deadline:
            self._poll_running()
            time.sleep(1.0)
        for process in list(self.running):
            process.kill()
        self._poll_running()

    def _write_manifest(self) -> None:
        manifest_config = self._config_payload()
        spec_hash = hash_file(self.args.experiment_spec) if self.args.experiment_spec.exists() else None
        manifest = {
            "schema_version": 2,
            "experiment": self.args.experiment,
            "created_at": datetime.now(UTC).isoformat(),
            "experiment_spec": str(self.args.experiment_spec),
            "experiment_spec_sha256": spec_hash,
            "config": manifest_config,
            "config_sha256": json_hash(manifest_config),
            "provenance": collect_provenance(cwd=Path.cwd()),
            "total_units": len(self.units),
            "max_attempts": self.args.max_attempts,
            "total_environment_steps": (
                len(self.units) * self.args.updates * self.args.batch_size * self.args.max_steps
            ),
            "units": self.units,
            "initial_attempts": self.jobs,
        }
        # Every invocation is recorded append-only; manifest.json itself is
        # raw evidence of the first launch and is never overwritten. Resumed
        # re-invocations may run different code, so each invocation's
        # provenance also gets a dedicated record in invocations.jsonl.
        self._append_raw_record("runner_invocation", manifest)
        append_jsonl(
            self.root / "invocations.jsonl",
            {
                "schema_version": 1,
                "experiment": self.args.experiment,
                "invoked_at": manifest["created_at"],
                "resume": self.args.resume,
                "config_sha256": manifest["config_sha256"],
                "experiment_spec_sha256": spec_hash,
                "provenance": manifest["provenance"],
            },
        )
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if existing.get("config_sha256") != manifest["config_sha256"]:
                print(
                    "warning: config differs from original manifest.json"
                    f" (original {existing.get('config_sha256')},"
                    f" current {manifest['config_sha256']});"
                    " keeping the original manifest, see raw_records.jsonl",
                    flush=True,
                )
            return
        write_json_atomic(manifest_path, manifest)

    def _write_status(self, state: str) -> None:
        job_statuses = [self._job_status(job) for job in self.jobs]
        total_updates = len(self.units) * self.args.updates
        latest_updates: dict[str, int] = {}
        for job in job_statuses:
            latest_updates[job["unit_id"]] = max(
                latest_updates.get(job["unit_id"], 0),
                int(job.get("update", 0)),
            )
        observed_updates = sum(latest_updates.values())
        elapsed_seconds = max(time.perf_counter() - self.started_at, 1e-9)
        status = {
            "schema_version": 2,
            "state": state,
            "timestamp": datetime.now(UTC).isoformat(),
            "unix_time": time.time(),
            "experiment": self.args.experiment,
            "concurrency": self.args.concurrency,
            "running": [
                {
                    "task": job["task"],
                    "seed": job["seed"],
                    "attempt": job["attempt"],
                    "pid": job.get("pid"),
                    "run_dir": job["run_dir"],
                    "log": job["log"],
                }
                for job in self.running.values()
            ],
            "completed_jobs": len(self.completed_unit_ids),
            "failed_jobs": len(self.failed_units),
            "failed_attempts_count": len(self.failed_attempts),
            "interrupted_attempts_count": len(self.interrupted_attempts),
            "interrupted_attempts": self.interrupted_attempts,
            "pending_jobs": len(self.pending),
            "total_jobs": len(self.units),
            "known_attempts": len(self.jobs),
            "valid_pairs_by_task": self._valid_pairs_by_task(),
            "observed_updates": observed_updates,
            "total_updates": total_updates,
            "progress_fraction": observed_updates / max(total_updates, 1),
            "elapsed_seconds": elapsed_seconds,
            "failed": self.failed_units,
            "failed_attempts": self.failed_attempts,
            "jobs": job_statuses,
        }
        write_json_atomic(self.root / "run_status.json", status)

    def _job_status(self, job: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": job["task"],
            "seed": job["seed"],
            "unit_id": job["unit_id"],
            "attempt": job["attempt"],
            "attempt_id": job["attempt_id"],
            "run_id": job["run_id"],
            "run_dir": job["run_dir"],
            "log": job["log"],
            "state": "pending",
            "update": 0,
        }
        payload.update(self._read_job_status(job))
        if any(
            running_job["run_id"] == job["run_id"]
            for running_job in self.running.values()
        ):
            payload["state"] = "running"
        return payload

    def _read_job_status(self, job: dict[str, Any]) -> dict[str, Any]:
        return self._read_status_dir(Path(job["run_dir"]))

    def _read_status_dir(self, run_dir: Path) -> dict[str, Any]:
        status_path = run_dir / "status.json"
        if not status_path.exists():
            return {}
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"state": "unreadable", "error": repr(exc)}
        return status

    def _append_raw_record(self, event: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.root / "raw_records.jsonl",
            {
                "schema_version": 1,
                "event": event,
                "timestamp": datetime.now(UTC).isoformat(),
                "experiment": self.args.experiment,
                "payload": payload,
            },
        )

    def _valid_pairs_by_task(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for job in self.completed:
            counts[job["task"]] = counts.get(job["task"], 0) + 1
        return counts

    def _config_payload(self) -> dict[str, Any]:
        return {
            **config_payload(self.args),
            "checkpoint_every": self.args.checkpoint_every,
            "checkpoint_every_seconds": self.args.checkpoint_every_seconds,
            "status_every_seconds": self.args.status_every_seconds,
            "concurrency": self.args.concurrency,
        }

    def _install_signal_handlers(self) -> None:
        def handle_signal(signum: int, _frame: object) -> None:
            print(f"received_signal={signum}", flush=True)
            self.interrupted = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _close_log_handles(self) -> None:
        for handle in self.log_handles.values():
            handle.close()
        self.log_handles.clear()


def validate_args(args: argparse.Namespace) -> None:
    if args.updates < 1:
        raise ValueError("updates must be at least 1")
    if args.seeds < 1:
        raise ValueError("seeds must be at least 1")
    if args.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.max_steps < 1:
        raise ValueError("max-steps must be at least 1")
    if args.clip_epsilon <= 0:
        raise ValueError("clip-epsilon must be positive")
    if args.entropy_coef < 0:
        raise ValueError("entropy-coef must be non-negative")
    if args.value_coef < 0:
        raise ValueError("value-coef must be non-negative")
    if args.recurrent_l2_coef < 0:
        raise ValueError("recurrent-l2-coef must be non-negative")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("grad-clip must be positive")
    if args.sgd_minibatch_size < 0:
        raise ValueError("sgd-minibatch-size must be non-negative")
    if args.max_seq_len < 1:
        raise ValueError("max-seq-len must be at least 1")
    if args.kl_coeff < 0 or args.kl_target <= 0:
        raise ValueError("KL coefficient must be non-negative and target positive")
    if args.learner_mode == "rllib_2_2":
        if args.architecture != "rnn":
            raise ValueError("rllib_2_2 learner mode requires architecture='rnn'")
        if args.sgd_minibatch_size <= args.max_seq_len:
            raise ValueError("sgd-minibatch-size must be larger than max-seq-len")
        if args.batch_size * args.max_steps < args.sgd_minibatch_size:
            raise ValueError("train batch must contain at least one SGD minibatch")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be at least 1")
    if args.max_total_attempts < args.max_attempts:
        raise ValueError("max-total-attempts must be at least max-attempts")
    if args.attempt_timeout_hours < 0:
        raise ValueError("attempt-timeout-hours must be non-negative")
    if args.experiment_spec and not args.experiment_spec.exists():
        raise FileNotFoundError(str(args.experiment_spec))


def require_fresh_or_resume(args: argparse.Namespace) -> None:
    if args.resume:
        return
    root = args.runs_root / args.experiment
    if not root.exists() or not _has_experiment_evidence(root):
        return
    if not args.force_fresh:
        raise SystemExit(
            f"refusing to run with --no-resume: experiment directory {root} is"
            " not empty and re-running would shadow prior evidence."
            " Use --resume to continue, --force-fresh to archive the existing"
            " directory, or a new --experiment name."
        )
    # Archive instead of deleting or overwriting: prior evidence stays intact.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archived = root.with_name(f"{root.name}.superseded-{stamp}")
    root.rename(archived)
    print(f"archived_previous_experiment={archived}", flush=True)


def _has_experiment_evidence(root: Path) -> bool:
    """True when the directory holds run evidence (not just a gate record)."""
    if any((root / name).exists() for name in ("manifest.json", "raw_records.jsonl", "run_status.json")):
        return True
    return any(root.glob("*/seed_*"))


def _task_names(raw: str) -> tuple[str, ...]:
    if raw == "both":
        return ("social", "non_social")
    return (raw,)


def _unit_id(task: str, seed: int) -> str:
    return f"{task}/seed_{seed:04d}"


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt_"))
    except ValueError as exc:
        raise ValueError(f"invalid attempt directory: {path}") from exc


def _training_state_update(path: Path) -> int | None:
    """Update index of the training state in a checkpoint, if present."""
    try:
        metadata = read_metadata(path)
        raw_state = metadata.get("training_state")
        if raw_state is None:
            return None
        return int(json.loads(raw_state)["update"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError, SafetensorError):
        return None

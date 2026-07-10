from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mouse_run_run.provenance import append_jsonl, json_ready, provenance_block, write_json_atomic

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - exercised only when tensorboard is absent.
    SummaryWriter = None  # type: ignore[assignment]


ROLLOUT_TAGS = {
    "collisions_per_episode": "rollout/collisions_per_episode",
    "chaser_return": "rollout/chaser_return",
    "explorer_return": "rollout/explorer_return",
    "chaser_partner_vision": "rollout/chaser_partner_vision",
    "explorer_partner_vision": "rollout/explorer_partner_vision",
    "chaser_new_fields": "rollout/chaser_new_fields",
    "explorer_new_fields": "rollout/explorer_new_fields",
    "final_distance": "rollout/final_distance",
}

OPTIM_TAGS = {
    "policy_loss": "optim/policy_loss",
    "value_loss": "optim/value_loss",
    "entropy": "optim/entropy",
    "approx_kl": "optim/approx_kl",
    "chaser_kl_coeff": "optim/chaser_kl_coeff",
    "explorer_kl_coeff": "optim/explorer_kl_coeff",
}

RNN_TAGS = {
    "chaser_subspace_norm": "rnn/chaser_action_subspace_norm",
    "explorer_subspace_norm": "rnn/explorer_action_subspace_norm",
}

HEALTH_TAGS = {
    "chaser_grad_norm": "health/chaser_grad_norm",
    "explorer_grad_norm": "health/explorer_grad_norm",
    "max_param_abs": "health/max_param_abs",
}


class TrainingObserver:
    def __init__(
        self,
        *,
        run_dir: Path | None,
        metrics_path: Path | None,
        status_path: Path | None,
        tensorboard_dir: Path | None,
        config: Mapping[str, Any],
        total_updates: int,
        env_steps_per_update: int,
        status_every_seconds: float,
        cost_per_hour: float | None,
        device: str,
    ) -> None:
        self.run_dir = run_dir
        self.metrics_path = metrics_path
        self.status_path = status_path
        self.tensorboard_dir = tensorboard_dir
        if self.run_dir:
            self.metrics_path = self.metrics_path or self.run_dir / "metrics.jsonl"
            self.status_path = self.status_path or self.run_dir / "status.json"
            self.tensorboard_dir = self.tensorboard_dir or self.run_dir / "tensorboard"

        self.enabled = any((self.metrics_path, self.status_path, self.tensorboard_dir))
        self.total_updates = total_updates
        self.env_steps_per_update = env_steps_per_update
        self.status_every_seconds = status_every_seconds
        self.cost_per_hour = cost_per_hour
        self.device = device
        self.started_at = time.perf_counter()
        self.started_unix = time.time()
        self.last_recorded_at = 0.0
        self.last_checkpoint_path: str | None = None
        self.writer = None

        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                self.run_dir / "config.json",
                {**config, "provenance": provenance_block(cwd=Path.cwd(), config=config)},
            )
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if self.status_path:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if self.tensorboard_dir:
            if SummaryWriter is None:
                raise RuntimeError("tensorboard is required for TensorBoard logging.")
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.tensorboard_dir))
            self.writer.add_text("config/json", json.dumps(json_ready(config), indent=2), 0)

    def due(self) -> bool:
        if not self.enabled:
            return False
        if self.last_recorded_at == 0.0:
            return True
        return time.perf_counter() - self.last_recorded_at >= self.status_every_seconds

    def record(
        self,
        *,
        update: int,
        metrics: Mapping[str, float],
        state: str = "running",
        checkpoint_path: Path | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if checkpoint_path:
            self.last_checkpoint_path = str(checkpoint_path)

        snapshot = self._snapshot(
            update=update,
            metrics=metrics,
            state=state,
            error=error,
        )
        if self.metrics_path:
            append_jsonl(self.metrics_path, snapshot)
        if self.status_path:
            write_json_atomic(self.status_path, snapshot)
        if self.writer:
            self._write_tensorboard(snapshot, metrics)
            self.writer.flush()
        self.last_recorded_at = time.perf_counter()
        return snapshot

    def close(self) -> None:
        if self.writer:
            self.writer.close()

    def _snapshot(
        self,
        *,
        update: int,
        metrics: Mapping[str, float],
        state: str,
        error: str | None,
    ) -> dict[str, Any]:
        now = time.perf_counter()
        elapsed_seconds = max(now - self.started_at, 1e-9)
        environment_steps = update * self.env_steps_per_update
        total_environment_steps = self.total_updates * self.env_steps_per_update
        env_steps_per_second = environment_steps / elapsed_seconds
        remaining_steps = max(total_environment_steps - environment_steps, 0)
        eta_seconds = remaining_steps / max(env_steps_per_second, 1e-9)
        elapsed_hours = elapsed_seconds / 3600.0
        estimated_cost_usd = (
            None
            if self.cost_per_hour is None
            else self.cost_per_hour * elapsed_hours
        )
        progress_fraction = update / max(self.total_updates, 1)
        return {
            "schema_version": 2,
            "state": state,
            "timestamp": datetime.now(UTC).isoformat(),
            "unix_time": time.time(),
            "device": self.device,
            "update": update,
            "total_updates": self.total_updates,
            "progress_fraction": progress_fraction,
            "environment_steps": environment_steps,
            "total_environment_steps": total_environment_steps,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hours": elapsed_hours,
            "eta_seconds": eta_seconds,
            "eta_hours": eta_seconds / 3600.0,
            "env_steps_per_second": env_steps_per_second,
            "estimated_cost_usd": estimated_cost_usd,
            "latest_checkpoint": self.last_checkpoint_path,
            "error": error,
            "healthy": state != "failed" and _metrics_are_finite(metrics),
            "metrics": dict(metrics),
        }

    def _write_tensorboard(
        self,
        snapshot: Mapping[str, Any],
        metrics: Mapping[str, float],
    ) -> None:
        if not self.writer:
            return
        step = int(snapshot["environment_steps"])
        self.writer.add_scalar("progress/update", snapshot["update"], step)
        self.writer.add_scalar("progress/fraction", snapshot["progress_fraction"], step)
        self.writer.add_scalar("runtime/elapsed_hours", snapshot["elapsed_hours"], step)
        self.writer.add_scalar("runtime/eta_hours", snapshot["eta_hours"], step)
        self.writer.add_scalar(
            "runtime/env_steps_per_second",
            snapshot["env_steps_per_second"],
            step,
        )
        if snapshot["estimated_cost_usd"] is not None:
            self.writer.add_scalar(
                "runtime/estimated_cost_usd",
                snapshot["estimated_cost_usd"],
                step,
            )
        for key, tag in ROLLOUT_TAGS.items():
            self.writer.add_scalar(tag, metrics[key], step)
        for key, tag in OPTIM_TAGS.items():
            self.writer.add_scalar(tag, metrics[key], step)
        for key, tag in RNN_TAGS.items():
            self.writer.add_scalar(tag, metrics[key], step)
        for key, tag in HEALTH_TAGS.items():
            self.writer.add_scalar(tag, metrics[key], step)
        self.writer.add_scalar("health/is_failed", 1.0 if snapshot["state"] == "failed" else 0.0, step)


def _metrics_are_finite(metrics: Mapping[str, float]) -> bool:
    for value in metrics.values():
        if isinstance(value, int):
            continue
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            return False
    return True

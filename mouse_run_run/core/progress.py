"""Small common progress/status surface for all long-running commands."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(frozen=True)
class ProgressUpdate:
    operation: str
    completed: int
    total: int
    elapsed_seconds: float
    rate: float
    eta_seconds: float
    metrics: Mapping[str, float]
    state: str = "running"

    def as_dict(self) -> dict[str, object]:
        return {
            "format": "mrr-status-v2",
            "operation": self.operation,
            "state": self.state,
            "timestamp": datetime.now(UTC).isoformat(),
            "completed": self.completed,
            "total": self.total,
            "fraction": self.completed / max(self.total, 1),
            "elapsed_seconds": self.elapsed_seconds,
            "rate": self.rate,
            "eta_seconds": self.eta_seconds,
            "metrics": dict(self.metrics),
        }


class ProgressSink(Protocol):
    def update(self, update: ProgressUpdate) -> None: ...


class TerminalProgress:
    def update(self, update: ProgressUpdate) -> None:
        metrics = " ".join(f"{key}={value:.4g}" for key, value in update.metrics.items())
        print(
            f"{update.operation} {update.completed}/{update.total} "
            f"elapsed={_duration(update.elapsed_seconds)} eta={_duration(update.eta_seconds)} "
            f"rate={update.rate:.2f}/s {metrics}".rstrip(),
            flush=True,
        )


class StatusFileProgress:
    def __init__(self, path: Path, downstream: ProgressSink | None = None) -> None:
        self.path = path
        self.downstream = downstream

    def update(self, update: ProgressUpdate) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(update.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        if self.downstream is not None:
            self.downstream.update(update)


class ProgressTracker:
    def __init__(self, operation: str, total: int, sink: ProgressSink) -> None:
        if total < 1:
            raise ValueError("progress total must be positive")
        self.operation = operation
        self.total = total
        self.sink = sink
        self.started_at = time.perf_counter()

    def emit(
        self,
        completed: int,
        metrics: Mapping[str, float] | None = None,
        *,
        state: str = "running",
    ) -> None:
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        rate = completed / elapsed
        eta = max(self.total - completed, 0) / max(rate, 1e-9)
        self.sink.update(
            ProgressUpdate(
                operation=self.operation,
                completed=completed,
                total=self.total,
                elapsed_seconds=elapsed,
                rate=rate,
                eta_seconds=eta,
                metrics={} if metrics is None else metrics,
                state=state,
            )
        )


def _duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"

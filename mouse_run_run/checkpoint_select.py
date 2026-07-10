"""Checkpoint discovery for paper-protocol evaluation and rollout tools.

The SPEC's analysis unit is "the latest successful finite attempt per unit".
Directory arguments therefore default to selecting each unit's newest
completed-and-healthy ``checkpoints/latest.safetensors``; mid-training
``update_*`` checkpoints and failed attempts are only reachable with
``all_checkpoints=True``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from safetensors import SafetensorError

from mouse_run_run.health import checkpoint_health
from mouse_run_run.serialization import CHECKPOINT_FORMAT, read_metadata


_MID_TRAINING_PATTERN = re.compile(r"(?:^|_)update_\d+$")


def select_checkpoints(paths: list[Path], *, all_checkpoints: bool = False) -> list[Path]:
    """Resolve file/directory arguments to checkpoint paths.

    Files are taken as-is. For directories:

    - default: per-unit latest successful attempt when the directory contains
      an experiment layout (``<task>/seed_*/attempt_*``); otherwise every
      final checkpoint, excluding mid-training ``update_*`` files;
    - ``all_checkpoints=True``: every checkpoint-format safetensors file.
    """
    checkpoints: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            _require_checkpoint_format(path)
            checkpoints.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        if all_checkpoints:
            checkpoints.extend(checkpoint_format_files(path))
            continue
        unit_checkpoints = latest_successful_unit_checkpoints(path)
        if unit_checkpoints:
            checkpoints.extend(unit_checkpoints)
            continue
        checkpoints.extend(
            candidate
            for candidate in checkpoint_format_files(path)
            if not _is_mid_training(candidate)
        )
    return sorted(dict.fromkeys(checkpoints))


def latest_successful_unit_checkpoints(root: Path) -> list[Path]:
    """Newest completed attempt's healthy latest.safetensors for each unit."""
    attempts_by_unit: dict[Path, list[Path]] = {}
    for attempt_dir in root.glob("**/attempt_*"):
        if attempt_dir.is_dir():
            attempts_by_unit.setdefault(attempt_dir.parent, []).append(attempt_dir)

    selected: list[Path] = []
    for unit_dir in sorted(attempts_by_unit):
        for attempt_dir in sorted(attempts_by_unit[unit_dir], reverse=True):
            if _status_state(attempt_dir) != "completed":
                continue
            checkpoint = attempt_dir / "checkpoints" / "latest.safetensors"
            if checkpoint.is_file() and checkpoint_health(checkpoint).ok:
                selected.append(checkpoint)
                break
    return selected


def checkpoint_identity(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Unit identity fields recorded in a checkpoint's own config metadata."""
    config = config or {}
    env = config.get("env") or {}
    task = env.get("task")
    seed = config.get("seed")
    unit_id = None
    if task is not None and seed is not None:
        unit_id = f"{task}/seed_{int(seed):04d}"
    return {
        "task": task,
        "seed": seed,
        "unit_id": unit_id,
        "experiment_id": config.get("experiment_id"),
        "run_id": config.get("run_id"),
        "attempt_id": config.get("attempt_id"),
    }


def checkpoint_format_files(root: Path) -> list[Path]:
    """Every checkpoint-format safetensors file under ``root``, sorted."""
    checkpoints = []
    for candidate in sorted(root.rglob("*.safetensors")):
        try:
            metadata = read_metadata(candidate)
        except (OSError, ValueError, SafetensorError):
            continue
        if metadata.get("format") == CHECKPOINT_FORMAT:
            checkpoints.append(candidate)
    return checkpoints


def _is_mid_training(path: Path) -> bool:
    return _MID_TRAINING_PATTERN.search(path.stem) is not None


def _status_state(attempt_dir: Path) -> str | None:
    status_path = attempt_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return status.get("state")


def _require_checkpoint_format(path: Path) -> None:
    metadata = read_metadata(path)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}: {metadata.get('format')}")

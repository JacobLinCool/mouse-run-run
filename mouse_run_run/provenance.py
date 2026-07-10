from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def command_line(argv: Sequence[str] | None = None) -> str:
    return shlex.join(list(sys.argv if argv is None else argv))


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(record), sort_keys=True) + "\n")


def read_jsonl(path: Path, *, skip_invalid: bool = False) -> list[Any]:
    """Records from a JSONL file; [] when the file does not exist.

    Blank lines are ignored. ``skip_invalid`` drops undecodable lines (e.g. a
    torn final append) instead of raising, which is what resume scans over
    append-only record files want.
    """
    if not path.exists():
        return []
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if skip_invalid:
                    continue
                raise
            records.append(record)
    return records


def provenance_block(*, cwd: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compact provenance stamp embedded in artifacts (checkpoints, run dirs).

    Smaller than :func:`collect_provenance`: just when the artifact was
    written, the git state that wrote it, and a hash of its governing config.
    """
    block: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "git": git_info(cwd),
    }
    if config is not None:
        block["config_sha256"] = json_hash(config)
    return block


def collect_provenance(*, cwd: Path, command: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "command": command or command_line(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "git": git_info(cwd),
        "torch": torch_info(),
        "dependencies": dependency_snapshot(),
    }


def git_info(cwd: Path) -> dict[str, Any]:
    return {
        "sha": _git(["rev-parse", "HEAD"], cwd),
        "short_sha": _git(["rev-parse", "--short", "HEAD"], cwd),
        "branch": _git(["branch", "--show-current"], cwd),
        "status_short": _git(["status", "--short"], cwd),
        "dirty": bool(_git(["status", "--short"], cwd)),
        "diff_sha256": _git_diff_hash(cwd),
    }


def torch_info() -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
        "cudnn_version": torch.backends.cudnn.version(),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
    }


def dependency_snapshot() -> dict[str, Any]:
    packages = ["numpy", "safetensors", "tensorboard", "torch", "triton", "huggingface_hub"]
    versions: dict[str, str | None] = {}
    for package in packages:
        versions[package] = _package_version(package)
    return versions


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_diff_hash(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _package_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_ready(item) for item in value]
    return value

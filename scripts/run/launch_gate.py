from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mouse_run_run.provenance import collect_provenance, git_info, hash_file, write_json_atomic


PRIMARY_EXPECTATIONS = {
    "updates": 20_000,
    "batch_size": 40,
    "max_steps": 100,
    "seeds": 10,
    "seed_start": 0,
    "tasks": "both",
    "preset": "paper_text",
    "device": "cuda",
    "finite_guard": True,
    "triton_env_step": True,
    "fused_agent_rollout": True,
    "subspace_metric_period": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--triton-equivalence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Skip primary-run expectations. The ONLY way to bypass them.",
    )
    args = parser.parse_args()

    config = _read_json(args.config)
    spec_path = Path(config.get("experiment_spec", ""))
    checks = []
    checks.append(_check_path("experiment_spec", spec_path))
    checks.append(_check_config_schema(config))
    # Primary expectations are gated exclusively on --allow-smoke; a config's
    # experiment name must never bypass them.
    if not args.allow_smoke:
        checks.extend(_check_primary_expectations(config))
        checks.append(_check_device(config))
    if args.triton_equivalence:
        checks.append(_check_triton(args.triton_equivalence))
    elif config.get("triton_env_step"):
        checks.append(
            {
                "name": "triton_equivalence",
                "ok": False,
                "detail": "required when triton_env_step=true",
            }
        )

    ok = all(check["ok"] for check in checks)
    record = {
        "schema_version": 1,
        "status": "ok" if ok else "failed",
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "config_sha256": hash_file(args.config),
        "experiment_spec": str(spec_path),
        "experiment_spec_sha256": hash_file(spec_path) if spec_path.exists() else None,
        "triton_equivalence": str(args.triton_equivalence) if args.triton_equivalence else None,
        "checks": checks,
        "provenance": collect_provenance(cwd=Path.cwd()),
    }
    output = args.output or Path("runs") / config.get("experiment", "unknown-experiment") / "launch_gate.json"
    write_json_atomic(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


def _check_path(name: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "ok": path.exists(),
        "detail": str(path),
    }


def _check_config_schema(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "experiment",
        "experiment_spec",
        "updates",
        "batch_size",
        "max_steps",
        "seeds",
        "seed_start",
        "tasks",
        "preset",
        "device",
        "finite_guard",
        "max_attempts",
        "checkpoint_every",
        "status_every_seconds",
    }
    missing = sorted(required - set(config))
    return {
        "name": "config_schema",
        "ok": not missing,
        "detail": {"missing": missing},
    }


def _check_primary_expectations(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for key, expected in PRIMARY_EXPECTATIONS.items():
        checks.append(
            {
                "name": f"primary_{key}",
                "ok": config.get(key) == expected,
                "detail": {"expected": expected, "actual": config.get(key)},
            }
        )
    checks.append(
        {
            "name": "max_attempts",
            "ok": int(config.get("max_attempts", 0)) >= 2,
            "detail": {"actual": config.get("max_attempts")},
        }
    )
    checks.append(
        {
            "name": "observability_period",
            "ok": float(config.get("status_every_seconds", 0.0)) <= 3600.0,
            "detail": {"actual_seconds": config.get("status_every_seconds")},
        }
    )
    checks.append(
        {
            "name": "attempt_timeout",
            "ok": float(config.get("attempt_timeout_hours", 0.0)) > 0.0,
            "detail": {"actual_hours": config.get("attempt_timeout_hours")},
        }
    )
    return checks


def _check_device(config: dict[str, Any]) -> dict[str, Any]:
    device = config.get("device")
    if device != "cuda":
        return {"name": "device_available", "ok": True, "detail": {"device": device}}
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on local install
        return {"name": "device_available", "ok": False, "detail": {"error": repr(exc)}}
    return {
        "name": "device_available",
        "ok": available,
        "detail": {"device": device, "cuda_available": available},
    }


def _check_triton(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"name": "triton_equivalence", "ok": False, "detail": str(path)}
    try:
        data = _read_json(path)
    except Exception as exc:
        return {
            "name": "triton_equivalence",
            "ok": False,
            "detail": {"path": str(path), "error": repr(exc)},
        }
    record_git = data.get("provenance", {}).get("git", {})
    current_git = git_info(Path.cwd())
    record_sha = record_git.get("sha")
    sha_matches = record_sha is not None and record_sha == current_git.get("sha")
    # The equivalence record only certifies the code it was produced from: a
    # stale record from another commit, or a dirty working tree on either
    # side, does not certify the code about to be launched.
    ok = (
        data.get("status") == "ok"
        and sha_matches
        and not record_git.get("dirty", True)
        and not current_git.get("dirty", True)
    )
    return {
        "name": "triton_equivalence",
        "ok": ok,
        "detail": {
            "path": str(path),
            "status": data.get("status"),
            "record_git_sha": record_sha,
            "current_git_sha": current_git.get("sha"),
            "sha_matches": sha_matches,
            "record_dirty": record_git.get("dirty"),
            "current_dirty": current_git.get("dirty"),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object expected: {path}")
    return data


if __name__ == "__main__":
    main()

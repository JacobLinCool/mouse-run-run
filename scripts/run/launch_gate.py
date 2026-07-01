from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic


PRIMARY_EXPECTATIONS = {
    "updates": 20_000,
    "batch_size": 40,
    "max_steps": 100,
    "seeds": 10,
    "tasks": "both",
    "preset": "paper_text",
    "finite_guard": True,
    "triton_env_step": True,
    "subspace_metric_period": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--triton-equivalence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    config = _read_json(args.config)
    spec_path = Path(config.get("experiment_spec", ""))
    checks = []
    checks.append(_check_path("experiment_spec", spec_path))
    checks.append(_check_config_schema(config))
    if not args.allow_smoke and config.get("experiment", "") != "paper-marl-smoke":
        checks.extend(_check_primary_expectations(config))
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
    output = args.output or Path("runs") / config["experiment"] / "launch_gate.json"
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
        "tasks",
        "preset",
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
    return checks


def _check_triton(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"name": "triton_equivalence", "ok": False, "detail": str(path)}
    data = _read_json(path)
    return {
        "name": "triton_equivalence",
        "ok": data.get("status") == "ok",
        "detail": {"path": str(path), "status": data.get("status")},
    }


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object expected: {path}")
    return data


if __name__ == "__main__":
    main()

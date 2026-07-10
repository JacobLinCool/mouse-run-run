from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mouse_run_run.provenance import collect_provenance, git_info, hash_file, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]

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

# Source IDs below resolve in the official-dynamics SPEC's "Implementation
# Source Registry". [PAPER-METHODS] fixes the reportable scale;
# [OFFICIAL-TRAIN]/[OFFICIAL-ARENAS] fix released-code environment semantics.
# Device, TF32, finite_guard, Triton, fused rollout, and metric cadence are
# [LOCAL-CALIBRATION] execution choices and must not be described as paper
# constants.
OFFICIAL_DYNAMICS_EXPECTATIONS = {
    "updates": 20_000,
    "batch_size": 40,
    "max_steps": 100,
    "seeds": 10,
    "seed_start": 0,
    "tasks": "both",
    "preset": "official_code",
    "spawn_mode": "official_exclude_last",
    "device": "cuda",
    "cuda_tf32": False,
    "finite_guard": True,
    "triton_env_step": True,
    "fused_agent_rollout": True,
    "subspace_metric_period": 10,
}

# This gate intentionally duplicates the resolved preset so a silent preset
# drift fails before a full launch. Numeric sources are [RAY-PPO-CONFIG],
# [OFFICIAL-TRAIN], and [OFFICIAL-MODEL]; loss semantics are [RAY-PPO-LOSS].
OFFICIAL_LEARNER_EXPECTATIONS = {
    "architecture": "rnn",
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "learning_rate": 5e-5,
    "ppo_epochs": 30,
    "clip_epsilon": 0.3,
    "entropy_coef": 0.0,
    "value_coef": 1.0,
    "value_clip": 10.0,
    "grad_clip": None,
    "sgd_minibatch_size": 128,
    "max_seq_len": 20,
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "learner_mode": "rllib_2_2",
    "rnn_initialization": "pytorch_default",
}

# The audited upstream contract value ([RAY-PPO-CONFIG]); experiments that
# intentionally deviate declare recurrent_l2_coef in their committed config
# under experiments/*/configs/.
OFFICIAL_RECURRENT_L2_DEFAULT = 3.0


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
    if not spec_path.is_absolute():
        # Configs record repo-root-relative spec paths.
        spec_path = REPO_ROOT / spec_path
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
        "provenance": collect_provenance(cwd=REPO_ROOT),
    }
    output = args.output or REPO_ROOT / "runs" / config.get("experiment", "unknown-experiment") / "launch_gate.json"
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
    expectations = dict(
        OFFICIAL_DYNAMICS_EXPECTATIONS
        if config.get("preset") == "official_code"
        else PRIMARY_EXPECTATIONS
    )
    # The fused rollout fast path exists for rnn and ssm; other architecture
    # variants must run with it disabled.
    if config.get("architecture", "rnn") not in ("rnn", "ssm"):
        expectations["fused_agent_rollout"] = False
    for key, expected in expectations.items():
        checks.append(
            {
                "name": f"primary_{key}",
                "ok": config.get(key) == expected,
                "detail": {"expected": expected, "actual": config.get(key)},
            }
        )
    if config.get("preset") == "official_code":
        for key, expected in OFFICIAL_LEARNER_EXPECTATIONS.items():
            # Omitted values resolve from the named preset; explicit values
            # must remain identical to the audited upstream contract.
            actual = config.get(key, expected)
            checks.append(
                {
                    "name": f"official_learner_{key}",
                    "ok": actual == expected,
                    "detail": {"expected": expected, "actual": actual},
                }
            )
        expected_l2, expected_l2_source = _expected_recurrent_l2(
            str(config.get("experiment", ""))
        )
        actual_l2 = config.get("recurrent_l2_coef", OFFICIAL_RECURRENT_L2_DEFAULT)
        checks.append(
            {
                "name": "official_learner_recurrent_l2_coef",
                "ok": actual_l2 == expected_l2,
                "detail": {
                    "expected": expected_l2,
                    "actual": actual_l2,
                    "expected_source": expected_l2_source,
                },
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


def _expected_recurrent_l2(experiment: str) -> tuple[float, str]:
    """Expected recurrent_l2_coef for an experiment, read from its committed
    config under experiments/*/configs/ (an omitted key means the audited
    official value). Falls back to the official value for experiments with no
    committed config."""
    for candidate in sorted(REPO_ROOT.glob("experiments/*/configs/*.json")):
        try:
            declared = _read_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if declared.get("experiment") == experiment:
            return (
                declared.get("recurrent_l2_coef", OFFICIAL_RECURRENT_L2_DEFAULT),
                str(candidate),
            )
    return OFFICIAL_RECURRENT_L2_DEFAULT, "official_code preset default"


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
    current_git = git_info(REPO_ROOT)
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

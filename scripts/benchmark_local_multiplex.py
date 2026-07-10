from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from mouse_run_run.benchmark import parse_positive_ints, steady_timing
from mouse_run_run.provenance import collect_provenance, write_json_atomic
from mouse_run_run.train import train
from mouse_run_run.training_config import (
    add_training_arguments,
    apply_preset_defaults,
    config_payload,
    train_config_from_payload,
)


# [PAPER-METHODS] reportable panel size. Source IDs resolve in the
# official-dynamics SPEC's "Implementation Source Registry".
DEFAULT_FULL_JOBS = 20  # 10 social + 10 non-social policy pairs.
DEFAULT_FULL_UPDATES = 20_000


def main() -> None:
    worker_parser = argparse.ArgumentParser(add_help=False)
    worker_parser.add_argument("--worker", action="store_true")
    worker_args, _ = worker_parser.parse_known_args()
    if worker_args.worker:
        _worker()
        return

    parser = argparse.ArgumentParser(parents=[worker_parser])
    # [LOCAL-CALIBRATION] candidate process multiplexing levels; not a paper
    # hyperparameter and not part of the learner-equivalence claim.
    parser.add_argument("--concurrencies", default="1,2,4,8,10")
    add_training_arguments(
        parser,
        updates=3,
        preset=None,
        device="cuda",
        devices=("cuda",),
        architectures=("rnn",),
        spawn_mode="official_exclude_last",
        cuda_tf32=False,
        subspace_metric_period=10,
        triton_env_step=True,
        fused_agent_rollout=True,
    )
    parser.add_argument("--task", choices=("social", "non_social"), default="social")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--full-jobs", type=int, default=DEFAULT_FULL_JOBS)
    parser.add_argument("--full-updates", type=int, default=DEFAULT_FULL_UPDATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply_preset_defaults(args)
    _validate_args(args)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {args.output}")
    artifact_root = args.output.parent / f"{args.output.stem}_artifacts"
    if artifact_root.exists():
        raise FileExistsError(f"refusing to overwrite benchmark artifacts: {artifact_root}")

    results = [
        _run_concurrency(args, concurrency, artifact_root)
        for concurrency in parse_positive_ints(args.concurrencies)
    ]
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "command": shlex.join(sys.argv),
        "provenance": collect_provenance(cwd=Path.cwd()),
        "benchmark": _benchmark_config(args),
        "projection": {
            "full_jobs": args.full_jobs,
            "full_updates": args.full_updates,
            "full_environment_steps": (
                args.full_jobs
                * args.full_updates
                * args.batch_size
                * args.max_steps
            ),
            "method": (
                "warm update 1 excluded; aggregate steady-state throughput "
                "uses the slowest completed worker at each concurrency"
            ),
        },
        "artifact_root": str(artifact_root),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, payload)
    _print_table(results)
    print(f"wrote_benchmark={args.output}", flush=True)


def _worker() -> None:
    payload = json.loads(os.environ["MRR_WORKER_CONFIG"])
    run_dir = Path(payload["run_dir"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    metrics = train(
        train_config_from_payload(
            payload,
            log_every=1,
            checkpoint=run_dir / "checkpoints" / "latest.safetensors",
            status_every_seconds=3600.0,
        )
    )
    timing = steady_timing(run_dir / "metrics.jsonl", payload["updates"])
    print(
        "MRR_RESULT="
        + json.dumps(
            {
                "seed": payload["seed"],
                "elapsed_seconds": perf_counter() - started,
                "steady_elapsed_seconds": timing["steady_elapsed_seconds"],
                "steady_update_seconds": timing["steady_update_seconds"],
                "peak_cuda_memory_mib": (
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                    if torch.cuda.is_available()
                    else None
                ),
                "run_dir": str(run_dir),
                "metrics": asdict(metrics),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_concurrency(
    args: argparse.Namespace,
    concurrency: int,
    artifact_root: Path,
) -> dict[str, Any]:
    started = perf_counter()
    concurrency_dir = artifact_root / f"concurrency_{concurrency:02d}"
    concurrency_dir.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for index in range(concurrency):
        seed = args.seed_start + index
        payload = {
            **_benchmark_config(args),
            "seed": seed,
            "run_dir": str(concurrency_dir / f"seed_{seed:04d}"),
        }
        env = os.environ.copy()
        env["MRR_WORKER_CONFIG"] = json.dumps(payload)
        env["PYTHONUNBUFFERED"] = "1"
        processes.append(
            (
                seed,
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
            )
        )

    pair_results = []
    failures = []
    for seed, process in processes:
        output, _ = process.communicate()
        result_lines = [
            line.removeprefix("MRR_RESULT=")
            for line in output.splitlines()
            if line.startswith("MRR_RESULT=")
        ]
        if process.returncode != 0 or not result_lines:
            failures.append(
                {
                    "seed": seed,
                    "returncode": process.returncode,
                    "output_tail": output[-4000:],
                }
            )
            continue
        pair_results.append(json.loads(result_lines[-1]))

    elapsed_seconds = perf_counter() - started
    steady_elapsed_seconds = max(
        (result["steady_elapsed_seconds"] for result in pair_results),
        default=0.0,
    )
    steady_environment_steps = (
        len(pair_results)
        * (args.updates - 1)
        * args.batch_size
        * args.max_steps
    )
    steady_steps_per_second = (
        steady_environment_steps / steady_elapsed_seconds
        if steady_elapsed_seconds > 0
        else 0.0
    )
    full_environment_steps = (
        args.full_jobs * args.full_updates * args.batch_size * args.max_steps
    )
    result = {
        "concurrency": concurrency,
        "elapsed_seconds": elapsed_seconds,
        "steady_elapsed_seconds": steady_elapsed_seconds,
        "steady_environment_steps": steady_environment_steps,
        "steady_env_steps_per_second": steady_steps_per_second,
        "projected_full_experiment_hours": (
            full_environment_steps / steady_steps_per_second / 3600.0
            if steady_steps_per_second > 0
            else None
        ),
        "peak_worker_cuda_memory_mib": max(
            (
                result["peak_cuda_memory_mib"]
                for result in pair_results
                if result["peak_cuda_memory_mib"] is not None
            ),
            default=None,
        ),
        "pair_results": pair_results,
        "failures": failures,
    }
    print(
        f"concurrency={concurrency} elapsed={elapsed_seconds:.2f}s "
        f"steady_steps_per_second={steady_steps_per_second:.1f} "
        f"projected_hours={result['projected_full_experiment_hours']} "
        f"failures={len(failures)}",
        flush=True,
    )
    return result


def _benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    return {**config_payload(args), "task": args.task}


def _validate_args(args: argparse.Namespace) -> None:
    if args.updates < 2:
        raise ValueError("updates must be at least 2 to exclude warm-up")
    if args.full_jobs < 1 or args.full_updates < 1:
        raise ValueError("full-jobs and full-updates must be positive")
    if not parse_positive_ints(args.concurrencies):
        raise ValueError("at least one concurrency is required")


def _print_table(results: list[dict[str, Any]]) -> None:
    print("concurrency steady_steps_per_second projected_full_hours failures", flush=True)
    for result in results:
        projected = result["projected_full_experiment_hours"]
        print(
            f"{result['concurrency']:>11} "
            f"{result['steady_env_steps_per_second']:>23.1f} "
            f"{projected if projected is not None else float('nan'):>20.2f} "
            f"{len(result['failures']):>8}",
            flush=True,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from mouse_run_run.env import GridWorldConfig
from mouse_run_run.train import TrainConfig, train


DEFAULT_FULL_JOBS = 20
DEFAULT_FULL_UPDATES = 20_000
DEFAULT_BATCH_SIZE = 40
DEFAULT_MAX_STEPS = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrencies", default="1,2,4,6,8,10,12")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--task", choices=("social", "non_social"), default="social")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cuda-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subspace-metric-period", type=int, default=1)
    parser.add_argument("--triton-env-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-jobs", type=int, default=DEFAULT_FULL_JOBS)
    parser.add_argument("--full-updates", type=int, default=DEFAULT_FULL_UPDATES)
    parser.add_argument("--output", type=Path, default=Path("runs/rtx3090-multiplex-50u.json"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        _worker()
        return

    results = [
        _run_concurrency(args, concurrency)
        for concurrency in _positive_ints(args.concurrencies)
    ]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "updates": args.updates,
            "task": args.task,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "ppo_epochs": args.ppo_epochs,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "cuda_tf32": args.cuda_tf32,
            "subspace_metric_period": args.subspace_metric_period,
            "triton_env_step": args.triton_env_step,
        },
        "projection": {
            "full_jobs": args.full_jobs,
            "full_updates": args.full_updates,
            "full_environment_steps": (
                args.full_jobs
                * args.full_updates
                * args.batch_size
                * args.max_steps
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_table(results)
    print(f"wrote_benchmark={args.output}", flush=True)


def _worker() -> None:
    payload = json.loads(os.environ["MRR_WORKER_CONFIG"])
    task = payload["task"]
    env = GridWorldConfig(
        grid_size=10,
        vision_radius=3,
        max_steps=payload["max_steps"],
        task=task,
        partner_visibility="none" if task == "non_social" else "partial",
    )
    started = perf_counter()
    metrics = train(
        TrainConfig(
            updates=payload["updates"],
            batch_size=payload["batch_size"],
            hidden_size=payload["hidden_size"],
            learning_rate=payload["learning_rate"],
            ppo_epochs=payload["ppo_epochs"],
            seed=payload["seed"],
            device=payload["device"],
            log_every=payload["log_every"],
            checkpoint=Path(payload["checkpoint"]),
            cuda_tf32=payload["cuda_tf32"],
            subspace_metric_period=payload["subspace_metric_period"],
            triton_env_step=payload["triton_env_step"],
            env=env,
        )
    )
    print(
        "MRR_RESULT="
        + json.dumps(
            {
                "seed": payload["seed"],
                "elapsed_seconds": perf_counter() - started,
                "metrics": asdict(metrics),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_concurrency(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    started = perf_counter()
    benchmark_dir = Path("/tmp/mouse-run-run-benchmark") / f"c{concurrency}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for index in range(concurrency):
        seed = args.seed_start + index
        payload = {
            "task": args.task,
            "seed": seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "ppo_epochs": args.ppo_epochs,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "log_every": args.log_every,
            "cuda_tf32": args.cuda_tf32,
            "subspace_metric_period": args.subspace_metric_period,
            "triton_env_step": args.triton_env_step,
            "checkpoint": str(benchmark_dir / f"seed_{seed:04d}.safetensors"),
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
        if process.returncode != 0:
            failures.append(
                {
                    "seed": seed,
                    "returncode": process.returncode,
                    "output_tail": output[-4000:],
                }
            )
            continue
        result_lines = [
            line.removeprefix("MRR_RESULT=")
            for line in output.splitlines()
            if line.startswith("MRR_RESULT=")
        ]
        if not result_lines:
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
    environment_steps = concurrency * args.updates * args.batch_size * args.max_steps
    env_steps_per_second = environment_steps / elapsed_seconds
    full_environment_steps = args.full_jobs * args.full_updates * args.batch_size * args.max_steps
    result = {
        "concurrency": concurrency,
        "task": args.task,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "environment_steps": environment_steps,
        "elapsed_seconds": elapsed_seconds,
        "env_steps_per_second": env_steps_per_second,
        "projected_full_experiment_hours": full_environment_steps / env_steps_per_second / 3600.0,
        "pair_results": pair_results,
        "failures": failures,
    }
    print(
        f"concurrency={concurrency} "
        f"seconds={elapsed_seconds:.3f} "
        f"env_steps_per_second={env_steps_per_second:.1f} "
        f"projected_hours={result['projected_full_experiment_hours']:.2f} "
        f"failures={len(failures)}",
        flush=True,
    )
    return result


def _positive_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("concurrencies must contain positive integers")
    return values


def _print_table(results: list[dict[str, Any]]) -> None:
    print("concurrency seconds env_steps_per_second projected_full_hours failures", flush=True)
    for result in results:
        print(
            f"{result['concurrency']:>11} "
            f"{result['elapsed_seconds']:>7.2f} "
            f"{result['env_steps_per_second']:>20.1f} "
            f"{result['projected_full_experiment_hours']:>20.2f} "
            f"{len(result['failures']):>8}",
            flush=True,
        )


if __name__ == "__main__":
    main()

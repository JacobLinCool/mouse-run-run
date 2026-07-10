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
from mouse_run_run.env import GridWorldConfig
from mouse_run_run.provenance import collect_provenance, write_json_atomic
from mouse_run_run.train import TrainConfig, train
from mouse_run_run.training_config import apply_preset_defaults, resolve_partner_visibility


# [PAPER-METHODS] reportable panel size. Source IDs resolve in the
# official-dynamics SPEC's "Implementation Source Registry".
DEFAULT_FULL_JOBS = 20  # 10 social + 10 non-social policy pairs.
DEFAULT_FULL_UPDATES = 20_000
DEFAULT_BATCH_SIZE = 40  # 40 complete 100-step episodes = 4,000 env steps/update.
DEFAULT_MAX_STEPS = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    # [LOCAL-CALIBRATION] candidate process multiplexing levels; not a paper
    # hyperparameter and not part of the learner-equivalence claim.
    parser.add_argument("--concurrencies", default="1,2,4,8,10")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--task", choices=("social", "non_social"), default="social")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--spawn-mode",
        choices=("full_grid", "official_exclude_last"),
        default="official_exclude_last",
    )
    parser.add_argument("--architecture", choices=("rnn",), default="rnn")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--preset",
        choices=("modern_fast", "paper_text", "official_code"),
        required=True,
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--value-coef", type=float)
    parser.add_argument("--value-clip", type=float)
    parser.add_argument("--recurrent-l2-coef", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--sgd-minibatch-size", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--kl-coeff", type=float)
    parser.add_argument("--kl-target", type=float)
    parser.add_argument("--learner-mode", choices=("full_batch", "rllib_2_2"))
    parser.add_argument("--rnn-initialization", choices=("modern", "pytorch_default"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cuda-tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finite-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subspace-metric-period", type=int, default=10)
    parser.add_argument("--triton-env-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-agent-rollout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-jobs", type=int, default=DEFAULT_FULL_JOBS)
    parser.add_argument("--full-updates", type=int, default=DEFAULT_FULL_UPDATES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    apply_preset_defaults(args)

    if args.worker:
        _worker()
        return
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
    task = payload["task"]
    env = GridWorldConfig(
        grid_size=10,
        vision_radius=3,
        max_steps=payload["max_steps"],
        task=task,
        partner_visibility=resolve_partner_visibility(task, None),
        spawn_mode=payload["spawn_mode"],
    )
    run_dir = Path(payload["run_dir"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    metrics = train(
        TrainConfig(
            updates=payload["updates"],
            batch_size=payload["batch_size"],
            architecture=payload["architecture"],
            hidden_size=payload["hidden_size"],
            gamma=payload["gamma"],
            gae_lambda=payload["gae_lambda"],
            learning_rate=payload["learning_rate"],
            ppo_epochs=payload["ppo_epochs"],
            clip_epsilon=payload["clip_epsilon"],
            entropy_coef=payload["entropy_coef"],
            value_coef=payload["value_coef"],
            value_clip=payload["value_clip"],
            recurrent_l2_coef=payload["recurrent_l2_coef"],
            grad_clip=payload["grad_clip"],
            sgd_minibatch_size=payload["sgd_minibatch_size"],
            max_seq_len=payload["max_seq_len"],
            kl_coeff=payload["kl_coeff"],
            kl_target=payload["kl_target"],
            learner_mode=payload["learner_mode"],
            rnn_initialization=payload["rnn_initialization"],
            seed=payload["seed"],
            device=payload["device"],
            log_every=1,
            checkpoint=run_dir / "checkpoints" / "latest.safetensors",
            run_dir=run_dir,
            status_every_seconds=3600.0,
            cuda_tf32=payload["cuda_tf32"],
            finite_guard=payload["finite_guard"],
            subspace_metric_period=payload["subspace_metric_period"],
            triton_env_step=payload["triton_env_step"],
            fused_agent_rollout=payload["fused_agent_rollout"],
            env=env,
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
                    [sys.executable, str(Path(__file__).resolve()), "--worker", "--preset", args.preset, "--output", str(args.output)],
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
    return {
        "updates": args.updates,
        "task": args.task,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "spawn_mode": args.spawn_mode,
        "architecture": args.architecture,
        "hidden_size": args.hidden_size,
        "gamma": args.gamma,
        "preset": args.preset,
        "gae_lambda": args.gae_lambda,
        "learning_rate": args.learning_rate,
        "ppo_epochs": args.ppo_epochs,
        "clip_epsilon": args.clip_epsilon,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "value_clip": args.value_clip,
        "recurrent_l2_coef": args.recurrent_l2_coef,
        "grad_clip": args.grad_clip,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "max_seq_len": args.max_seq_len,
        "kl_coeff": args.kl_coeff,
        "kl_target": args.kl_target,
        "learner_mode": args.learner_mode,
        "rnn_initialization": args.rnn_initialization,
        "device": args.device,
        "cuda_tf32": args.cuda_tf32,
        "finite_guard": args.finite_guard,
        "subspace_metric_period": args.subspace_metric_period,
        "triton_env_step": args.triton_env_step,
        "fused_agent_rollout": args.fused_agent_rollout,
    }


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

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from dataclasses import fields

import torch

from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig
from mouse_run_run.provenance import collect_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--vision-radius", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    cases = (
        ("social", "partial"),
        ("non_social", "none"),
        ("social", "full"),
    )
    started = datetime.now(UTC).isoformat()
    for task, partner_visibility in cases:
        config = GridWorldConfig(
            grid_size=args.grid_size,
            vision_radius=args.vision_radius,
            max_steps=args.steps,
            task=task,
            partner_visibility=partner_visibility,
        )
        for seed in range(args.seeds):
            verify_case(
                config=config,
                seed=seed,
                batch_size=args.batch_size,
                steps=args.steps,
                device=device,
            )
        print(f"ok task={task} partner_visibility={partner_visibility} seeds={args.seeds}")
    torch.cuda.synchronize()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ok",
                    "started_at": started,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "command": shlex.join(sys.argv),
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                    "seeds": args.seeds,
                    "grid_size": args.grid_size,
                    "vision_radius": args.vision_radius,
                    "cases": [
                        {"task": task, "partner_visibility": visibility}
                        for task, visibility in cases
                    ],
                    "provenance": collect_provenance(cwd=Path.cwd()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print("triton env rollout equivalence ok")


def verify_case(
    *,
    config: GridWorldConfig,
    seed: int,
    batch_size: int,
    steps: int,
    device: torch.device,
) -> None:
    torch.manual_seed(seed)
    reference = BatchedChaseEnv(config, batch_size, device)
    fused = BatchedChaseEnv(config, batch_size, device)
    reference.reset_state()
    copy_env(reference, fused)
    compare_observations(reference, fused, step=0, config=config)

    chaser_actions = torch.randint(4, (steps, batch_size), device=device)
    explorer_actions = torch.randint(4, (steps, batch_size), device=device)
    for step in range(steps):
        reference_result = reference.step_training(chaser_actions[step], explorer_actions[step])
        fused_result = fused.step_training_fused(chaser_actions[step], explorer_actions[step])
        compare_step_results(reference_result, fused_result, step=step, config=config)
        compare_env_states(reference, fused, step=step, config=config)
        compare_observations(reference, fused, step=step + 1, config=config)


def copy_env(source: BatchedChaseEnv, target: BatchedChaseEnv) -> None:
    target.chaser_position = source.chaser_position.clone()
    target.explorer_position = source.explorer_position.clone()
    target.chaser_visited = source.chaser_visited.clone()
    target.explorer_visited = source.explorer_visited.clone()
    target.done = source.done.clone()
    target.step_count = source.step_count.clone()


def compare_step_results(
    reference_result: object,
    fused_result: object,
    *,
    step: int,
    config: GridWorldConfig,
) -> None:
    for field in fields(type(reference_result)):
        name = field.name
        reference_value = getattr(reference_result, name)
        fused_value = getattr(fused_result, name)
        compare_tensor(
            reference_value,
            fused_value,
            name=f"result.{name}",
            step=step,
            config=config,
        )


def compare_env_states(
    reference: BatchedChaseEnv,
    fused: BatchedChaseEnv,
    *,
    step: int,
    config: GridWorldConfig,
) -> None:
    for name in (
        "chaser_position",
        "explorer_position",
        "chaser_visited",
        "explorer_visited",
        "done",
        "step_count",
    ):
        compare_tensor(
            getattr(reference, name),
            getattr(fused, name),
            name=f"state.{name}",
            step=step,
            config=config,
        )


def compare_observations(
    reference: BatchedChaseEnv,
    fused: BatchedChaseEnv,
    *,
    step: int,
    config: GridWorldConfig,
) -> None:
    reference_chaser, reference_explorer = reference.observations()
    fused_chaser, fused_explorer = fused.observations()
    compare_tensor(
        reference_chaser,
        fused_chaser,
        name="observation.chaser",
        step=step,
        config=config,
    )
    compare_tensor(
        reference_explorer,
        fused_explorer,
        name="observation.explorer",
        step=step,
        config=config,
    )


def compare_tensor(
    reference: torch.Tensor,
    fused: torch.Tensor,
    *,
    name: str,
    step: int,
    config: GridWorldConfig,
) -> None:
    if reference.dtype.is_floating_point:
        if torch.allclose(reference, fused, atol=1e-6, rtol=0.0):
            return
        max_diff = (reference - fused).abs().max().item()
        raise AssertionError(f"{config} step={step} {name} max_diff={max_diff}")
    if torch.equal(reference, fused):
        return
    mismatch_count = int((reference != fused).sum().item())
    raise AssertionError(f"{config} step={step} {name} mismatches={mismatch_count}")


if __name__ == "__main__":
    main()

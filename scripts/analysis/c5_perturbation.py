"""C5: causal test of shared neural dimensions by null-space perturbation.

Reproduces the paper's Fig. 6s-u claim: projecting the chaser's activation
into the null space of the top-10 shared (PLSC) dimensions before computing
its action reduces collisions and partner-in-vision and increases distance,
whereas removing a comparable amount of random (non-shared) variance does not.

For each pair: build the chaser's top-10 shared basis and a top-25 random
control basis from self-play rollouts, then evaluate three chaser conditions
(unperturbed / shared-removed / random-removed) against the trained explorer.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import UTC, datetime
from pathlib import Path

import torch

from _common import ANALYSES_ROOT, REPO_ROOT, RUNS_ROOT, manifest_sidecar, write_manifest
from mouse_run_run.analysis import (
    load_rollout,
    random_variance_basis,
    shared_dimension_basis,
    valid_episode_indices,
)
from mouse_run_run.checkpoint_loading import load_policy_pair
from mouse_run_run.env import BatchedChaseEnv
from mouse_run_run.provenance import collect_provenance, write_json_atomic
from mouse_run_run.training_config import select_device


@torch.no_grad()
def perturbed_eval(
    checkpoint: Path,
    projection: torch.Tensor | None,
    *,
    episodes: int,
    max_steps: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Self-play eval with the chaser's hidden projected by (I - P P^T) before
    its action head. projection is the (H, k) removed basis, or None for the
    unperturbed control."""
    torch.manual_seed(seed)
    pair = load_policy_pair(checkpoint, device=device, max_steps=max_steps)
    env_config = pair.env_config
    chaser = pair.chaser
    explorer = pair.explorer

    P = projection.to(device) if projection is not None else None

    def remove(h: torch.Tensor) -> torch.Tensor:
        if P is None:
            return h
        return h - (h @ P) @ P.T

    env = BatchedChaseEnv(env_config, episodes, device)
    chaser_obs, explorer_obs = env.reset()
    chaser_state_h = chaser.initial_hidden(episodes, device)
    explorer_state_h = explorer.initial_hidden(episodes, device)
    collisions = torch.zeros(episodes, device=device)
    vision = torch.zeros(episodes, device=device)
    distance_sum = torch.zeros(episodes, device=device)

    for _ in range(max_steps):
        c_out = chaser(chaser_obs, chaser_state_h)
        e_out = explorer(explorer_obs, explorer_state_h)
        # Remove the shared dimensions from the chaser's action-driving hidden.
        c_logits = chaser.action_layer(remove(c_out.hidden))
        c_action = torch.distributions.Categorical(logits=c_logits).sample()
        e_action = torch.distributions.Categorical(logits=e_out.logits).sample()
        result = env.step(c_action, e_action)
        collisions += result.collision.float()
        vision += result.chaser_partner_visible.float()
        distance_sum += result.distance
        chaser_state_h = c_out.state
        explorer_state_h = e_out.state
        chaser_obs = result.chaser_observation
        explorer_obs = result.explorer_observation

    return {
        "collisions_per_episode": float(collisions.mean().item()),
        "partner_in_vision": float((vision / max_steps).mean().item()),
        "average_distance": float((distance_sum / max_steps).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_root", type=Path, nargs="?", default=RUNS_ROOT / "mouse-run-run-1" / "paper_rollouts")
    parser.add_argument("--output", type=Path, default=ANALYSES_ROOT / "c5_perturbation.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-k", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = select_device(args.device)
    units = []
    inputs: list[Path] = []
    for ro in sorted(args.rollout_root.glob("*.safetensors")):
        meta, tensors = load_rollout(ro)
        inputs.append(ro)
        config = meta.get("checkpoint_config") or {}
        task = (config.get("env") or {}).get("task")
        if task != "social":
            continue
        if len(valid_episode_indices(tensors)) < 3:
            continue
        checkpoint = Path(meta["checkpoint"])
        if not checkpoint.is_absolute():
            # Rollout metadata records repo-root-relative checkpoint paths.
            checkpoint = REPO_ROOT / checkpoint
        inputs.append(checkpoint)
        shared = torch.tensor(shared_dimension_basis(tensors, top_k=args.top_k), dtype=torch.float32)
        control = torch.tensor(random_variance_basis(tensors, top_k=args.random_k, seed=args.seed), dtype=torch.float32)
        base = perturbed_eval(checkpoint, None, episodes=args.episodes, max_steps=args.max_steps, device=device, seed=args.seed)
        pert = perturbed_eval(checkpoint, shared, episodes=args.episodes, max_steps=args.max_steps, device=device, seed=args.seed)
        rand = perturbed_eval(checkpoint, control, episodes=args.episodes, max_steps=args.max_steps, device=device, seed=args.seed)
        seed = config.get("seed")
        units.append({"seed": seed, "unperturbed": base, "shared_removed": pert, "random_removed": rand})
        print(
            f"seed_{int(seed):04d}: collisions base={base['collisions_per_episode']:.1f} "
            f"shared={pert['collisions_per_episode']:.1f} random={rand['collisions_per_episode']:.1f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "rollout_root": str(args.rollout_root),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "top_k": args.top_k,
        "random_k": args.random_k,
        "seed": args.seed,
        "units": units,
        "summary": _summarize(units),
        "provenance": collect_provenance(cwd=REPO_ROOT),
    }
    write_json_atomic(args.output, payload)
    write_manifest(
        manifest_sidecar(args.output),
        script=Path(__file__),
        inputs=inputs,
        outputs={"analysis": args.output},
        extra={
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "top_k": args.top_k,
            "random_k": args.random_k,
            "seed": args.seed,
        },
    )
    print(json.dumps(payload["summary"], indent=2))


def _summarize(units: list[dict]) -> dict:
    def col(cond, key):
        return [u[cond][key] for u in units]

    out = {"n_units": len(units)}
    for key in ("collisions_per_episode", "partner_in_vision", "average_distance"):
        base = col("unperturbed", key)
        shared = col("shared_removed", key)
        rand = col("random_removed", key)
        out[key] = {
            "unperturbed": _ms(base),
            "shared_removed": _ms(shared),
            "random_removed": _ms(rand),
            "shared_delta": _ms([s - b for s, b in zip(shared, base)]),
            "random_delta": _ms([r - b for r, b in zip(rand, base)]),
        }
    return out


def _ms(v: list[float]) -> dict:
    if not v:
        return {"mean": None, "std": None}
    return {"mean": st.fmean(v), "std": st.pstdev(v) if len(v) > 1 else 0.0}


if __name__ == "__main__":
    main()

"""C4: partner representation in the chaser's neural action subspace.

Reproduces the paper's Fig. 6p-r claims for the RNN agents:
- partner representation increases over training for social chasers and not
  for non-social chasers (trend over update_* checkpoints);
- final partner representation correlates with collision performance across
  seeds (social only).

Collects fresh self-play rollouts per checkpoint (in-memory) and computes the
non-redundant variance of the neural action subspace explained by the
partner's behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mouse_run_run.analysis import chaser_action_weight, partner_representation
from mouse_run_run.env import GridWorldConfig
from mouse_run_run.policy import build_policy
from mouse_run_run.provenance import collect_provenance, write_json_atomic
from mouse_run_run.rollout import _collect_batch
from mouse_run_run.serialization import load_checkpoint
from mouse_run_run.train import select_device


@torch.no_grad()
def measure_checkpoint(
    checkpoint: Path,
    *,
    episodes: int,
    max_steps: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    config, _, chaser_state, explorer_state = load_checkpoint(checkpoint)
    env_config = replace(GridWorldConfig(**config["env"]), max_steps=max_steps)
    chaser = build_policy(config.get("architecture", "rnn"), env_config.observation_size, hidden_size=config["hidden_size"]).to(device)
    explorer = build_policy(config.get("architecture", "rnn"), env_config.observation_size, hidden_size=config["hidden_size"]).to(device)
    chaser.load_state_dict(chaser_state)
    explorer.load_state_dict(explorer_state)
    chaser.eval()
    explorer.eval()
    tensors = _collect_batch(
        env_config=env_config,
        batch_size=episodes,
        chaser=chaser,
        explorer=explorer,
        device=device,
        deterministic=False,
        opponent_mode="self_play",
        degenerate_threshold_fraction=0.01,
    )
    tensors = {k: v.cpu() for k, v in tensors.items()}
    Wa = chaser_action_weight(checkpoint)
    pr = partner_representation(tensors, Wa, grid_size=env_config.grid_size, seed=seed)
    collisions = float(tensors["collision"].float().sum(0).mean().item())
    return {
        "partner_unique": pr.partner_unique,
        "r2_full": pr.r2_full,
        "collisions_per_episode": collisions,
        "n_valid_episodes": pr.n_valid_episodes,
    }


def _update_of(path: Path) -> int:
    m = re.search(r"update_(\d+)", path.stem)
    if m:
        return int(m.group(1))
    return 10**9  # latest sorts last


def _training_checkpoints(unit_dir: Path, points: int) -> list[Path]:
    ckpts = sorted(
        (unit_dir / "checkpoints").glob("*.safetensors"), key=_update_of
    )
    if not ckpts:
        return []
    updates = [c for c in ckpts if "update_" in c.stem]
    latest = [c for c in ckpts if c.stem in ("latest", "final")]
    if len(updates) <= points:
        chosen = updates
    else:
        idx = np.linspace(0, len(updates) - 1, points).round().astype(int)
        chosen = [updates[i] for i in sorted(set(idx))]
    return chosen + latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path, nargs="?", default=Path("runs/mouse-run-run-1"))
    parser.add_argument("--output", type=Path, default=Path("runs/analyses/c4_partner_representation.json"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--trend-points", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = select_device(args.device)
    results = {"social": [], "non_social": []}
    for task in ("social", "non_social"):
        for unit_dir in sorted((args.experiment_root / task).glob("seed_*/attempt_01")):
            seed = int(unit_dir.parent.name.removeprefix("seed_"))
            trend = []
            for ckpt in _training_checkpoints(unit_dir, args.trend_points):
                m = measure_checkpoint(
                    ckpt, episodes=args.episodes, max_steps=args.max_steps, device=device, seed=args.seed
                )
                m["update"] = _update_of(ckpt)
                m["is_latest"] = ckpt.stem in ("latest", "final")
                trend.append(m)
                print(
                    f"{task}/seed_{seed:04d} u={m['update']}: partner_unique={m['partner_unique']:.4f} coll={m['collisions_per_episode']:.1f}",
                    flush=True,
                )
            results[task].append({"seed": seed, "trend": trend})

    summary = _summarize(results)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_root": str(args.experiment_root),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "results": results,
        "summary": summary,
        "provenance": collect_provenance(cwd=Path.cwd()),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(summary, indent=2))


def _summarize(results: dict) -> dict:
    out = {}
    for task, units in results.items():
        firsts, lasts, coll = [], [], []
        for u in units:
            trend = [p for p in u["trend"] if p["n_valid_episodes"] >= 2]
            if len(trend) < 2:
                continue
            firsts.append(trend[0]["partner_unique"])
            lasts.append(trend[-1]["partner_unique"])
            coll.append(trend[-1]["collisions_per_episode"])
        if not lasts:
            continue
        firsts, lasts, coll = np.array(firsts), np.array(lasts), np.array(coll)
        # C4(b): correlation of final partner representation with collisions.
        corr = (
            float(np.corrcoef(lasts, coll)[0, 1]) if len(lasts) > 2 and lasts.std() > 0 and coll.std() > 0 else None
        )
        out[task] = {
            "n_units": len(lasts),
            "partner_unique_early_mean": float(firsts.mean()),
            "partner_unique_late_mean": float(lasts.mean()),
            "delta_early_to_late": float((lasts - firsts).mean()),
            "corr_partner_vs_collisions": corr,
        }
    return out


if __name__ == "__main__":
    main()

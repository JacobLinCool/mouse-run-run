"""Per-unit residualized PLSC top_r vs behavioral coupling (vision), all archs."""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from _common import (
    ANALYSES_ROOT,
    CROSS_ARCH_EXPERIMENTS,
    RUNS_ROOT,
    manifest_sidecar,
    write_manifest,
)
from mouse_run_run.analysis import episode_hidden_pairs, load_rollout, plsc_shared_dimensions
from mouse_run_run.provenance import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=ANALYSES_ROOT / "cross_arch_perunit.json")
    parser.add_argument("--permutations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inputs: list[Path] = []
    rows = []
    for name, exp in CROSS_ARCH_EXPERIMENTS.items():
        root = args.runs_root / exp / "paper_rollouts"
        for f in sorted(root.glob("*.safetensors")):
            meta, t = load_rollout(f)
            inputs.append(f)
            cfg = meta.get("checkpoint_config", {})
            env = cfg.get("env") or {}
            task = env.get("task")
            seed = cfg.get("seed")
            ch, ex = episode_hidden_pairs(t)
            vis = t["chaser_partner_visible"].float().mean().item()
            coll = t["collision"].float().sum(0).mean().item()
            if len(ch) < 3:
                rows.append(
                    dict(
                        arch=name,
                        task=task,
                        seed=seed,
                        status="degenerate",
                        n_valid=len(ch),
                        vision=vis,
                        collisions=coll,
                        top_r=None,
                    )
                )
                continue
            r = plsc_shared_dimensions(
                ch, ex, permutations=args.permutations, seed=args.seed, subtract_time_mean=True
            )
            rows.append(
                dict(
                    arch=name,
                    task=task,
                    seed=seed,
                    status="ok",
                    n_valid=len(ch),
                    vision=vis,
                    collisions=coll,
                    top_r=r.top_dim_correlation,
                )
            )
        print(f"{name} done", flush=True)
    write_json_atomic(args.output, rows)
    write_manifest(
        manifest_sidecar(args.output),
        script=Path(__file__),
        inputs=inputs,
        outputs={"per_unit": args.output},
        extra={"permutations": args.permutations, "seed": args.seed},
    )

    print("\n=== SOCIAL units: residualized top_r vs vision (sorted by vision) ===")
    for name in CROSS_ARCH_EXPERIMENTS:
        soc = [r for r in rows if r["arch"] == name and r["task"] == "social" and r["status"] == "ok"]
        soc.sort(key=lambda r: r["vision"])
        lo = [r for r in soc if r["vision"] < 0.4]
        hi = [r for r in soc if r["vision"] >= 0.4]

        def mm(g): return f"{st.fmean([r['top_r'] for r in g]):.3f} (n={len(g)})" if g else "—"

        ndeg = len([r for r in rows if r["arch"] == name and r["task"] == "social" and r["status"] == "degenerate"])
        print(f"{name:12} low-vision(<40%): {mm(lo):18} high-vision(>=40%): {mm(hi):18} degenerate={ndeg}")


if __name__ == "__main__":
    main()

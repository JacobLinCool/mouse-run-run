"""Fast cross-architecture PLSC on time-mean-subtracted hidden states."""

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
    parser.add_argument(
        "--output", type=Path, default=ANALYSES_ROOT / "cross_arch_plsc_residualized.json"
    )
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inputs: list[Path] = []
    out = {}
    for name, exp in CROSS_ARCH_EXPERIMENTS.items():
        root = args.runs_root / exp / "paper_rollouts"
        if not root.exists():
            continue
        per_task = {"social": [], "non_social": []}
        for f in sorted(root.glob("*.safetensors")):
            meta, t = load_rollout(f)
            inputs.append(f)
            cfg = meta.get("checkpoint_config", {})
            task = (cfg.get("env") or {}).get("task")
            ch, ex = episode_hidden_pairs(t)
            if len(ch) < 3:
                continue
            # equal-length episodes required; paper rollouts are all 500 steps
            r = plsc_shared_dimensions(
                ch, ex, permutations=args.permutations, seed=args.seed, subtract_time_mean=True
            )
            per_task[task].append((r.top_dim_correlation, r.n_significant, r.n_episodes))
        out[name] = per_task
        for task in ("social", "non_social"):
            vals = per_task[task]
            if vals:
                tr = [v[0] for v in vals]
                ns = [v[1] for v in vals]
                print(f"{name:12} {task:11} residualized top_r={st.fmean(tr):.3f}±{st.pstdev(tr):.3f}  n_sig={st.fmean(ns):.0f}  units={len(vals)}", flush=True)
    write_json_atomic(args.output, out)
    write_manifest(
        manifest_sidecar(args.output),
        script=Path(__file__),
        inputs=inputs,
        outputs={"plsc": args.output},
        extra={"permutations": args.permutations, "seed": args.seed},
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

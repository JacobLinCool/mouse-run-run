"""Cross-architecture linear CKA on a common observation probe set.

Feed the SAME game-state observation stream (RNN social rollouts, non-
degenerate) through every architecture's social chaser networks and compare
the resulting representations. CKA is invariant to rotation/scaling, so it
measures whether architectures encode the shared task the same way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import (
    ANALYSES_ROOT,
    CROSS_ARCH_EXPERIMENTS,
    RUNS_ROOT,
    manifest_sidecar,
    write_manifest,
)
from mouse_run_run.analysis import linear_cka, load_rollout, replay_hidden, valid_episode_indices
from mouse_run_run.provenance import write_json_atomic

PROBE_FILE_PATTERN = "*social__seed_000[012]*"
PROBE_FILE_COUNT = 3
PROBE_EPISODES_PER_FILE = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument(
        "--probe-experiment",
        default=CROSS_ARCH_EXPERIMENTS["RNN"],
        help="Experiment whose social rollouts provide the common observation probe.",
    )
    parser.add_argument("--output", type=Path, default=ANALYSES_ROOT / "cross_arch_cka.json")
    args = parser.parse_args()

    # Common probe: chaser observations from 3 RNN social rollouts (non-degenerate).
    probe_root = args.runs_root / args.probe_experiment / "paper_rollouts"
    probe_files = sorted(probe_root.glob(PROBE_FILE_PATTERN))[:PROBE_FILE_COUNT]
    probes = []
    for f in probe_files:
        _, t = load_rollout(f)
        idx = valid_episode_indices(t)[:PROBE_EPISODES_PER_FILE]
        probes.append(t["chaser_observation"][:, idx])
    probe = torch.cat(probes, dim=1)  # (T, E, 200)
    print(f"probe: {probe.shape}", flush=True)

    # For each architecture, replay the probe through one social checkpoint.
    checkpoints = {}
    reps = {}
    for name, exp in CROSS_ARCH_EXPERIMENTS.items():
        ck = args.runs_root / exp / "social/seed_0000/attempt_01/checkpoints/latest.safetensors"
        if not ck.exists():
            ck = next(
                (args.runs_root / exp).glob("social/seed_*/attempt_*/checkpoints/latest.safetensors")
            )
        checkpoints[name] = ck
        h = replay_hidden(ck, probe, agent="chaser")
        reps[name] = h.reshape(-1, h.shape[-1]).double().numpy()
        print(f"{name}: {reps[name].shape}", flush=True)

    names = list(CROSS_ARCH_EXPERIMENTS)
    print("\n=== cross-architecture linear CKA (chaser, common probe) ===")
    print("           " + "  ".join(f"{n[:5]:>6}" for n in names))
    mat = {}
    for a in names:
        row = []
        for b in names:
            c = linear_cka(reps[a], reps[b])
            row.append(c)
            mat[f"{a}|{b}"] = c
        print(f"{a:11}" + "  ".join(f"{v:6.3f}" for v in row))
    write_json_atomic(args.output, mat)
    write_manifest(
        manifest_sidecar(args.output),
        script=Path(__file__),
        inputs=[*probe_files, *checkpoints.values()],
        outputs={"cka": args.output},
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

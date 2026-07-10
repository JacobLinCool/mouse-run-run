"""Cross-architecture linear CKA on a common observation probe set.

Feed the SAME game-state observation stream (RNN social rollouts, non-
degenerate) through every architecture's social chaser networks and compare
the resulting representations. CKA is invariant to rotation/scaling, so it
measures whether architectures encode the shared task the same way.
"""
import json
from pathlib import Path

import torch

from mouse_run_run.analysis import linear_cka, load_rollout, replay_hidden, valid_episode_indices

EXPS = {
    "RNN": "mouse-run-run-1",
    "MLP": "mouse-run-run-2-mlp",
    "SSM": "mouse-run-run-2-ssm",
    "Transformer": "mouse-run-run-2-transformer",
}
# Common probe: chaser observations from 3 RNN social rollouts (non-degenerate).
probe_files = sorted(Path("runs/mouse-run-run-1/paper_rollouts").glob("*social__seed_000[012]*"))[:3]
probes = []
for f in probe_files:
    _, t = load_rollout(f)
    idx = valid_episode_indices(t)[:8]
    probes.append(t["chaser_observation"][:, idx])
probe = torch.cat(probes, dim=1)  # (T, E, 200)
print(f"probe: {probe.shape}", flush=True)

# For each architecture, average replayed hidden across its 3 matched social seeds.
reps = {}
for name, exp in EXPS.items():
    ck = Path(f"runs/{exp}/social/seed_0000/attempt_01/checkpoints/latest.safetensors")
    if not ck.exists():
        ck = next(Path(f"runs/{exp}").glob("social/seed_*/attempt_*/checkpoints/latest.safetensors"))
    h = replay_hidden(ck, probe, agent="chaser")
    reps[name] = h.reshape(-1, h.shape[-1]).double().numpy()
    print(f"{name}: {reps[name].shape}", flush=True)

names = list(EXPS)
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
json.dump(mat, open("runs/analyses/cross_arch_cka.json", "w"), indent=1)

# Paper-scale runtime

## Target workload

The main chaser/explorer MARL workload is:

- 2 tasks: social and non-social
- 10 random seeds per task
- 20 independent pairs total
- 20,000 PPO updates per pair
- 40 episodes per update
- 100 timesteps per episode
- 4,000 environment steps per update
- 80,000,000 environment steps per pair
- 1,600,000,000 environment steps for all 20 pairs

## Measured wall-time projections

Current code (2026-07-02 performance round), measured on a rented RTX 4070 Ti
SUPER, 200-update runs at the primary configuration (`subspace_metric_period=10`,
Triton env step, fused agent rollout, TF32, finite guard on), concurrency 10:

| variant | agg. steps/s | projected 20-pair total |
|---|---:|---:|
| pre-review code (archived opt-v2) | 90.5k | 4.91 h |
| review fixes (branch-free obs, per-step-guard accumulation) | 108.9k | 4.08 h |
| + lean finite guard (post-rollout validation only) | 129.6k | 3.43 h |
| + fused agent rollout (**primary configuration**) | **142.9k** | **3.11 h** |

Older reference points from `runs-archive/pre-formal-20260701T193246Z`:
RTX 3090 at concurrency 10 measured ~96.9k steps/s on the pre-review code
(~4.6 h projected); c10 was ~10% faster than c6, hence `"concurrency": 10`.

Evaluated and not adopted: `torch.compile` on the fused rollout step (40.2 ms
vs 38.4 ms eager per c1 rollout — slower), `mode="reduce-overhead"` (CUDA-graph
aliasing conflict with the recurrent state), and manual whole-rollout CUDA
graphs (would require an in-place env-state refactor for static addresses;
bounded upside ~10-15% at c1, less at c10 where ten processes already overlap
kernel launches).

Per-attempt timeout is 24 h (`attempt_timeout_hours`), far above the ~3-5 h
expected upper bound for a single pair under full contention; a hung worker
fails the attempt rather than stalling the experiment.

## Current execution path

Use `scripts/run_local_experiment.py` on the selected CUDA host. This is the
canonical batch runner for RunPod and local CUDA boxes.

Smoke test:

```bash
uv run python scripts/run/launch_gate.py \
  --config experiments/paper_marl_2026/configs/smoke.json \
  --allow-smoke
uv run python scripts/run_local_experiment.py \
  --config experiments/paper_marl_2026/configs/smoke.json
```

Paper-scale launch gate:

```bash
uv run python scripts/verify_triton_env.py \
  --output runs/mouse-run-run-0701/triton_equivalence.json
uv run python scripts/run/launch_gate.py \
  --config experiments/paper_marl_2026/configs/primary.json \
  --triton-equivalence runs/mouse-run-run-0701/triton_equivalence.json
```

Paper-scale run:

```bash
uv run python scripts/run_local_experiment.py \
  --config experiments/paper_marl_2026/configs/primary.json
```

The primary config uses `paper_text`, which includes the recurrent L2 penalty
described in the paper text while keeping the fast Triton environment step.

## Outputs

Each job writes to:

```text
runs/<experiment>/<task>/seed_XXXX/attempt_XX/
```

Expected files:

- `config.json`: saved training config
- `status.json`: latest status, ETA, cost, and metrics
- `metrics.jsonl`: append-only training metric snapshots
- `tensorboard/`: TensorBoard event files
- `checkpoints/latest.safetensors`: latest/final model checkpoint
- `checkpoints/update_*.safetensors`: periodic checkpoints

The runner counts a unit as completed only when the attempt status is
`completed` and `checkpoints/latest.safetensors` passes finite tensor
validation. Genuinely failed attempts remain in `raw_records.jsonl` and can be
retried up to `max_attempts`; interrupted attempts do not consume attempts and
resume from the newest checkpoint with training state.

## Evaluation commands

Paper-style random-opponent evaluation:

```bash
uv run evaluate-paper-marl \
  runs/<experiment> \
  --output runs/<experiment>/paper_random_eval.jsonl \
  --device cuda
```

Neural-analysis rollout collection:

```bash
uv run collect-paper-marl-rollouts \
  runs/<experiment> \
  --output-root runs/<experiment>/paper_rollouts \
  --device cuda
```

The rollout output stores hidden states, actions, positions, events, and
degenerate-episode masks without dropping raw episodes.

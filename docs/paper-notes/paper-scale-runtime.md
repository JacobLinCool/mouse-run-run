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

From `runs-archive/pre-formal-20260701T193246Z` benchmark records
(200-update runs at the primary configuration, `subspace_metric_period=10`,
Triton env step, TF32):

- RTX 3090, concurrency 10: ~96.9k aggregate env steps/s
  → all 20 pairs in ~4.6 hours (~9.7k steps/s per pair).
- RTX 3090 baseline scaling showed concurrency 10 about 10% faster than
  concurrency 6, hence `"concurrency": 10` in `primary.json`.
- RTX 4070 Ti SUPER, concurrency 10: projected ~4.9 hours total.

Per-attempt timeout is 24 h (`attempt_timeout_hours`), roughly 5x the
expected ~4-5 h upper bound for a single pair under full contention; a hung
worker fails the attempt rather than stalling the experiment.

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

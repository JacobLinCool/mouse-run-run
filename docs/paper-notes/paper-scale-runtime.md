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
validation. Failed attempts remain in `raw_records.jsonl` and can be retried.

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

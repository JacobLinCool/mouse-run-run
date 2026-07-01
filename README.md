# mouse-run-run

Modern PyTorch reproduction scaffold for the chaser/explorer MARL experiment
from Zhang et al. (2025).

## Setup

```bash
uv sync
```

## Verify

```bash
uv run python scripts/verify_torch.py
```

The verification script checks the installed Python and PyTorch versions, reports MPS availability on macOS, and runs the RNN hidden-state projection used for the neural action subspace:

```text
h_t = ReLU(W_input x_t + b_input + W_rec h_{t-1} + b_rec)
projection = W_action.T W_action h_t
```

## Train

```bash
uv run train-marl --updates 200 --batch-size 40 --device cpu --run-dir runs/modern-social --checkpoint runs/modern-social/checkpoints/latest.safetensors
```

The training script runs a batched chaser/explorer grid-world experiment with
two independent recurrent PPO actor-critic policies. The environment follows the
paper's main MARL task:

- 10x10 grid world
- 7x7 partner vision
- 100-timestep episodes
- Supplementary Table 3 rewards
- 256-unit ReLU RNN policies
- 4 actions: up, right, down, left

Each observation is a 2-channel 10x10 map flattened to the 200-dimensional input
shown in the paper:

- channel 0: the agent's own position
- channel 1: the other agent's position when inside the vision radius

Train the non-social control:

```bash
uv run train-marl --task non_social --updates 200 --batch-size 40 --device cpu --run-dir runs/modern-nonsocial --checkpoint runs/modern-nonsocial/checkpoints/latest.safetensors
```

Paper-scale training uses `--updates 20000 --batch-size 40 --max-steps 100`.
That is `80,000,000` environment steps for one agent pair.

Fast training remains the default. To add the paper-text recurrent L2 penalty
without changing the fast environment path, use `--preset paper_text`. The
`official_code` preset exposes the stronger recurrent L2 and PPO clip settings
found in the released code path.

Validate checkpoints:

```bash
uv run validate-marl-checkpoints runs/modern-social
```

Evaluate a checkpoint:

```bash
uv run evaluate-marl runs/modern-social/checkpoints/latest.safetensors --episodes 512 --device auto --stochastic --opponent self_play
uv run evaluate-marl runs/modern-social/checkpoints/latest.safetensors --episodes 512 --device auto --stochastic --opponent random_explorer
uv run evaluate-paper-marl runs/modern-social/checkpoints/latest.safetensors --output runs/raw/paper-eval.jsonl
```

Collect rollout tensors for neural analyses:

```bash
uv run collect-marl-rollouts runs/modern-social/checkpoints/latest.safetensors --episodes 128 --output runs/modern-social-rollouts.safetensors --device auto
uv run collect-marl-rollouts runs/modern-social/checkpoints/latest.safetensors --paper-analysis --output runs/modern-social-analysis-25x500.safetensors --device auto
uv run collect-paper-marl-rollouts runs/modern-social --output-root runs/modern-social-paper-rollouts
```

Open an interactive trajectory viewer for saved checkpoints:

```bash
uv run visualize-marl --runs-root runs --port 8765
```

Checkpoints and rollout files use `safetensors`. Config and metrics are stored
as JSON metadata; model weights and rollout arrays are stored as tensor payloads.
The rollout file stores hidden states, observations, actions, positions, rewards,
collision events, approach/escape events, visibility flags, and new field events
with shapes such as `(timesteps, episodes, hidden_size)`. Analysis rollouts also
store `episode_degenerate` and stationary-step counts so degenerate episodes can
be excluded downstream without rewriting raw evidence.

## Formal Experiment

The paper-scale experiment source of truth is:

- `experiments/paper_marl_2026/SPEC.md`
- `experiments/paper_marl_2026/configs/primary.json`
- `experiments/paper_marl_2026/configs/smoke.json`

Run the local smoke gate and smoke training:

```bash
uv run python scripts/run/launch_gate.py --config experiments/paper_marl_2026/configs/smoke.json --allow-smoke
uv run python scripts/run_local_experiment.py --config experiments/paper_marl_2026/configs/smoke.json
```

On the CUDA host, verify Triton env-only equivalence before launch:

```bash
uv run python scripts/verify_triton_env.py --output runs/mouse-run-run-0701/triton_equivalence.json
uv run python scripts/run/launch_gate.py \
  --config experiments/paper_marl_2026/configs/primary.json \
  --triton-equivalence runs/mouse-run-run-0701/triton_equivalence.json
```

Run the paper-scale 20-pair experiment:

```bash
uv run python scripts/run_local_experiment.py --config experiments/paper_marl_2026/configs/primary.json
```

Each attempt writes artifacts under
`runs/<experiment>/<task>/seed_XXXX/attempt_XX/`:

- `status.json`: latest running/completed/failed status, ETA, cost, and metrics
- `metrics.jsonl`: append-only training metric snapshots
- `tensorboard/`: TensorBoard event files
- `checkpoints/latest.safetensors`: final/latest model checkpoint
- `checkpoints/update_*.safetensors`: periodic model checkpoints

The batch runner treats the trained pair as the experimental unit and the
attempt as raw evidence. A unit is completed only when `status.json` is
`completed` and `checkpoints/latest.safetensors` passes finite checkpoint
validation. Failed attempts are kept in `raw_records.jsonl` and retried up to
`max_attempts`.

Generate paper-style evaluation, analysis rollouts, canonical tables, and a
summary report:

```bash
uv run evaluate-paper-marl runs/mouse-run-run-0701 \
  --output runs/mouse-run-run-0701/paper_random_eval.jsonl \
  --device cuda
uv run collect-paper-marl-rollouts runs/mouse-run-run-0701 \
  --output-root runs/mouse-run-run-0701/paper_rollouts \
  --device cuda
uv run python scripts/transform/build_tables.py runs/mouse-run-run-0701
uv run python scripts/analysis/summarize_experiment.py runs/tables/mouse-run-run-0701
```

Upload artifacts to Hugging Face:

```bash
uv run upload-marl-artifacts runs/mouse-run-run-0701 \
  --repo-id JacobLinCool/mouse-run-run-0701 \
  --repo-type dataset \
  --dry-run
uv run upload-marl-artifacts runs/mouse-run-run-0701 \
  --repo-id JacobLinCool/mouse-run-run-0701 \
  --repo-type dataset
```

## Local Runtime Estimate

Short local benchmarks on this implementation:

- MPS: 10 updates took 17.8 seconds, about 2.25k environment steps/second.
- CPU: 5 updates took 2.17 seconds, about 9.2k environment steps/second.

At paper scale, this estimates:

- MPS: about 9.9 hours per pair, or about 8.2 days for 20 pairs sequentially.
- CPU: about 2.4 hours per pair, or about 2.0 days for 20 pairs sequentially.

This environment is still loop-heavy and uses small recurrent networks, so MPS
dispatch overhead dominates. Use CPU locally unless the environment step and PPO
update are further vectorized or compiled.

See `docs/paper-notes/paper-scale-runtime.md` for the full workload math and
current CUDA execution path.

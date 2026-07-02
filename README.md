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

Besides the grid-world playback, the viewer shows the network activity behind
every step: a peak-sorted units × time activation raster per agent, the hidden
state trajectory in PC1–PC2 space (partner-visible steps ringed in amber), and
value estimate / action-subspace norm / distance timelines, plus a 16×16
snapshot of the current hidden activation. All charts share the playback
cursor, are clickable to seek, and the transport answers to the keyboard
(Space play/pause, ←/→ step, Shift for ±5, Home/End).

The **Shared & Unique Subspace (PLSC)** panel makes the paper's cross-agent
decomposition (Fig. 5/6, C3) interactive. Press Compute and the viewer pools
self-play episodes for the loaded checkpoint, excludes degenerate ones, builds
the z-scored cross-covariance of the two agents' hidden states, and shows the
computation as a five-step pipeline with live numbers: pooled samples →
`R = Xᶜᵀ Xᵉ / (n−1)` → SVD + temporal-permutation null → shared subspace
(significant dimensions) → unique complement. It renders the singular-value
spectrum against the null band, the cross-covariance heatmap, each agent's
shared-vs-unique variance split, and one pooled episode's activity norm
decomposed into shared and unique parts over time (cursor-linked to playback).
It uses the same `mouse_run_run.plsc` core as the offline
`analyze_shared_neural.py`, and warns when the pool is rank-deficient
(samples < hidden units), which happens on short smoke checkpoints but not at
the paper's 25×500 analysis scale. Query parameters
(`?checkpoint=...&seed=3&deterministic=false&t=18`) preload and auto-play a
trajectory, which makes states linkable and screenshotable (`&ss=1` also
computes the PLSC subspace panel).

Checkpoints and rollout files use `safetensors`. Config and metrics are stored
as JSON metadata; model weights and rollout arrays are stored as tensor payloads.
The rollout file stores hidden states, observations, actions, positions, rewards,
collision events (joint and per-agent), approach/escape events, visibility flags,
and new field events. Time alignment (rollout schema v3): state-series tensors
(`*_position`, `distance`, `*_partner_visible`) have length `max_steps + 1` and
index `t` is state `s_t`; `observations`/`hidden`/`actions` at index `t` are
computed from `s_t` (`hidden[t]` produced `action[t]`); rewards and event flags
at index `t` describe the transition `s_t -> s_{t+1}`. Analysis rollouts also
store `episode_degenerate` and stuck-run counts so degenerate episodes can be
excluded downstream without rewriting raw evidence. Degenerate episodes are
those whose longest consecutive stuck run (stationary with a valid action,
excluding collision-blocked steps) exceeds 1% of the episode length.

Checkpoints saved with training state (optimizer, RNG, update index) support
resuming interrupted training:

```bash
uv run train-marl --updates 20000 ... --resume-from runs/modern-social/checkpoints/update_012000.safetensors
```

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
validation. Genuinely failed attempts (NaN/Inf, crash, invalid checkpoint,
timeout) are kept in `raw_records.jsonl` and retried up to `max_attempts`.
Interrupted attempts (runner shutdown, host reboot) do not count toward
`max_attempts`; the next invocation resumes the unit from the newest healthy
checkpoint with training state, bounded by `max_total_attempts`. Workers are
killed after `attempt_timeout_hours` wall-clock hours (0 disables). Re-running
with `resume: false` refuses to touch a non-empty experiment directory unless
`--force-fresh` archives it first.

Generate paper-style evaluation, analysis rollouts, canonical tables, and a
summary report. When given a directory, both tools default to the latest
successful attempt's `checkpoints/latest.safetensors` per unit (the SPEC's
analysis unit); pass `--all-checkpoints` to include mid-training `update_*`
checkpoints, and `--seed` (default 0) makes every checkpoint face the same
standardized random opponent. Already-evaluated checkpoints are skipped unless
`--force`:

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

`summarize_experiment.py` writes `runs/reports/<experiment>/report.md` with
mean ± std tables per task, plus `figures/training_curves.png` (per-update
mean ± std bands across valid pairs, social vs non_social) and
`figures/evaluation_comparison.png` (random-opponent metrics grouped by
opponent mode and task). Pass `--no-figures` for a tables-only report.

Run the network-activation analysis over the paper rollouts:

```bash
uv run python scripts/analysis/analyze_neural.py runs/mouse-run-run-0701/paper_rollouts \
  --output-root runs/reports/mouse-run-run-0701/neural
```

For each rollout (degenerate episodes excluded, matching the paper rule) this
produces a per-checkpoint figure — peak-sorted activation rasters, PCA
variance spectrum, the PC1–PC2 hidden-state embedding colored by
chaser-explorer distance, and collision-triggered hidden-state speed — plus a
social vs non_social aggregate over PCA participation ratio, dimensionality,
hidden-state speed, and the fraction of partner-visibility-modulated units.
Statistics land in `neural_summary.json` with input hashes in `MANIFEST.json`.

`analyze_neural.py` describes one network at a time. The paper's central neural
result (Fig. 5/6, claim C3) is about structure *shared between the two agents*,
measured with Partial Least Squares Correlation (PLSC). Reproduce it with:

```bash
uv run python scripts/analysis/analyze_shared_neural.py runs/mouse-run-run-0701/paper_rollouts \
  --output-root runs/reports/mouse-run-run-0701/shared_neural
```

For each trained pair this z-scores the two agents' time-aligned hidden states,
takes the SVD of their cross-covariance matrix, and tests each shared dimension
against a temporal-permutation null (one agent's timepoints shuffled). It
reports the number of significant shared dimensions and the top-dimension
correlation per pair and aggregated by task, so social and non_social pairs can
be compared as in the paper. Where PCA is an SVD of one agent's data matrix
(maximum variance within a network), PLSC is an SVD of the cross-covariance
matrix (maximum covariance between networks). Outputs are
`shared_neural_summary.json`, per-pair spectra/scatter figures, an aggregate
comparison, and `MANIFEST.json`; the permutation seed is fixed so results are
reproducible. Still unimplemented from the paper: SVM decoding, the
neural-action-space partner-representation GLM, and null-space perturbation
(C4/C5).

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

# mouse-run-run

`mouse-run-run` is a research harness for multi-agent reinforcement learning
and neural-representation experiments.  It trains independently parameterized
agents, records time-aligned internal activations, fits shared subspaces such as
PLSC, applies causal interventions at readout or recurrent-state boundaries,
and replays baseline and disturbed behavior side by side.

The active runtime is the strict **mrr-v2** architecture. Historical runtime
specs are available in Git history; the working tree contains only executable
v2 definitions and deliberately does not guess or load earlier schemas.

## Setup

Python 3.14 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest
uv run ruff check
```

## Researcher-facing code

The default experiment lives in [`experiments/chase_grid`](experiments/chase_grid):

- `environment.py` — world state, observations, transition dynamics, and replay renderer.
- `reward.py` — reward values and precedence.
- `model.py` — PyTorch policy architecture and named activation sites.
- `experiment.py` — typed composition of environment, policies, PPO, and defaults.
- `analyses.py` / `interventions.py` — experiment-specific analysis recipes.

The reusable machinery is under `mouse_run_run/core`, `training`, `artifacts`,
`analyses`, `figures`, and `replay`.  Changing the chase environment, reward, or model does
not require editing the simulation or PPO loops.

The goal-directed paper replication lives in
[`experiments/zhang_2025_rnn`](experiments/zhang_2025_rnn).  Its formal
confirmatory, behavior-only development, protocol-sensitivity, and tiny CPU
readiness definitions use the same strict study pipeline.

## Zhang 2025 reproduction study

Validate and print the confirmatory workload without creating artifacts or
loading a model:

```bash
uv run mrr study plan experiments.zhang_2025_rnn.study:definition
```

It reports 20 fresh-seed units, 18,000,000 environment steps per unit,
360,000,000 total steps, and 180,000 optimizer steps. Formal execution on the
NVIDIA host is:

```bash
uv run pytest tests/test_v2_triton.py

uv run mrr study run experiments.zhang_2025_rnn.study:definition \
  --output runs/zhang-2025-rnn-confirmatory \
  --device cuda --backend triton --parallelism 10 --stage all
```

Run `--stage train`, `behavior`, `neural`, or `causal` to schedule one stage.
Every later invocation adds `--resume`.  Resume requires an identical typed
definition/config hash and a valid v3 safetensors checkpoint; there is no
legacy or partial-schema fallback.

```bash
uv run mrr study run experiments.zhang_2025_rnn.study:definition \
  --output runs/zhang-2025-rnn-confirmatory \
  --device cuda --backend triton --parallelism 10 \
  --stage neural --resume
```

The readiness-only CPU path is:

```bash
uv run mrr study run experiments.zhang_2025_rnn.study:smoke_definition \
  --output /tmp/mrr-zhang-smoke \
  --device cpu --backend torch --parallelism 1 --stage all
```

Smoke reports deliberately keep scientific gates at `NOT_RUN`.  See the study
[`README`](experiments/zhang_2025_rnn/README.md) for artifacts, analysis, and
replay commands, including the behavior-only development sweep and optional
20,000-update protocol-sensitivity definition.

## Train

Run a tiny CPU smoke experiment:

```bash
uv run mrr train experiments.chase_grid.experiment:definition \
  --run-dir runs/v2/smoke \
  --device cpu \
  --batch-size 4 \
  --set training.updates=2 \
  --set training.horizon=10 \
  --set chaser_model.hidden_size=16 \
  --set explorer_model.hidden_size=16
```

For a real experiment, edit the typed Python definition and run the same
command without smoke overrides.  Small scalar/list values can be overridden
with repeated `--set path.to.field=JSON_VALUE`; structural model or environment
changes belong in their Python files.

Resume from a v2 checkpoint:

```bash
uv run mrr train experiments.chase_grid.experiment:definition \
  --run-dir runs/v2/social-resumed \
  --resume-from runs/v2/social/checkpoints/update_000200.safetensors
```

Agents may use different models, but every agent owns an independent policy
and optimizer.  The current learner is independent PPO.

Each update appends PPO losses and the behaviour of the update's own rollout to
`metrics.jsonl` and TensorBoard.  Behaviour keys match the rollout episode
tables, so a training curve and an evaluation table are read the same way:
`event.<name>` is the mean episode count, `event_fraction.<name>` the mean
fraction of active steps, and `world.distance_mean` / `world.distance_final`
the agent separation.  Field-of-view and collision therefore appear as
`event_fraction.chaser_partner_visible`, `event_fraction.explorer_partner_visible`,
and `event.collision`.

## Collect rollouts

```bash
uv run mrr rollout runs/v2/social/checkpoints/latest.safetensors \
  --experiment experiments.chase_grid.experiment:definition \
  --output runs/v2/rollouts/social-baseline \
  --episodes 128 \
  --horizon 100 \
  --batch-size 32 \
  --seed 7
```

A rollout directory contains:

- `tensors.safetensors`: observations, actions, rewards, logits, values,
  activations, events, and T+1 world state;
- `episodes.parquet`: one analysis-ready row per episode;
- `manifest.json`: strict schema, axis alignment, resolved rollout settings,
  and intervention declarations.

## PLSC and CKA

Fit PLSC with a permutation threshold and rank-matched unique controls:

```bash
uv run mrr analyze plsc runs/v2/rollouts/social-baseline \
  --output runs/v2/analyses/social-plsc \
  --agent-a chaser \
  --agent-b explorer \
  --site hidden \
  --threshold \
  --null-model episode_shuffle \
  --permutations 200 \
  --alpha 0.05 \
  --control-rank 10
```

Use `--rank 10` instead of `--threshold` for a fixed-rank causal study.
Subspaces are fitted in centered/z-scored coordinates and stored with their
inverse transform, so projections return to the original hidden-state units.

```bash
uv run mrr analyze cka runs/v2/rollouts/social-baseline \
  --output runs/v2/analyses/social-cka \
  --site hidden
```

Analysis tables are Parquet; dense spectra, bases, and matrices are
safetensors.

## Readout and recurrent interventions

Readout-only removal changes the current action/value computation while
preserving the original carried state:

```bash
uv run mrr rollout runs/v2/social/checkpoints/latest.safetensors \
  --output runs/v2/rollouts/shared-readout-removed \
  --episodes 128 --horizon 100 --batch-size 32 --seed 7 \
  --subspace runs/v2/analyses/social-plsc \
  --agent chaser --basis shared --operation remove --target readout
```

Changing only `--target recurrent` makes the transformed activation drive the
current readout and become the next recurrent state.  `--basis` also accepts
`top_unique` and `random_unique`; `--operation keep` tests a subspace in
isolation.

Compare paired baseline and disturbed episodes:

```bash
uv run mrr analyze compare \
  runs/v2/rollouts/social-baseline \
  runs/v2/rollouts/shared-readout-removed \
  --output runs/v2/analyses/shared-readout-effect
```

## Replay saved behavior

```bash
uv run mrr replay runs/v2/rollouts/social-baseline \
  --compare runs/v2/rollouts/shared-readout-removed \
  --port 8765
```

Open `http://127.0.0.1:8765`.  The viewer reads the saved evidence directly and
synchronizes environment frames, actions, rewards, events, activation rasters,
and PCA trajectories.  It never generates a new trajectory from a checkpoint.
Repeat `--compare` to synchronize more than two causal conditions.

## Paper figures

A study output directory renders directly into the published Figure 5 layout:

```bash
uv run mrr figures fig5 runs/zhang-2025-rnn-development \
  --output runs/zhang-2025-rnn-development/figures
```

Panels come from the artifacts a study already writes, never from TensorBoard
event files: training curves (c-h) read each unit's `metrics.jsonl`,
random-opponent panels (i-n) read `tables/behavior.parquet`, and the movement
panels (o-r) read the stored behavior rollouts. Group comparisons use the same
Torch permutation null as the PLSC and decoder analyses; with `n` seeds per
group the smallest reachable p-value is fixed by the number of distinct splits,
so small sweeps report `ns` by construction.

Figure 6 needs two derived tables first, both built from artifacts the study
already wrote:

```bash
uv run mrr figures variance runs/zhang-2025-rnn-confirmatory   # panels l-n
uv run mrr figures sweep runs/zhang-2025-rnn-confirmatory      # panels p-r
uv run mrr figures fig6 runs/zhang-2025-rnn-confirmatory \
  --output runs/zhang-2025-rnn-confirmatory/figures
```

`variance` regresses grouped behavioural predictors onto neural activity and
reports what the partner explains beyond the agent itself; `sweep` repeats that
measurement in the action-readout subspace at every training checkpoint. Panel o
is animal data and is drawn as an explicit gap.

`--box-checkpoint` selects the checkpoint the box plots summarize, and
`--early-checkpoint` / `--late-checkpoint` select the two stages compared by the
flow field, polar plot, and angle histogram; each defaults to the extremes of
the checkpoints present in the behavior table. Every panel writes the numbers it
draws to `source_data/*.parquet`, and `manifest.json` records the selections,
the shared flow-arrow scale, and any panel that was skipped.

## Acceleration

The environment backend is explicit:

```bash
uv run mrr train ... --backend torch
uv run mrr train ... --backend triton --device cuda
```

Triton is installed only on supported Linux/x86-64 hosts.  Selecting Triton on
an unsupported device is an error; there is no silent Torch fallback.  Any
environment semantic change must pass the seeded Torch/Triton parity test.

## Artifact policy

- Weights, optimizer/RNG state, rollouts, activations, and dense analysis
  arrays use safetensors.
- Episode and analysis tables use Parquet.
- Operational progress uses terminal updates plus `status.json`; training
  metrics use JSONL.
- Rendered figures are PNG/SVG next to the Parquet source data they were drawn
  from; figures never read TensorBoard event files.
- `.pt`, pickle, `torch.save`, legacy loaders, and schema fallbacks are not
  part of v2.

See [`docs/architecture-v2.md`](docs/architecture-v2.md) and
[`docs/onboarding.md`](docs/onboarding.md) for the contracts and extension
workflow.

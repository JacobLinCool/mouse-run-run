# Zhang 2025 goal-directed RNN study

This package is the canonical code-ready replication of the artificial-agent
claims in Zhang et al. (2025), *Inter-brain neural dynamics in biological and
artificial intelligence systems* (Nature, DOI
`10.1038/s41586-025-09196-4`). It contains no new scientific results.

The primary contract is phenomenon fidelity, not optimizer-trajectory
fidelity. Environment, architecture, controls, endpoints, held-out seeds, and
the analysis checkpoint are fixed. The released RLlib optimization schedule is
available separately as a protocol-sensitivity study.

## Primary scientific contract

- `social` and `non_social` conditions.
- 10×10 grid, physical 7×7 FOV, spawn coordinates `0..8`.
- Independent 256-unit vanilla ReLU RNN chaser and explorer.
- Fast PPO: one full 4,000-timestep recurrent batch per agent/update, recurrent
  L2 `0.3`, and a predeclared 2,000-update analysis checkpoint.
- Confirmatory seeds `100..109`, disjoint from development and historical
  `0..9` runs.
- Random-opponent behavior, paper PLSC, identical-action and non-social vision
  controls, diagnostic grouped linear SVM, and paired causal interventions.
- Nature absolute values are comparison references, never hard tolerances.

The confirmatory plan has 20 units, 160 million environment steps, and 80,000
optimizer steps in total. The optional protocol-sensitivity plan has the same
1.6 billion environment steps and 816 million optimizer steps as the expensive
30-epoch interpretation.

## Development checkpoint selection

Development is behavior-only by construction. It evaluates checkpoints
`500..5000` with identical random-opponent world and action-random streams and
never runs PLSC, decoder, or causal stages:

```bash
uv run mrr study plan \
  experiments.zhang_2025_rnn.study:development_definition

uv run mrr study run \
  experiments.zhang_2025_rnn.study:development_definition \
  --output runs/zhang-2025-rnn-development \
  --device cuda --backend triton --parallelism 6 --stage all
```

`--stage all` resolves to `train, behavior` for this definition. Select one
checkpoint globally from the behavior trajectories, not separately per seed.
`tables/behavior_checkpoint_contrasts.parquet` reports chaser collision, FOV,
and distance advantages plus explorer collision avoidance and distance
advantage; positive values point in the target social-behavior direction.
If the predeclared checkpoint changes, edit `CONFIRMATORY_UPDATE` in `study.py`
before any confirmatory run and rerun the plan/tests. Never select a checkpoint
from confirmatory PLSC or causal outcomes.

## Confirmatory run

Inspect without running:

```bash
uv run mrr study plan experiments.zhang_2025_rnn.study:definition
```

Run stages separately so each boundary can be checked:

```bash
uv run mrr study run experiments.zhang_2025_rnn.study:definition \
  --output runs/zhang-2025-rnn-confirmatory \
  --device cuda --backend triton --parallelism 10 --stage train

uv run mrr study run experiments.zhang_2025_rnn.study:definition \
  --output runs/zhang-2025-rnn-confirmatory \
  --device cuda --backend triton --parallelism 10 --stage behavior --resume

uv run mrr study run experiments.zhang_2025_rnn.study:definition \
  --output runs/zhang-2025-rnn-confirmatory \
  --device cuda --backend triton --parallelism 10 --stage neural --resume

uv run mrr study run experiments.zhang_2025_rnn.study:definition \
  --output runs/zhang-2025-rnn-confirmatory \
  --device cuda --backend triton --parallelism 10 --stage causal --resume
```

The neural stage runs only the frozen 2,000-update checkpoint. It produces 40
neural rollouts and 50 PLSC fits: one primary fit per rollout plus
identical-action fits for the 10 social rollouts. Diagnostic decoders run only
for the 20 primary-visibility rollouts. The protocol study would otherwise
produce 240 neural rollouts and 300 PLSC fits.

## Protocol-sensitivity study

This is not the default and should only be run if optimizer-level sensitivity
becomes a research question:

```bash
uv run mrr study plan \
  experiments.zhang_2025_rnn.study:protocol_definition

uv run mrr study run \
  experiments.zhang_2025_rnn.study:protocol_definition \
  --output runs/zhang-2025-rnn-protocol-sensitivity \
  --device cuda --backend triton --parallelism 10 --stage all
```

It retains 20,000 updates, 30 PPO epochs, 128-timestep minibatches, recurrent
L2 `3.0`, seeds `0..9`, and late checkpoints `15k..20k`.

## Outputs

```text
runs/zhang-2025-rnn-confirmatory/
  study.json, status.json, report.md, gates.json
  tables/
    behavior.parquet
    behavior_checkpoint_summary.parquet
    behavior_checkpoint_contrasts.parquet
    neural.parquet
    plsc_seed_summary.parquet
    decoder.parquet
    causal.parquet
    nature_comparison.parquet
  units/{social,non_social}/seed_XXXX/
    train/checkpoints/*.safetensors
    behavior/update_002000/{matchup}/
      tensors.safetensors, episodes.parquet, manifest.json
    neural/update_002000/{none,partial,full}/
      rollout/{tensors.safetensors,episodes.parquet,manifest.json}
      plsc/{subspace.safetensors,spectrum.parquet,manifest.json}
      plsc_identical_action_excluded/
      decoder/results.parquet
    causal/
      control_basis.safetensors
      {baseline,shared_readout,shifted_random_pc_readout,shared_recurrent}/
```

Raw episodes remain in rollout artifacts. Only neural analyses exclude an
episode when its longest consecutive non-collision stuck run exceeds five
steps. The confirmatory checkpoint is fixed before hidden-state analysis, so
checkpoints are not treated as independent samples.

## Replay causal conditions

```bash
uv run mrr replay \
  runs/zhang-2025-rnn-confirmatory/units/social/seed_0100/causal/baseline \
  --compare runs/zhang-2025-rnn-confirmatory/units/social/seed_0100/causal/shared_readout \
  --compare runs/zhang-2025-rnn-confirmatory/units/social/seed_0100/causal/shifted_random_pc_readout \
  --compare runs/zhang-2025-rnn-confirmatory/units/social/seed_0100/causal/shared_recurrent \
  --experiment experiments.zhang_2025_rnn.replay:social_experiment \
  --port 8765
```

The viewer synchronizes every condition by episode and timestep. Paired causal
rollouts share initial-world seeds and explicit action-uniform random streams.

## Readiness boundary

`smoke_definition` exercises 2 conditions × 2 seeds with tiny budgets. It
always emits `NOT_RUN` for behavior, PLSC, and causal scientific gates. A
CUDA/Triton parity pass remains mandatory on the formal execution host;
CPU-only development reports that test as an explicit skip.

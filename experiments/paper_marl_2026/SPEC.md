# Experiment Spec

## Research Question / Claim

Can a modern PyTorch PPO implementation reproduce the chaser-explorer MARL training pattern from Zhang et al. (2025) while preserving paper-style evaluation and degenerate-episode exclusion rules?

## Experimental Unit

One trained pair of agents for one task condition and random seed.

Unit id:

```text
{task}/seed_{seed:04d}
```

Each unit may have multiple attempts. Failed attempts remain raw evidence; the primary panel uses the latest successful finite attempt per unit.

## Systems / Conditions Compared

- `social`: chaser and explorer receive partial partner visibility.
- `non_social`: partner visibility is disabled.

Both conditions use the same PPO code path, model architecture, seeds, episode length, batch size, checkpoint validation, TensorBoard logging, and analysis pipeline.

## Benchmark / Task Version

- Grid size: `10 x 10`
- Vision radius: `3`
- Episode length: `100` timesteps during training
- Training batch: `40 episodes x 100 timesteps = 4,000 environment steps`
- Updates: `20,000`
- Planned environment steps per pair: `80,000,000`
- Planned valid pairs: `10 social + 10 non_social`

## Primary Metrics

- Valid trained pairs per condition.
- Random-opponent evaluation over `100 episodes x 100 timesteps`, per condition (`social` vs `non_social`), against the latest successful attempt's checkpoint per unit, with a shared evaluation seed so every checkpoint faces the same standardized random opponent.
- Collisions per episode.
- Chaser return and explorer return.
- Partner-in-vision fraction.
- New fields explored.
- Final and average distance.

## Secondary / Diagnostic Metrics

- PPO policy loss, value loss, entropy, approximate KL.
- Gradient norm.
- Maximum absolute parameter value.
- RNN neural action subspace norms.
- Environment steps per second.
- Degenerate episode counts in paper analysis rollouts.
- Failed attempt count and failure reason.

## Planned Panels / Slices

- `primary_valid_runs_v1`: completed runs with healthy finite checkpoint.
- `paper_random_opponent_valid_v1`: successful standardized random-opponent evaluations.
- `paper_neural_behavior_non_degenerate_v1`: `25 x 500` analysis rollouts excluding paper-style degenerate episodes.

## Failure, Timeout, and Exclusion Rules

- A training attempt fails if any loss, logits-derived tensor, gradient norm, model parameter, metric, or checkpoint tensor becomes NaN or Inf.
- A completed attempt is invalid unless `status.json` says `completed` and `checkpoints/latest.safetensors` passes finite checkpoint validation.
- Genuinely failed attempts (NaN/Inf, crash, invalid checkpoint, timeout) are preserved and retried up to the configured `max_attempts`.
- A worker is killed and its attempt marked `timeout` after `attempt_timeout_hours` wall-clock hours (primary: 24).
- Interrupted attempts (runner shutdown, host reboot) do not count toward `max_attempts`. The next invocation starts a new attempt that resumes from the newest healthy checkpoint with training state; `max_total_attempts` (default 10) bounds total attempt directories per unit.
- NaN checkpoints are training failures, not paper-style exclusions.
- Paper-style degenerate episode exclusion applies only to analysis rollouts. Interpretation of the paper rule: an episode is degenerate when the longest consecutive run of stuck steps (agent issued a valid action but stayed in the same tile for a non-social reason — wall bump or explicit stationarity; collision-blocked steps are excluded because collision is the rewarded social outcome) exceeds `1%` of episode length, per agent or jointly.

## Trial Plan / Seeds / Budget

- Seeds: `0..9`
- Tasks: `social`, `non_social`
- Attempts: up to `2` per unit by default
- Primary result requires `10/10` valid pairs per task.

## Execution Code / Run Command

Smoke:

```bash
uv run python scripts/run/launch_gate.py --config experiments/paper_marl_2026/configs/smoke.json --allow-smoke
uv run python scripts/run_local_experiment.py --config experiments/paper_marl_2026/configs/smoke.json
```

Primary (the gate requires a Triton equivalence record produced from the
current commit with a clean working tree):

```bash
uv run python scripts/verify_triton_env.py --output runs/mouse-run-run-0701/triton_equivalence.json
uv run python scripts/run/launch_gate.py \
  --config experiments/paper_marl_2026/configs/primary.json \
  --triton-equivalence runs/mouse-run-run-0701/triton_equivalence.json
uv run python scripts/run_local_experiment.py --config experiments/paper_marl_2026/configs/primary.json
```

## Raw Evidence to Collect

- Experiment manifest: `runs/{experiment}/manifest.json`
- Runner status: `runs/{experiment}/run_status.json`
- Attempt raw records: `runs/{experiment}/raw_records.jsonl`
- Per-attempt logs, status, metrics, TensorBoard events, config, and checkpoints.
- Paper random-opponent evaluation JSONL.
- Paper `25 x 500` rollout safetensors plus rollout records.
- Triton equivalence gate record.

## Canonical Tables to Produce

```bash
uv run python scripts/transform/build_tables.py runs/{experiment}
```

Outputs:

- `runs/tables/{experiment}/runs.jsonl`
- `runs/tables/{experiment}/evaluations.jsonl`
- `runs/tables/{experiment}/rollouts.jsonl`
- `runs/tables/{experiment}/exclusions.jsonl`
- `runs/tables/{experiment}/panels.json`

## Planned Analysis Outputs

```bash
uv run python scripts/analysis/summarize_experiment.py runs/tables/{experiment}
```

Outputs:

- `runs/reports/{experiment}/summary.json`
- `runs/reports/{experiment}/report.md`
- `runs/reports/{experiment}/MANIFEST.json`

## Known Limitations

This is a modern implementation, not the original RLlib code. The primary preset uses the paper-text recurrent L2 coefficient while retaining the Triton env-only acceleration path after equivalence verification.

## Spec Revision Log

- 2026-07-02: Initial formal spec. Fixed primary preset to `paper_text`, retained Triton env-only acceleration, and defined NaN checkpoints as failed attempts rather than exclusions.
- 2026-07-02 (rev 2, pre-launch review): degenerate rule reinterpreted as longest consecutive stuck run excluding collision-blocked steps; interrupted attempts no longer consume `max_attempts` and resume from checkpoint (optimizer/RNG state now serialized); per-agent gradient clipping (was joint); approach/escape events gated on new-field per official event precedence; paper evaluation/rollout selection fixed to the latest successful attempt per unit with a shared seed; evaluation summary grouped by task; 24h attempt timeout; launch gate binds the Triton equivalence record to the current git sha.

# Experiment Spec

## Research Question / Claim

For an already-trained social RNN with clear pursuit behavior, are the top 10
chaser-side PLSC shared dimensions necessary or sufficient for the chaser's
action selection?  Do shared-dimension interventions alter pursuit and spatial
exploration more than rank-matched directions in the unique complement?

This is an exploratory, single-checkpoint causal case study.  It does not train
or fine-tune a model and does not test generalization across independently
trained checkpoints.

## Background and Checkpoint Selection

The paper reports that removing the top 10 shared dimensions before the action
readout reduces social behavior.  It also compares those directions with 25
random principal components chosen to remove comparable variance
([Zhang et al., 2025](https://doi.org/10.1038/s41586-025-09196-4), Fig. 6s–u
and Extended Data Fig. 13g).

The selected checkpoint is:

```text
runs/mouse-run-run-1/social/seed_0004/attempt_01/checkpoints/latest.safetensors
```

Selection was made before this intervention's results were collected.  Among
the existing social RNN checkpoints, seed 4 has the clearest pursuit behavior:
the existing standardized random-explorer evaluation reports 20.57 collisions
per episode and average distance 2.16; the prior self-play C5 evaluation reports
25.47 collisions, 94.87% partner visibility, and average distance 1.83.

The PLSC fit uses the checkpoint's existing paper rollout:

```text
runs/mouse-run-run-1/paper_rollouts/mouse-run-run-1__social__seed_0004__attempt_01__checkpoints__latest__759bf4944f.safetensors
```

It contains 23 non-degenerate episodes and 11,500 fitted timesteps.

## Experimental Unit

One fixed checkpoint, one evaluation seed, and one intervention condition.
Episode-level rows are paired by evaluation seed and episode index because all
conditions start from identical initial positions.

Evaluation seeds quantify rollout variability for this checkpoint.  They are
not independent trained-model replicates.

## Subspace Definition and Intervention

Let `h` be the 256-dimensional chaser RNN activation.  The fit-rollout chaser
mean and per-unit standard deviation define:

```text
z = (h - mean) / scale
```

PLSC is fitted from the cross-covariance of the centered/z-scored, time-aligned
chaser and explorer hidden states.  The chaser-side top-10 orthonormal basis is
`S`.  At evaluation time, transformed activity is returned to the original
hidden coordinates:

```text
h_intervened = mean + scale * z_intervened
```

Only the chaser action head receives `h_intervened`.  The unmodified recurrent
state is threaded into the next timestep.  This is a readout-level intervention
and does not perturb recurrent dynamics.

## Conditions Compared

1. `unperturbed`: original hidden state and logits.
2. `shared_removed`: `z - z S S^T`; tests necessity of shared top-10.
3. `unique_removed`: `z S S^T`; removes the entire 246-dimensional unique
   complement and tests whether shared top-10 alone are sufficient.
4. `top_unique_removed`: removes the ten highest-variance directions in the
   unique complement.  The basis is orthonormal and exactly orthogonal to `S`.
5. `random_unique_removed`: removes a seeded random 10-dimensional basis in
   the unique complement.  The basis is orthonormal and exactly orthogonal to
   `S`.

Conditions 4 and 5 are rank-matched specificity controls.  Condition 3 is not
rank matched and must only be interpreted as a sufficiency diagnostic.

## Primary Metrics

- Collisions per episode.
- Chaser partner-in-vision fraction.
- Average chaser–explorer distance.
- Chaser new fields explored per episode.
- Chaser approach fraction.

## Secondary / Diagnostic Metrics

- Chaser return and explorer return.
- Explorer visibility, exploration, and escape fraction.
- Final distance.
- KL divergence between original and intervened chaser action distributions.
- Fraction of timesteps whose argmax action changes.
- Relative hidden-state change.
- Fraction of intervened hidden units below zero.
- Degenerate episode fraction and longest stuck-run diagnostics.
- Chaser spatial occupancy heatmap and action frequencies.

## Planned Panel

`all_conditions_v1`: all five conditions and all configured evaluation seeds,
paired by `(evaluation_seed, episode_index)`.

No outcome-dependent exclusions are allowed.  Degenerate episodes remain in
the canonical table and are reported as a diagnostic rather than removed.

## Failure and Validity Rules

- The basis fit fails on non-finite data, constant hidden units, rank
  deficiency, non-orthonormal bases, or overlap between shared and unique
  control bases.
- A condition record fails on execution error or a checkpoint/basis dimension
  mismatch.
- The canonical transformation fails unless every configured
  `(evaluation_seed, condition)` record is present and successful.
- The transformation also fails if paired conditions do not have identical
  initial chaser and explorer positions.
- Failed or partial raw batches remain on disk and are never overwritten.

## Trial Plan / Seeds / Budget

Smoke config:

- 2 evaluation seeds.
- 16 episodes per seed and condition.
- 5 conditions, 100 timesteps.

Primary exploratory config:

- 20 evaluation seeds (`0..19`).
- 256 episodes per seed and condition.
- 5 conditions, 100 timesteps.
- Total: 25,600 evaluated episodes and 2,560,000 environment transitions.

The basis and random-unique control use fixed seed `20260726`.

## Execution Code / Commands

Smoke:

```bash
uv run python scripts/run/run_subspace_intervention.py \
  --config experiments/rnn_subspace_intervention_seed4_2026/configs/smoke.json \
  --batch-id smoke_001
```

Primary:

```bash
uv run python scripts/run/run_subspace_intervention.py \
  --config experiments/rnn_subspace_intervention_seed4_2026/configs/primary.json \
  --batch-id primary_001
```

The batch id is immutable.  Reusing an existing batch id is an error.

## Raw Evidence

```text
runs/raw/rnn_subspace_intervention_seed4_2026/{batch_id}/
  start.json
  fitted_subspaces.json
  fitted_subspaces.safetensors
  records/eval_seed_*__{condition}.json
  completion.json
```

Each condition record contains every episode's metrics and initial positions,
plus aggregate spatial occupancy counts.  Partial and failed batches are
preserved under their original batch ids.

## Canonical Tables

```bash
uv run python scripts/transform/build_subspace_intervention_tables.py \
  runs/raw/rnn_subspace_intervention_seed4_2026/primary_001 \
  --output-root runs/tables/rnn_subspace_intervention_seed4_2026
```

Outputs:

- `episodes.jsonl`: one row per evaluation seed, episode, and condition.
- `condition_runs.jsonl`: one row per evaluation seed and condition.
- `panels.json`: ordered panel and pairing definition.
- `MANIFEST.json`: source/output hashes, row counts, and provenance.

## Planned Analysis Outputs

```bash
uv run python scripts/analysis/analyze_subspace_intervention.py \
  runs/tables/rnn_subspace_intervention_seed4_2026 \
  --output-root runs/reports/rnn_subspace_intervention_seed4_2026
```

Outputs:

- `summary.json`
- `report.md`
- `CLAIM_EVIDENCE.md`
- `figures/behavior_interventions.png`
- `figures/chaser_occupancy.png`
- `MANIFEST.json`

Condition means and paired deltas use evaluation-seed means.  Reported 95%
intervals are paired bootstrap intervals over evaluation seeds.

## Known Limitations

- One selected model cannot establish checkpoint-population generality.
- The top-10 definition follows the paper's perturbation analysis, not the
  much larger number of nominally significant dimensions in this replication.
- `unique_removed` removes 246 dimensions and can create a strong
  out-of-distribution intervention; it tests sufficiency, not specificity.
- PLSC is estimated from trajectories generated by the same checkpoint.  This
  is intentional for a within-model intervention but does not establish that
  the basis transfers across policies.
- Readout-only intervention isolates immediate action selection.  Perturbing
  the recurrent state itself would answer a different question.

## Spec Revision Log

- 2026-07-26: Initial pre-data specification.  Selected seed 4 from existing
  evaluation evidence; fixed the five-condition panel, centered/z-scored
  intervention coordinates, readout-only target, metrics, seeds, and
  append-only evidence layout before running the new intervention.
- 2026-07-26 (post-smoke diagnostic revision): Added explicit reporting of the
  fitted-rollout variance captured by each intervention basis after inspecting
  the smoke behavior.  The raw fitted-subspace artifact already recorded these
  values.  No condition, seed, episode budget, intervention, primary metric, or
  inclusion rule changed; the variance comparison is labeled as an exploratory
  interpretation diagnostic because rank-matched bases were not
  variance-matched.

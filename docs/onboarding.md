# Onboarding

## Mental model

One typed Python `ExperimentDefinition` composes a stable set of named agents,
one batched environment, an independent policy factory per agent, PPO defaults,
and an optional replay renderer.  The shared `SimulationEngine` is the only
agent/environment stepping loop.

At every timestep each policy performs two distinct operations:

1. `advance(observation, state)` produces named activations and a candidate
   next state.
2. `readout(features)` produces action logits and a value estimate.

That boundary makes causal semantics precise.  A `readout` intervention changes
only operation 2.  A `recurrent` intervention changes the current readout and
the state carried into the next `advance` call.

## Read in this order

1. `experiments/chase_grid/experiment.py` — the complete composition.
2. `experiments/chase_grid/environment.py` and `reward.py` — task semantics.
3. `experiments/chase_grid/model.py` — architecture and activation sites.
4. `mouse_run_run/core/simulation.py` — the shared data flow.
5. `mouse_run_run/training/ppo.py` — the one implemented learner.
6. `mouse_run_run/artifacts/rollout.py` — exact tensor and table alignment.
7. `mouse_run_run/analyses/plsc.py` and `interventions.py` — fit, project,
   inverse-transform, and intervene.
8. `experiments/zhang_2025_rnn/study.py` — goal-directed development,
   confirmatory, and protocol-sensitivity settings.
9. `mouse_run_run/studies/runner.py` — strict stage scheduling and aggregation.
10. `mouse_run_run/analyses/paper_plsc.py` — Zhang-specific dual-statistic nulls.

## Common changes

### Change the environment

Edit `ChaseGridConfig` or `ChaseGridEnvironment` in `environment.py`.  Keep all
agent-indexed outputs keyed by the experiment's `AgentId` tuple and keep world
state at T+1 alignment.  Update both the renderer and the Triton parity test
when recorded world semantics change.

### Change rewards

Edit only `reward.py`.  `GridReward.compute` receives mutually exclusive event
masks and returns one reward tensor per agent.

### Change the model

Edit `model.py`.  A policy must implement `initial_state`, `advance`, `readout`,
`carry_state`, and `evaluate_sequence`.  Declare every recorded/intervenable
tensor in `activation_sites`.  The included RNN and CNN→RNN models demonstrate
both flat and convolutional encoders without trainer changes.

### Add an analysis

Read a strict `RolloutArtifact`, produce Parquet rows for tabular results, and
put dense matrices in safetensors.  Analysis must not rerun the policy or
silently exclude episodes.

### Add an intervention

Implement the small `Intervention` protocol or construct a
`SubspaceIntervention`.  Specify its agents, activation site, and either the
`readout` or `recurrent` target explicitly.  The simulation engine validates
the declaration against each policy before stepping.

## Required checks

```bash
uv run python -m compileall -q mouse_run_run experiments
uv run pytest
uv run ruff check
```

On a CUDA host, the Triton parity test is mandatory before using
`--backend triton` for research runs.

Run the complete CPU readiness fixture with:

```bash
uv run mrr study run experiments.zhang_2025_rnn.study:smoke_definition \
  --output /tmp/mrr-zhang-smoke --device cpu --backend torch --stage all
```

Its numerical results are not evidence; behavior, PLSC, and causal gates remain
`NOT_RUN` by construction.

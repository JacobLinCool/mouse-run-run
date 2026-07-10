# Experiment Spec: Official-Dynamics Fast-Runner Reproduction

## Research Question / Claim

Does the paper's behavioral and neural-representation pattern reproduce when
the existing vectorized PyTorch runner uses the learning dynamics actually
inherited by the authors' Ray RLlib 2.2 training code?

This is a new experiment family. Its results must not be pooled with or
silently substituted for `paper_marl_2026`, whose completed checkpoints used
the earlier full-batch fast learner.

## Upstream Contract

- Official MARL repository:
  `https://github.com/hongw-lab/marl_environment_chase`
- Audited upstream commit: `aedb83715242a2e1bf2f68ed8aa5baf86d3a3fd0`
- Ray version pinned by upstream: `ray[all]==2.2.0`
- PyTorch version pinned by upstream: `torch==1.12.1+cu116`
- Ray PPO source tag audited: `ray-2.2.0`

The port targets algorithmic equivalence, not bit-identical trajectories.
Rollout collection remains vectorized and the modern runtime uses a newer
PyTorch/CUDA stack, so random-number ordering and floating-point reductions
differ from the distributed Ray execution.

### Implementation Source Registry

Source IDs in execution-code comments resolve here. Paper-derived, released-
code, inherited-library, and local engineering values are deliberately kept
separate.

- `[PAPER-METHODS]`: Zhang et al. (2025), *Inter-brain neural dynamics in
  biological and artificial intelligence systems*, Methods (artificial-agent
  training) and Supplementary Table 3; [DOI
  10.1038/s41586-025-09196-4](https://doi.org/10.1038/s41586-025-09196-4).
- `[OFFICIAL-TRAIN]`: released argument defaults, policy map, environment/model
  config, and stopping rule in
  [`MultiAgent_Train_Final.py`](https://github.com/hongw-lab/marl_environment_chase/blob/aedb83715242a2e1bf2f68ed8aa5baf86d3a3fd0/MultiAgent_Train_Final.py#L15-L145).
- `[OFFICIAL-MODEL]`: vanilla ReLU RNN, independent action/value heads,
  PyTorch-default initialization, and unsquared recurrent-weight norm in
  [`simple_rnn_v2_3_2.py`](https://github.com/hongw-lab/marl_environment_chase/blob/aedb83715242a2e1bf2f68ed8aa5baf86d3a3fd0/simple_rnn_v2_3_2.py#L20-L149).
- `[OFFICIAL-ARENAS]`: grid/vision/horizon, spawn range, action and movement
  order, observations, rewards, and event precedence in the released
  [`social`](https://github.com/hongw-lab/marl_environment_chase/blob/aedb83715242a2e1bf2f68ed8aa5baf86d3a3fd0/MultiAgentArena_v1d_5.py#L10-L269)
  and
  [`non-social`](https://github.com/hongw-lab/marl_environment_chase/blob/aedb83715242a2e1bf2f68ed8aa5baf86d3a3fd0/MultiAgentArena_v1d_11.py#L9-L294)
  arenas.
- `[RAY-PPO-CONFIG]`: Ray 2.2.0 PPO defaults in
  [`PPOConfig`](https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/algorithms/ppo/ppo.py#L82-L110),
  with inherited `gamma=0.99` in
  [`AlgorithmConfig`](https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/algorithms/algorithm_config.py#L246-L252).
- `[RAY-PPO-LOSS]`: recurrent masking, clipped surrogate, categorical KL,
  squared value-error clipping, entropy, and loss composition in
  [`PPOTorchPolicy.loss`](https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/algorithms/ppo/ppo_torch_policy.py#L69-L175).
- `[RAY-SGD-SLICING]`: per-policy advantage standardization and epoch loop in
  [`rllib.utils.sgd`](https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/utils/sgd.py#L14-L135),
  plus recurrent boundary slicing in
  [`SampleBatch._get_slice_indices`](https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/policy/sample_batch.py#L1035-L1078).
- `[RAY-KL-ADAPT]`: per-policy KL state and the `2.0x/0.5x` target thresholds
  in
  [`KLCoeffMixin`](https://github.com/ray-project/ray/blob/ray-2.2.0/rllib/policy/torch_mixins.py#L73-L101).
- `[LOCAL-CALIBRATION]`: concurrency, timeout, finite checks, TF32, Triton,
  checkpoint cadence, and capacity projections are local execution choices,
  not paper constants; see [`RUNS.md`](RUNS.md) and
  [`cuda_calibration_20260710.json`](evidence/cuda_calibration_20260710.json).

## Experimental Unit

One independently initialized chaser/explorer policy pair for one task
condition and seed:

```text
{task}/seed_{seed:04d}
```

The agents have separate parameters, Adam optimizers, optimizer moments, and
adaptive KL coefficients.

## Systems / Conditions Compared

- `social`: partner input is visible within the 7 x 7 field of view.
- `non_social`: the partner input channel is always hidden.
- `official_code_l2_v1`: recurrent L2 coefficient `3.0`, matching the
  released training code and demo checkpoint metadata.
- `methods_text_l2_v1`: recurrent L2 coefficient `0.3`, matching the Methods
  text; every other learner setting is identical to `official_code_l2_v1`.

The L2 panels are separate sensitivity panels because the source code and
paper text disagree by a factor of ten.

## Environment / Task Contract

- Grid: `10 x 10` (`[PAPER-METHODS]`, `[OFFICIAL-TRAIN]`,
  `[OFFICIAL-ARENAS]`)
- Episode length: `100` (`[PAPER-METHODS]`, `[OFFICIAL-TRAIN]`,
  `[OFFICIAL-ARENAS]`)
- Train batch: `40 episodes = 4,000 environment steps`
  (`[PAPER-METHODS]`, `[RAY-PPO-CONFIG]`)
- Initialization positions: coordinates `0..8`, reproducing the official
  `np.random.randint(height - 1)` behavior (`[OFFICIAL-ARENAS]`)
- Chaser moves first; explorer moves second (`[OFFICIAL-ARENAS]`; released
  `agent2=chaser`, `agent1=explorer`)
- Reward and event precedence match the released social/non-social arenas
  (`[PAPER-METHODS]`, `[OFFICIAL-ARENAS]`)
- Physical partner-in-FOV remains measurable in the non-social condition even
  though it is not supplied to the policy

## Learner Contract

The `official_code` preset resolves to:

| Setting | Value | Source |
| --- | ---: | --- |
| learning rate | `5e-5` | `[RAY-PPO-CONFIG]` |
| gamma | `0.99` | `[RAY-PPO-CONFIG]` |
| GAE lambda | `1.0` | `[RAY-PPO-CONFIG]` |
| PPO epochs | `30` | `[RAY-PPO-CONFIG]` |
| SGD minibatch | `128` timesteps | `[RAY-PPO-CONFIG]` |
| recurrent sequence length | `20` | `[OFFICIAL-TRAIN]` |
| policy clip | `0.3` | `[OFFICIAL-TRAIN]`, `[RAY-PPO-CONFIG]` |
| value-loss coefficient | `1.0` | `[RAY-PPO-CONFIG]` |
| squared value-loss clip | `10.0` | `[RAY-PPO-CONFIG]`, `[RAY-PPO-LOSS]` |
| entropy coefficient | `0.0` | `[RAY-PPO-CONFIG]` |
| initial KL coefficient | `0.2` per policy | `[OFFICIAL-TRAIN]`, `[RAY-PPO-CONFIG]` |
| KL target | `0.01` | `[RAY-PPO-CONFIG]` |
| gradient clipping | disabled | `[OFFICIAL-TRAIN]`, `[RAY-PPO-CONFIG]` |
| RNN/head initialization | PyTorch defaults | `[OFFICIAL-MODEL]` |
| recurrent L2 | `3.0` unsquared norm | `[OFFICIAL-TRAIN]`, `[OFFICIAL-MODEL]` |
| Adam | separate optimizer per policy | `[OFFICIAL-TRAIN]`, RLlib policy ownership |

Ray selects `simple_optimizer` for this multi-agent `PPOTorchPolicyV2`
configuration, and its collector supplies an unpadded batch with 200 sequence
lengths of 20. Ray 2.2's recurrent `SampleBatch._get_slice_indices` therefore
produces 33 nominal 128-step minibatches per policy and epoch. Each contains six
complete 20-step sequences plus the first eight steps of a seventh; the seventh
sequence is reintroduced in full at the start of the next minibatch. The final
32 timesteps of the 4,000-step policy batch are not selected. The port preserves
this released-code behavior, including the eight-step overlap. Advantage
standardization, exact categorical KL, adaptive KL updates, padded-step
masking, and the official custom recurrent-weight norm are also reproduced.

## Trial Plan / Seeds / Budget

- Seeds: `0..9`
- Tasks: `social`, `non_social`
- Updates: `20,000`
- Planned pairs per L2 panel: `20`
- Planned environment steps per panel: `1.6 billion`
- Run `smoke.json` first.
- Run `calibration_cuda.json` before a full panel and record measured update
  time, peak memory, concurrency, projected wall time, and code/config hashes.
- Launch `official_code_l2_v1` first. Launch `methods_text_l2_v1` only as the
  pre-declared L2 ambiguity panel, never as a replacement chosen after seeing
  the first panel's outcome.

## Primary Metrics

The behavioral and neural metrics, standardized random-opponent evaluation,
and paper-style rollout inclusion rules remain those defined in
`experiments/paper_marl_2026/SPEC.md`. The primary comparison is between each
new panel and the original paper's source-data values; the previous fast-runner
panel is a labeled implementation baseline.

Additional learner audit metrics:

- exact categorical KL per update
- current chaser and explorer KL coefficients
- policy loss, clipped value loss, entropy, and per-policy gradient norm
- minibatch count and resolved learner configuration in manifests/checkpoints
- environment steps per second, update wall time, and peak device memory during
  calibration

## Failure / Exclusion Rules

- Non-finite losses, gradients, parameters, metrics, or checkpoint tensors are
  failed attempts.
- Failed, timed-out, and interrupted attempts remain append-only raw evidence.
- Resume requires equality of all learner-defining configuration fields and
  restores both optimizer states, both adaptive KL coefficients, action RNG,
  minibatch-order RNG, and CUDA RNG.
- Behavioral degeneracy exclusions apply only during analysis rollouts and use
  the existing declared rule; they never erase training failures.

## Execution Commands

Smoke:

```bash
uv run python scripts/run_local_experiment.py \
  --config experiments/paper_marl_official_dynamics_2026/configs/smoke.json
```

CUDA calibration:

```bash
uv run python scripts/verify_triton_env.py \
  --output runs/mouse-run-run-official-calibration/triton_equivalence.json
uv run python scripts/run/launch_gate.py \
  --config experiments/paper_marl_official_dynamics_2026/configs/calibration_cuda.json \
  --triton-equivalence runs/mouse-run-run-official-calibration/triton_equivalence.json \
  --allow-smoke
uv run python scripts/run_local_experiment.py \
  --config experiments/paper_marl_official_dynamics_2026/configs/calibration_cuda.json
```

Primary code-L2 panel:

```bash
uv run python scripts/verify_triton_env.py \
  --output runs/mouse-run-run-official-code-l2/triton_equivalence.json
uv run python scripts/run/launch_gate.py \
  --config experiments/paper_marl_official_dynamics_2026/configs/official_code_l2.json \
  --triton-equivalence runs/mouse-run-run-official-code-l2/triton_equivalence.json
uv run python scripts/run_local_experiment.py \
  --config experiments/paper_marl_official_dynamics_2026/configs/official_code_l2.json
```

Methods-text L2 sensitivity panel:

```bash
uv run python scripts/verify_triton_env.py \
  --output runs/mouse-run-run-official-methods-l2/triton_equivalence.json
uv run python scripts/run/launch_gate.py \
  --config experiments/paper_marl_official_dynamics_2026/configs/methods_text_l2.json \
  --triton-equivalence runs/mouse-run-run-official-methods-l2/triton_equivalence.json
uv run python scripts/run_local_experiment.py \
  --config experiments/paper_marl_official_dynamics_2026/configs/methods_text_l2.json
```

## Raw Evidence / Canonical Outputs

Each experiment writes the existing append-only runner manifest,
`raw_records.jsonl`, per-attempt configs/logs/status/metrics/TensorBoard events,
checkpoints, and final run status under `runs/{experiment}`. Existing transform,
evaluation, rollout, and analysis commands generate canonical tables and
reports without hand-editing.

## Known Limitations

- The port preserves the fast vectorized environment and fused two-policy
  rollout equations; it does not recreate Ray's five worker processes or its
  exact random-number interleaving.
- The runtime uses the repository's current PyTorch version. CUDA matmul TF32
  is disabled and cuDNN TF32 remains enabled to match PyTorch 1.12 defaults,
  but backend kernels are newer.
- The official code does not set a training seed. This experiment adds explicit
  seeds for reproducibility, so seed identities do not correspond to an
  official run.

## Spec Revision Log

- 2026-07-10 (v1, before new data collection): created a separate experiment
  family for the RLlib-2.2-equivalent learner; fixed environment spawn range,
  non-social physical-FOV measurement, optimizer ownership, recurrent slicing,
  exact/adaptive KL, initialization, resume state, and the two pre-declared L2
  panels.
- 2026-07-10 (v1.1, documentation-only, after CUDA calibration): added stable
  paper, official-code, Ray 2.2.0, and local-calibration source IDs and
  permalinks. No configuration, metric, panel, inclusion rule, or execution
  behavior changed.

# Zhang et al. 2025 MARL Replication Notes

## Analysis Plan

- Paper set: Zhang et al. (2025), "Inter-brain neural dynamics in biological and artificial intelligence systems", Nature 645, 991-1001, DOI 10.1038/s41586-025-09196-4.
- Local source: `/Users/jacoblincool/Downloads/s41586-025-09196-4.pdf`.
- Supplementary source: official Nature supplementary PDF, `41586_2025_9196_MOESM1_ESM.pdf`.
- Official code source: `https://github.com/hongw-lab/marl_environment_chase`, observed HEAD `aedb83715242a2e1bf2f68ed8aa5baf86d3a3fd0`.
- Goal: replication assessment and implementation relevance for the MARL/RNN/artificial-agent part of the paper.
- Research question: what exactly must be implemented to reproduce the chaser-explorer MARL experiment and the neural-action-subspace/shared-subspace analyses?
- Required depth: experiment-focused read of main Fig. 5, Fig. 6, Methods, Supplementary Table 3/4, Supplementary Note 13, and official MARL code structure.
- Deliverable: reusable implementation note for rebuilding the experiment in this repo.

## Paper Identity

The paper studies inter-brain/inter-agent shared neural dynamics in both biological mouse dyads and artificial multi-agent reinforcement learning systems. The part relevant to this repo is the chaser-explorer MARL experiment in Fig. 5 and Fig. 6.

The artificial-agent claim is not simply "RNNs can chase." The stronger claim is:

1. social reward structure makes two independently parameterized recurrent agents develop social interaction behavior;
2. trained social agents develop shared neural dimensions across agents, measured with PLSC;
3. these shared dimensions are behaviorally meaningful and causally relevant, because removing the top shared dimensions from the chaser reduces social behavior.

## Core Claims For The MARL Part

### C1: Social rewards produce social behavior

- Claim: chaser/explorer agents trained in a social environment learn role-specific social behavior; non-social controls do not.
- Evidence in paper: Fig. 5 shows chaser reward, explorer reward, collisions, time with partner in vision, new fields explored, and evaluation against a standardized random agent over training.
- Implementation implication: random-agent evaluation is required. Training-batch reward alone is insufficient because each agent's opponent changes as the co-trained partner learns.

### C2: RNN activity encodes social events and partner behavior

- Claim: trained social agents' RNN activity decodes collision and partner approach/escape above chance; non-social agents do not.
- Evidence in paper: Fig. 6b-e SVM balanced accuracy.
- Implementation implication: rollouts must save per-timestep RNN hidden states, positions, actions, collision events, approach events, and escape events.

### C3: Shared neural dimensions emerge in social agents

- Claim: social agents have more significant shared dimensions and higher top-dimension correlation than non-social controls.
- Evidence in paper: Fig. 6f-h and Extended Data Fig. 13a-d.
- Controls:
  - non-social agents given the same partial partner vision;
  - non-social agents given full partner vision;
  - removing moments where both agents take identical actions.
- Implementation implication: this cannot be tested by training only one social environment. We need social, non-social-no-partner-input, non-social-partial-vision, and non-social-full-vision evaluation variants.

### C4: Partner representation in the neural action space predicts chaser performance

- Claim: as social chasers train, partner representation in the neural action space increases and correlates with collision frequency.
- Evidence in paper: Fig. 6p-q; non-social chasers do not show the same relationship in Fig. 6r.
- Implementation implication: compute neural action space as `W_action.T @ W_action @ h_t`, then quantify non-redundant variance explained by partner behavior.

### C5: Perturbing top shared dimensions impairs social actions

- Claim: projecting the chaser RNN activation away from the top 10 shared dimensions reduces collisions and time with partner in vision, and increases distance.
- Evidence in paper: Fig. 6s-u.
- Control: remove a comparable amount of random non-overlapping variance using random principal components.
- Implementation implication: policy inference must support online hidden-state intervention before action logits are computed.

## Chaser-Explorer Environment Specification

### World

- Grid world: `10 x 10`.
- Episode length: `100` timesteps.
- Agents:
  - explorer: tries to explore new tiles and avoid collision;
  - chaser: tries to collide with/block the explorer.
- Actions: four discrete moves.
  - Official code mapping: `0=up`, `1=right`, `2=down`, `3=left`.
- Field of vision: `7 x 7`.
- Initial positions: randomly initialized at episode reset, with no overlap.
- Observation size: `200`.
  - first 100 entries: one-hot self location;
  - second 100 entries: one-hot opponent location if opponent is in vision, otherwise all zero.

Important official-code mapping:

- Paper names: `chaser`, `explorer`.
- Official repo names:
  - `agent1` = explorer;
  - `agent2` = chaser.
- In `MultiAgentArena_v1d_5.py`, `agent2` moves first, then `agent1`.

### Events

- `collision`: an agent attempts to move into the other agent's current location; movement is blocked and a collision event is emitted.
- `new field`: an agent enters a tile it has not previously visited in the episode.
- `approach`: for the chaser, action reduces distance to the explorer's last position.
- `escape`: for the explorer, action increases distance from the chaser's last position.
  - `escape_far`: updated distance `>= 5` units (official code uses `>=`, not strict);
  - `escape_near`: updated distance in `[3, 5)`;
  - `escape_close`: updated distance `< 3`.

Official event precedence (from `_move` early returns): per agent, at most one
event per step, with priority `collision > own new_field > escape/approach`.
A step that enters a new field never also emits approach/escape. The modern
port enforces the same gating.

### Rewards

From Supplementary Table 3.

| Task | Event | Chaser | Explorer |
|---|---:|---:|---:|
| Social | Collision | +1.0 | -1.0 |
| Social | New fields explored | +0.1 | +1.0 |
| Social | Otherwise | -0.1 | -0.5 |
| Non-social | Collision | -0.1 | -0.5 |
| Non-social | New fields explored | +0.1 | +1.0 |
| Non-social | Otherwise | -0.1 | -0.5 |

Notes:

- In the non-social task, collision reward equals the baseline "otherwise" reward. This removes the social collision incentive.
- In the non-social task, the partner location is not provided as input.
- Official code uses the equivalent mapping with `agent1=explorer`, `agent2=chaser`:
  - social: `agent1` collision `-1.0`, new field `+1.0`, otherwise `-0.5`; `agent2` collision `+1.0`, new field `+0.1`, otherwise `-0.1`;
  - non-social: `agent1` collision `-0.5`, new field `+1.0`, otherwise `-0.5`; `agent2` collision `-0.1`, new field `+0.1`, otherwise `-0.1`.

Supplementary Table 4 gives an alternative social reward set for Extended Data Fig. 13f:

| Event | Chaser | Explorer |
|---|---:|---:|
| Collision | +1.0 | -1.0 |
| New fields explored | +0.1 | +0.1 |
| Otherwise | -0.1 | -0.1 |

This alternative set should not be used for the main Fig. 5/6 reproduction unless reproducing Extended Data Fig. 13f.

## Agent Architecture

The paper uses two independent recurrent actor-critic agents. Chaser and explorer do not share parameters.

Paper equations:

```text
h_t = ReLU(W_input x_t + b_input + W_rec h_{t-1} + b_rec)
a_t = W_action h_t + b_action
v_t = W_value h_t + b_value
```

Dimensions:

- `x_t`: `R^200`
- hidden state `h_t`: `R^256`
- `W_input`: `R^(256 x 200)`
- `W_rec`: `R^(256 x 256)`
- `W_action`: `R^(4 x 256)`
- action logits `a_t`: `R^4`
- value output `v_t`: scalar

Official code implementation:

- RLlib custom recurrent model: `simple_rnn_v2_3_2.py`.
- Uses `torch.nn.RNN(..., nonlinearity="relu")`.
- RNN hidden size: `256`.
- Action and value are parallel linear readouts from the RNN output.
- `max_seq_len = 20` in the RLlib config.

## Training Protocol

Paper Methods:

- Algorithm: PPO via RLlib.
- Training horizon: 20,000 epochs.
- Each epoch: 4,000 environment steps sampled from 40 complete episodes.
- Episode length during training: 100 timesteps.
- PPO: KL-divergence penalty on policy updates, RLlib default parameters.
- L2 regularization on model weights with reported loss weight `lambda = 0.3`.
- Ten independent pairs of agents for each task condition: social and non-social.

Official code:

- Dependency target: Python 3.8, Ray/RLlib 2.2.0, old Torch stack.
- `MultiAgent_Train_Final.py` exposes:
  - `--kl-coeff` default `0.2`;
  - `--clip-param` default `0.3`;
  - `--l2-curr` default `3`;
  - `--l2-inp` default `0`;
  - `--train-iter` default `2000`;
  - `--checkpoint-freq` default `100`;
  - `--num-workers` default `5`;
  - `--num-gpus` default `1`.
- Demo `params.json` files show `l2_lambda = 3.0`, `kl_coeff = 0.2`, `rnn_hidden_size = 256`, and `num_workers = 7`.

Open implementation questions:

- Paper says L2 `lambda = 0.3`, while official code defaults and demo params show `3.0`. We should resolve this before claiming exact reproduction.
- Paper says 20,000 epochs; official script default is 2,000 iterations, but evaluation code expects `checkpoint-020000`. For paper-faithful reproduction, use 20,000.
- RLlib default parameters are version-sensitive. Exact reproduction should use Ray 2.2.0, not current latest Ray.

## Evaluation Protocol

### Main behavioral evaluation

Paper evaluates agents against a standardized random opponent so that all agents and training stages face a common benchmark.

- Random agent: samples actions uniformly at each timestep.
- Evaluation against random agent: 100 episodes, 100 timesteps each.
- Chaser metrics:
  - number of collisions;
  - percentage of time partner is in vision;
  - average distance to partner;
  - number of new fields.
- Explorer metrics:
  - number of new fields;
  - average distance from partner.
- Fig. 5 uses the first 100 timesteps in rollouts for metrics.

### Neural/behavior analysis rollouts

For neural-behavior analyses:

- Evaluate each trained pair for 25 episodes.
- Each analysis episode lasts 500 timesteps.
- Exclude degenerate episodes where agents stop moving in the same state or repeat movements within the same tile for more than 1% of episode length.

Degenerate-rule interpretation used by this repo (no official reference
implementation exists; the official analysis repo contains only MATLAB PLSC
utilities):

- A step is *stuck* for an agent when it issued a valid action but stayed in
  the same tile for a non-social reason (wall bump or stationarity).
  Collision-blocked steps are excluded: collision is the rewarded social
  outcome, and counting it would flag every successful social episode
  (a trained chaser collides far more than 1% of steps).
- "For more than 1% of episode length" is read as a sustained duration: the
  longest consecutive stuck run must exceed the threshold (strictly), so
  scattered one-off wall bumps do not accumulate into an exclusion.
- The episode is excluded when either agent's longest stuck run, or the
  longest jointly-stuck run, exceeds the threshold.

## Neural Analyses To Reproduce

### SVM decoding

Goal: test whether RNN activity represents social events and partner behavior.

Targets:

- chaser: collision and partner escape;
- explorer: collision and partner approach.

Use only timesteps when the partner is in vision for approach/escape decoding.

Metric:

```text
balanced_accuracy = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
```

### Shared neural space via PLSC

Goal: identify shared dimensions between two agents' RNN activity.

Procedure:

- Use the same PLSC framework as the mouse neural analysis.
- Identify dimensions that maximize cross-covariance between chaser and explorer hidden activity.
- Assess significance against temporally permuted controls.
- Compare:
  - social trained agents;
  - non-social agents;
  - non-social agents with partial partner vision;
  - non-social agents with full partner vision.

### Partner representation in neural action space

Goal: quantify partner-behavior information in the action-relevant subspace of the RNN.

Neural action subspace:

```text
z_t = W_action.T @ W_action @ h_t
```

Predictors for AI agents:

- collision events;
- x/y positions of self and partner;
- actions of self and partner;
- self behaviors;
- partner behaviors.

The paper assesses the non-redundant contribution of partner behavior by fitting a linear model and measuring the reduction in variance explained after permuting partner-behavior predictors.

Important note: the paper explicitly says `W_action.T @ W_action` does not have to be a projection matrix because the downstream model is linear.

### Perturbation

Goal: test whether shared dimensions causally affect social behavior.

Procedure:

1. Concatenate rollout episodes.
2. Identify the top 10 PLSCs for each agent pair.
3. For chaser inference, project RNN activation into the null space of the PLSCs:

```text
h_perturbed = (I - P @ P.T) @ h
```

4. Compute action logits from `h_perturbed`.
5. Evaluate over 100 episodes of 100 timesteps.

Metrics:

- number of collisions;
- percentage of time partner is in vision;
- average distance between chaser and explorer.

Control:

- Remove comparable random non-overlapping variance using top 25 random principal components from temporally permuted activity.

## Current PyTorch Port Status

The repo provides two explicitly separated training regimes. The completed
`paper_marl_2026` experiments used the original fast full-batch learner through
the `modern_fast`/`paper_text` presets. The new `official_code` preset preserves
the vectorized PyTorch environment while reproducing the learning dynamics
inherited by the released Ray RLlib 2.2 training script.

Implemented:

- Paper-style `10 x 10` grid world.
- `7 x 7` partner vision via `vision_radius = 3`.
- `200`-dimensional observation.
- `100`-step episodes by default.
- Supplementary Table 3 social and non-social rewards.
- Social and non-social task variants.
- Partner-visibility modes: `partial`, `none`, `full`.
- Official action order: up, right, down, left.
- Sequential movement with chaser moving first.
- Collision as a blocking event, not an episode terminator.
- Independent 256-unit recurrent actor-critic policies.
- Recurrent PPO-style clipped policy updates in current PyTorch.
- Random-opponent evaluation modes for Fig. 5-style behavioral checks.
- Rollout export for hidden states, observations, actions, positions, rewards, collision events, approach/escape events, visibility flags, and new-field events.
- Safetensors checkpoint and rollout artifacts, with config/metrics stored as JSON metadata.

Also implemented (2026-07-02 review round):

- Official event precedence for approach/escape (gated on collision and own new-field).
- Per-agent collision flags in step results and rollouts.
- Degenerate-episode exclusion with the interpretation documented above (longest stuck run, collision-blocked steps excluded).
- Rollout schema v3 time alignment: state series `(max_steps + 1)` aligned to states, `hidden[t]` produced `action[t]`, events describe transition `t -> t+1`; the alignment convention is stored in rollout metadata.
- Seeded, reproducible evaluation and rollout collection; the randomized agent's policy is never sampled, so the RNG stream is stable.
- Per-agent gradient clipping (a joint norm would couple the two independent agents).
- Checkpoint training state (optimizer, RNG, update index) and resume-from-checkpoint.
- RLlib-style value clipping (`value_clip`, default 10.0 = RLlib 2.2's
  `vf_clip_param` default, which the official code inherited). Empirically
  necessary: on a 4070 Ti Super, per-agent clipping made `modern_fast`
  (recurrent L2 = 0) learn fast enough that the bootstrapped value targets
  chased diverging predictions from ~update 60 into overflow by ~update 190
  (seed 0); `paper_text` (L2 = 0.3) was stable over 200 updates, but 20,000
  updates would enter the same fast-learning regime, so the reference clamp
  is applied everywhere.
- Fused agent rollout (`fused_agent_rollout`, on in the primary config): both
  agents' rollout forwards run as one stacked batch (gather + bmm). Bit-exact
  against the unfused path on CPU fp32 over a full 100-step recurrent horizon;
  on CUDA with TF32 the reduction order differs, so trajectories are not
  bit-identical (no run is bit-reproducible on CUDA anyway — cuDNN, TF32).
  Action sampling stays per-agent to preserve the RNG stream layout.
- Rollout-time finite guarding relies on the post-rollout validation of stored
  log-probs/values/advantages/returns (still before any optimizer step); even
  a sync-free per-step accumulated flag cost ~24% of rollout wall time in
  kernel launches, and non-finite values necessarily propagate into the
  validated tensors.

Official-dynamics learner implemented (2026-07-10):

- Separate explorer and chaser parameters, Adam optimizers, optimizer moments,
  and adaptive KL coefficients, with the official explorer-first policy order.
- RLlib 2.2 PPO defaults inherited by the released script: learning rate
  `5e-5`, GAE lambda `1.0`, 30 epochs, exact categorical KL penalty and target,
  zero entropy bonus, no gradient clipping, and PyTorch-default initialization.
- Recurrent `max_seq_len=20` batching and Ray 2.2's released 33-minibatch
  slicing behavior, including the eight-step boundary overlap.
- Exact resume of both policies, both optimizer states, adaptive KL state, and
  action/minibatch/CUDA RNG state.
- `official_exclude_last` spawn mode for the released code's `0..8` initial
  coordinate range, while `full_grid` remains the declared paper-text/modern
  regime.
- CUDA Triton/reference environment equivalence, a healthy five-update
  calibration, and zero-failure 10-worker capacity calibration are recorded in
  `experiments/paper_marl_official_dynamics_2026/RUNS.md`.

Remaining gaps:

- The new official-dynamics path is algorithmically aligned, not a bit-for-bit
  recreation of Ray's five worker processes. Vectorized collection, random
  number interleaving, and the newer PyTorch/CUDA kernels can produce different
  trajectories.
- The official-dynamics path has passed smoke and CUDA calibration, but has not
  yet trained the declared ten independent social and ten non-social pairs.
  Consequently, the completed `paper_marl_2026` results must remain labeled as
  the earlier fast-learner reproduction until the new panel finishes.
- PLSC shared-dimension extraction with temporal-permutation significance is
  implemented in `mouse_run_run/plsc.py` (the reusable core) and consumed by
  both `scripts/analysis/analyze_shared_neural.py` (offline: z-scored hidden
  states, cross-covariance SVD, per-rank permutation null, significant-dimension
  count and top-dimension correlation, aggregated social vs non_social, C3) and
  the interactive viewer's "Shared & Unique Subspace" panel (pools self-play
  episodes for a checkpoint, shows the cross-covariance → SVD → shared/unique
  split as a numbered pipeline, the spectrum vs null, the variance split, and a
  pooled episode's shared/unique norm over time; it warns when the pool is
  rank-deficient). Not yet implemented: SVM balanced-accuracy decoding,
  neural-action-space partner-representation GLMs, and null-space perturbation
  (C4/C5). `analyze_neural.py` and the Network Activity panel provide
  single-network diagnostics (PCA, rasters, event-triggered speed, visibility
  tuning), which are exploratory rather than a reproduction of the paper's
  cross-agent analyses.
- The L2 regularization discrepancy between paper text (`lambda = 0.3`) and
  official code/demo params (`3.0`) cannot be resolved from the released
  materials. The new experiment therefore pre-declares separate
  `methods_text_l2` and `official_code_l2` sensitivity panels; both use the
  official unsquared recurrent-weight norm.

Recent smoke result:

- An earlier 20-update run was used as a systems check, not as a converged model.
- 128 stochastic self-play evaluation episodes produced about `1.60` collisions per episode.
- 128 stochastic random-explorer evaluation episodes produced about `1.88` collisions per episode.
- New checkpoints and rollout files should use `.safetensors`; a smoke checkpoint at `runs/safetensors-smoke.safetensors` and rollout at `runs/safetensors-rollout-smoke.safetensors` verified the format.

## Recommended Reproduction Plan

1. Validate modern env semantics against selected traces from the official environment.

2. Run training at meaningful scale.
   - one social pair and one non-social pair;
   - then ten-seed social/non-social runs;
   - evaluate each checkpoint against random opponents.

3. Implement analysis modules.
   - SVM balanced-accuracy decoding;
   - PLSC shared dimension extraction and temporal permutation significance —
     done (`scripts/analysis/analyze_shared_neural.py`);
   - neural-action-space partner representation;
   - null-space perturbation of top 10 PLSCs;
   - random-PC perturbation control.

4. Validate in stages.
   - random-agent baseline;
   - Fig. 5 behavioral metrics;
   - Fig. 6 neural metrics;
   - perturbation controls.

## Sources

- Local paper PDF: `/Users/jacoblincool/Downloads/s41586-025-09196-4.pdf`.
- Nature article page: https://www.nature.com/articles/s41586-025-09196-4.
- Official supplementary PDF: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09196-4/MediaObjects/41586_2025_9196_MOESM1_ESM.pdf.
- Official MARL code: https://github.com/hongw-lab/marl_environment_chase.
- Official analysis code noted by paper: https://github.com/hongw-lab/code_for_2024_zhang-phi.

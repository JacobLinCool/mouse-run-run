# Zhang et al. 2025 MARL Replication Notes

## Analysis Plan

- Paper set: Zhang et al. (2025), "Inter-brain neural dynamics in biological and artificial intelligence systems", Nature 645, 991-1001, DOI 10.1038/s41586-025-09196-4.
- Local source: local copy of the paper PDF (`s41586-025-09196-4.pdf`).
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

### What the released config actually pins (read 2026-08-18)

`Demo/Trained_model_examples/model1/params.json` is a partial RLlib config dump
with exactly 12 overridden keys:

`clip_param 0.3`, `kl_coeff 0.2`, `grad_clip null`, `framework torch`,
`num_gpus 1`, `num_workers 7`, `env_config {height 10, width 10, ts 100}`,
`model {custom_model rnn_noFC, fc_size 200, l2_lambda 3.0, l2_lambda_inp 0.0,
rnn_hidden_size 256, max_seq_len 20}`, `multiagent`, `callbacks`, `env`,
`_fake_gpus`.

Every optimizer setting is therefore an inherited RLlib 2.2.0 PPO default, read
from `ray-2.2.0/rllib/algorithms/ppo/ppo.py`:

| Setting | Value | Source line |
| --- | --- | --- |
| `lr` | `5e-5` | 107 |
| `num_sgd_iter` | `30` | 94 |
| `sgd_minibatch_size` | `128` | 93 |
| `train_batch_size` | `4000` | 106 |
| `lambda_` | `1.0` | 91 |
| `kl_target` | `0.01` | 102 |
| `vf_clip_param` | `10.0` | 100 |
| `vf_loss_coeff` | `1.0` | 96 |
| `entropy_coeff` | `0.0` | 97 |

Two consequences:

- The learning rate appears in neither the paper nor the training CLI. `5e-5`
  is what the released runs used, but only as a default that comes paired with
  30 SGD epochs over 128-sample minibatches. Our `GOAL_DIRECTED_PPO` keeps one
  full-batch epoch per update, roughly 470x fewer optimizer steps, so it needs
  a compensating learning rate; `1e-3` reproduces the behavioural separation
  that `5e-5` does not. `PROTOCOL_PPO` keeps the released pairing intact.
- The default `train_batch_size = 4000` is exactly the paper's "4,000
  environment steps per epoch from 40 complete episodes", which corroborates
  that the batch settings were left at their defaults rather than tuned.

Open implementation questions:

- Paper says L2 `lambda = 0.3`, while official code defaults and demo params show `3.0`. The released `params.json` confirms the trained demo models used `3.0`; our study uses the paper's `0.3`. We should resolve this before claiming exact reproduction.
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

## What The Released Analysis Code Covers

`hongw-lab/code_for_2024_zhang-phi` is 12 MATLAB files, all written for the
animal data. Its README groups them as:

- **PLSC** — `plsc.m` (the transformation), `computeSharedNullDistribution.m`
  (null distribution and significant-dimension count), `getNeuralSpace.m`
  (shared and unique spaces). With `Utils/tempShift.m` and `Utils/timePermute.m`
  this is the direct source for our `analyses/paper_plsc.py`, and the one method
  where a line-by-line numerical comparison is possible.
- **CCA behaviour space** — `computeCoordNullDistribution.m`,
  `getBehaviorSpace.m`.
- **ROC** — `simulateROC.m`, `simulateROC_Wrapper.m`.
- **Non-redundant variance via PLSR** — `computeNonRedundantVar.m`.

Not released anywhere, and therefore reconstructed here from the paper text:
the agent-side neural analyses (SVM decoding of the artificial agents, the
Fig. 5o-r movement geometry), the causal readout and recurrent interventions,
all figure code, and the glue that carries RLlib checkpoints into analysis.

## Current PyTorch Port Status

The runnable implementation is the strict typed study in
[`experiments/zhang_2025_rnn`](../../experiments/zhang_2025_rnn). It provides:

- the paper-aligned chaser–explorer environment and independent recurrent
  actor-critic policies;
- explicit formal, behavior-only development, protocol-sensitivity, and tiny
  CPU smoke definitions;
- train, behavior, neural, and causal stages through one validated study
  runner;
- safetensors checkpoints, Parquet analysis tables, deterministic planning,
  and strict resume/config validation;
- direct tests for environment dynamics, PPO math, analysis controls, study
  planning, artifact validation, and stage execution.

Operational commands and current workload definitions live in the study
[`README`](../../experiments/zhang_2025_rnn/README.md). Historical v1 runners,
JSON experiment configs, and artifact readers are available in Git history
only; the active runtime has no compatibility or fallback path for them.

## Sources

- Local paper PDF: local copy of `s41586-025-09196-4.pdf`.
- Nature article page: https://www.nature.com/articles/s41586-025-09196-4.
- Official supplementary PDF: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09196-4/MediaObjects/41586_2025_9196_MOESM1_ESM.pdf.
- Official MARL code: https://github.com/hongw-lab/marl_environment_chase.
- Official analysis code noted by paper: https://github.com/hongw-lab/code_for_2024_zhang-phi.

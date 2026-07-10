# Onboarding

This guide is for new contributors to the multi-agent reinforcement learning project.

## Project Idea

The project trains two neural-network agents in a small grid world. At each
timestep, each agent sees a flattened two-channel map:

- channel 0 marks the agent's own position;
- channel 1 marks the other agent's position when visible.

The chaser learns to catch the explorer. The explorer learns to avoid the chaser
and explore new cells. After training, the analysis code studies the hidden
activations of the two neural networks and asks whether their internal activity
contains shared structure.

## Vocabulary

- **Agent**: a program that chooses actions in an environment.
- **Environment**: the grid world that updates positions and gives rewards.
- **Observation**: the input vector an agent receives before choosing an action.
- **Policy**: the neural network that turns an observation into action scores.
- **Value**: the network's estimate of future reward.
- **Rollout**: one collected batch of agent-environment interaction.
- **PPO**: the reinforcement learning update rule used to improve the policies.
- **Hidden state**: the neural activity vector inside a policy network.
- **PLSC**: an analysis that finds dimensions shared between the two agents'
  hidden-state trajectories.

## Read The Code In This Order

1. `mouse_run_run/env.py`
   Start here to understand the rules of the world: reset, observation
   creation, movement, collisions, rewards, and visibility.

2. `mouse_run_run/policy.py`
   Read `PolicyBase` and `RNNActorCritic` first. The other architectures are
   comparison models.

3. `mouse_run_run/training_rollout.py`
   This file answers: "How do we collect one batch of experience from the two
   agents before learning from it?"

4. `mouse_run_run/ppo.py`
   This file answers: "Given the rollout, how do we compute advantages, losses,
   gradients, and one optimizer update?"

5. `mouse_run_run/train.py`
   This is the conductor. It wires together the environment, agents, rollout
   collection, PPO update, metrics, and checkpoint saving.

6. `mouse_run_run/analysis.py` and `mouse_run_run/plsc.py`
   Read these after the training loop makes sense. They operate on saved
   hidden-state tensors rather than controlling training.

## A Tiny Experiment

Run this from the repository root:

```bash
uv run train-marl \
  --updates 1 \
  --batch-size 2 \
  --max-steps 5 \
  --hidden-size 8 \
  --device cpu \
  --log-every 1 \
  --checkpoint runs/onboarding-smoke/checkpoints/latest.safetensors
```

This creates only a tiny checkpoint. It should finish quickly and print one
metrics line. The important fields are:

- `collisions`: average number of chaser-explorer collisions per episode;
- `chaser_return` and `explorer_return`: average rewards;
- `vision`: fraction of steps where each agent could see the partner;
- `loss` and `kl`: PPO training diagnostics.

After that, validate the checkpoint:

```bash
uv run validate-marl-checkpoints runs/onboarding-smoke
```

## How The RL Loop Fits Together

The training loop repeats four steps:

1. Reset many grid worlds in parallel.
2. Ask both policies for actions at every timestep.
3. Store observations, actions, rewards, old log probabilities, and values.
4. Run PPO to make the stored actions more likely when they led to better
   future reward than expected.

The important separation is:

- `training_rollout.py` records what happened;
- `ppo.py` learns from what happened;
- `train.py` repeats the process and saves evidence.

## What To Change First

Good first tasks:

- add or improve a test for a reward or visibility rule;
- run a tiny smoke training command and inspect the checkpoint metadata;
- add a small explanatory comment where the math is not obvious;
- compare social versus non-social behavior using `evaluate-marl`.

Avoid changing paper-scale configs until the small commands and tests pass.

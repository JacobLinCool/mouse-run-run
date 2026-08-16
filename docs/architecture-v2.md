# Architecture v2

`mouse-run-run` is split into a stable execution core and editable research
experiments.  The core owns multi-agent stepping, PPO, artifacts, progress,
analysis contracts, and replay.  An experiment owns the environment semantics,
reward, policy architecture, renderer, and the analysis/intervention recipes it
wants to expose.

The core never switches behavior based on a paper name or an old artifact
shape.  All agent-indexed values are keyed by a stable `AgentId`; all policy
representations are exposed as named activation sites.  Training, ordinary
rollout collection, and causal interventions use the same simulation engine.

Artifacts written by this architecture use strict versioned schemas (including
the state-complete `mrr-checkpoint-v3`). Model, optimizer, RNG, rollout,
activation, and dense analysis arrays are safetensors. Episode
and analysis tables are Parquet.  Existing artifacts remain archived in their
current directories and are intentionally not accepted by the v2 readers.

The Zhang 2025 layer adds a typed `StudyDefinition` above this core. It owns
multi-seed scheduling, paper-specific evaluation recipes, seed-level
aggregation, and scientific gates; it does not add another environment loop or
checkpoint format. `study plan` is a pure validation operation, while
`study run` composes the same training, rollout, intervention, and artifact
contracts used by standalone commands.

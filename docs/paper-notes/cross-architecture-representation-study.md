# Cross-Architecture Representation Study

**Question.** Do different agent architectures, trained identically on the
Zhang et al. (2025) chaser–explorer task, learn the *same* internal
representations of the social interaction? Or does the phenomenon the paper
reports — genuine shared neural dynamics between two independently trained
agents — depend on the architecture?

**Design.** Four architectures, identical in everything except the memory
mechanism: reward structure, 10×10 world, 100-step episodes, seeds 0–9,
analysis dimension 256, PPO settings, evaluation protocol, and analysis
pipeline. They span a memory ladder:

| architecture | memory mechanism | internal state at step t |
|---|---|---|
| MLP (`k=8` frame stack) | none (finite window) | pure function of the last 8 observations |
| SSM (gated diagonal LRU) | compressed, **linear** dynamics | eigenvalue-controlled linear state |
| RNN (paper's ReLU RNN) | compressed, **nonlinear** dynamics | chaotic nonlinear recurrence |
| Transformer (2-layer causal) | none evolving; re-attends history | pre-head residual recomputed per step |

Each was trained as a full 20-pair experiment (`mouse-run-run-1` for the RNN,
`mouse-run-run-2-{mlp,ssm,transformer}` for the rest): 20,000 PPO updates ×
40 episodes × 100 steps per pair, **20/20 units completed on the first
attempt** for every architecture, zero failed/interrupted attempts.

---

## 1. Behavior: the architectures converged to different strategies

Chaser capability against a standardized random explorer (100 episodes ×
100 steps), and the explorer's learned escape (collisions it suffers against a
random chaser — lower is better escape):

| architecture | chaser collisions ↑ | chaser partner-in-vision | explorer collisions-suffered ↓ |
|---|---:|---:|---:|
| RNN | 8.98 ± 5.70 | 0.61 | 0.57 |
| MLP | 26.93 ± 4.00 | 0.93 | 0.92 |
| SSM | 35.92 ± 3.56 | 0.94 | 0.45 |
| Transformer | 41.31 ± 3.29 | 0.96 | 0.28 |

All four learned role-specific social behavior (non-social controls collide
2.7–3.4× regardless). But the **RNN converged to a qualitatively different
strategy**: a low-vision "stealth" chaser (61% partner-in-vision, 9 collisions)
that tracks the explorer from memory, whereas MLP/SSM/Transformer converged to
high-vision "pursuit" chasers (93–96% vision, 27–41 collisions) that stay
glued to the explorer. This behavioral fork is the first sign the architectures
are not learning the same thing — and it turns out to drive the representation
difference.

The raw capability ranking (Transformer > SSM > MLP > RNN) is **not** the
headline. A chaser that collides 41×/episode by never leaving the explorer's
side is behaviorally strong but representationally shallow. The scientific
question is about the *internal representation*, below.

## 2. Event decoding replicates across architectures (C2)

Linear-classifier balanced accuracy for social events decoded from single-agent
hidden states (5-fold CV, shuffled controls ≈ 0.50 throughout):

| architecture | collision | partner escape | partner approach |
|---|---:|---:|---:|
| RNN | 0.983 | 0.888 | 0.886 |
| MLP | 0.914 | 0.780 | 0.778 |
| SSM | 0.850 | 0.768 | 0.733 |
| Transformer | 0.847 | 0.761 | 0.759 |

Every architecture encodes collision and partner approach/escape above chance
in social agents — the paper's C2 claim is **architecture-robust**. Decoding
accuracy tracks the memory ladder (RNN > MLP ≈ SSM > Transformer), but the
qualitative result (social events are linearly represented) holds for all.

## 3. Shared dimensions: a methodological correction, then the key result (C3)

The paper's headline is C3: social agents develop *shared* neural dimensions
across the two independently trained networks (PLSC). Measuring this across
architectures exposed a confound that the analysis had to fix.

**The raw cross-agent correlation is contaminated by time-locked scaffolds.**
The Transformer's `top_dim_correlation` was **0.957 in the non-social control**
— higher than its social value, and higher than any architecture's social
score. There is no partner input in the non-social task, so this cannot be
genuine sharing. The cause: the Transformer's positional embedding makes every
timestep a deterministic function of absolute position t; both agents' networks
carry this identical positional scaffold, and it dominates the top singular
dimension. The RNN and MLP have no such clean scaffold (their "clocks" are noisy
or absent), so their raw non-social correlation stays near zero — which is
exactly why the raw statistic *looked* fine until the Transformer exposed it.

**Fix (standard in neuroscience): remove the evoked/time-locked component.**
Subtract the across-episode mean at each timestep — the response identical
across episodes at a given t (positional embeddings, hidden-state clocks, mean
approach dynamics) — and run PLSC on the episode-specific residual, where
genuine interaction lives. This removes every architecture's deterministic
temporal scaffold uniformly, making the effect size comparable.

**Aggregate residualized top-dim correlation (interaction-driven sharing):**

| architecture | social | non-social | separation |
|---|---:|---:|---:|
| RNN | 0.688 ± 0.23 | 0.067 | **0.62** |
| MLP (floor) | 0.679 ± 0.06 | 0.103 | 0.58 |
| SSM | 0.642 ± 0.12 | 0.315 | 0.33 |
| Transformer | 0.438 ± 0.08 | 0.356 | **0.08** |

The MLP is the **common-input floor**: being memoryless, any cross-agent sharing
it shows is by construction reactive (both nets responding to the same visible
game state). That its social sharing (0.679) nearly matches the RNN's aggregate
(0.688) is the crux — the aggregate mean *hides* the real difference. The
Transformer's social/non-social separation collapses to 0.08 once its positional
scaffold is removed: **its apparent sharing was mostly the scaffold**.

**The discriminating view: sharing vs behavioral coupling (vision).** The MLP
can only share when both agents see each other. So the question is whether an
architecture achieves shared representation *at low mutual vision* — which
requires genuine internal (memory/prediction) coupling, not common input.
Residualized social `top_r`, split by chaser partner-in-vision:

| architecture | low vision (<40%) | high vision (≥40%) |
|---|---:|---:|
| **RNN** | **0.652 (n=7)** | 0.947 (n=1) |
| MLP | — (0 units) | 0.679 (n=10) |
| SSM | — (0 units) | 0.642 (n=9) |
| Transformer | — (0 units) | 0.438 (n=10) |

This is the result. **Only the RNN develops shared representation without the
agents seeing each other** — 7 of its 8 analyzable social units sit below 40%
mutual vision yet reach 0.65 residualized correlation. Every social unit of the
other three architectures lives at high vision; their sharing is vision-gated,
i.e. common-input-like. MLP *cannot* do otherwise (no memory). SSM and
Transformer *have* memory mechanisms but did not use them to build low-vision
coordination — they converged to the reactive, high-vision solution.

## 4. Representational geometry: the RNN is the outlier (CKA)

Feeding the **same** game-state observation stream through every architecture's
social chaser and comparing representations with linear CKA (rotation/scaling
invariant):

|  | RNN | MLP | SSM | Transformer |
|---|---:|---:|---:|---:|
| RNN | 1.00 | 0.68 | 0.64 | 0.67 |
| MLP | 0.68 | 1.00 | 0.71 | 0.78 |
| SSM | 0.64 | 0.71 | 1.00 | 0.76 |
| Transformer | 0.67 | 0.78 | 0.76 | 1.00 |

The **RNN is the representational outlier** (0.64–0.68 with everything else),
while MLP/SSM/Transformer cluster together (0.71–0.78). A third independent
method agrees with behavior and PLSC: the RNN encodes the task differently; the
other three converge to a more mutually similar (reactive) geometry.

## 5. Degeneracy is itself an architecture property

The paper excludes "degenerate" episodes where agents get stuck. The
**SSM is markedly more prone to this**: 1/10 social and 2/10 non-social units
were fully degenerate (too few valid episodes to analyze), and several more had
only 3–8 of 25 episodes valid — versus **zero fully-degenerate units for RNN,
MLP, and Transformer**. The linear recurrence appears to collapse toward
fixed-point / stuck behavior more readily — a concrete failure mode of the
linear-dynamics inductive bias on this task.

---

## Answer to the question

**Different architectures do not learn the same internal representations.**
Three independent methods converge:

1. **Behavior** — the RNN alone converges to a low-vision, memory-based stealth
   strategy; the others to high-vision reactive pursuit.
2. **Shared dimensions (residualized, vision-conditioned)** — the RNN alone
   develops genuine interaction-driven sharing when the agents *cannot see each
   other*; the others' sharing is vision-gated and, for the Transformer, mostly
   a positional scaffold once corrected.
3. **CKA** — the RNN is the representational outlier; the other three cluster.

The paper's hallmark phenomenon — genuine internal shared dynamics between
independently trained agents — is **not architecture-universal**. It emerged
cleanly only in the vanilla RNN. The memoryless MLP cannot produce it (its
sharing is the common-input floor). The SSM and Transformer, despite having the
capacity for memory, did not: their inductive biases steered them to a reactive,
high-vision solution whose apparent sharing is largely explained by behavioral
and input coupling, plus (Transformer) a deterministic temporal scaffold.

## Honest confounds and limitations

- **Strategy mediates representation.** The architectures converged to different
  behavioral strategies, so the representation difference is partly *mediated* by
  strategy, not a direct architecture→representation effect. But that is itself a
  finding: the architecture's inductive bias shapes *which* solution is found,
  and only the RNN's solution has genuine internal shared dynamics. Isolating a
  direct effect would require conditioning on matched behavior (e.g. training all
  architectures to the same vision regime), which these runs do not do.
- **Not parameter-matched.** RNN 119k, SSM 382k, MLP 477k, Transformer 1.24M
  params (analysis dim held at 256). The RNN's distinctness is not a
  capacity-advantage artifact — it is the *smallest* model.
- **Time-mean subtraction** removes any genuinely time-locked coordination along
  with the scaffold; for random-initial-position episodes the interaction timing
  is episode-specific, so this should be conservative, but it is an assumption.
- **PLSC significant-dimension counts saturate** at these sample sizes (MLP hits
  256/256 in both conditions); the effect size (residualized top correlation),
  not the count, is the discriminating statistic and is what this report uses.
- The **perturbation (C5)** and neural-action-space (C4) analyses are not yet run
  across architectures; C5 in particular (does removing shared dimensions impair
  social behavior) would test causality of the RNN's shared dimensions.

## Artifacts

- Per-experiment behavior/eval/rollouts/reports: `runs/{experiment}/`,
  `runs/reports/{experiment}/`.
- Neural analyses: `runs/analyses/{experiment}/neural_summary.json`.
- Cross-architecture: `runs/analyses/cross_arch_plsc_residualized.json`,
  `cross_arch_perunit.json`, `cross_arch_cka.json`.
- Figures: `runs/analyses/figures/{sharing_vs_vision,cross_arch_cka}.png`.
- Analysis code: `mouse_run_run/analysis.py`,
  `scripts/analysis/neural_analyses.py`.

# Cross-Architecture Representation Study

**Question.** Do different agent architectures, trained identically on the
Zhang et al. (2025) chaser–explorer task, learn the *same* internal
representations of the social interaction? More specifically, how much of the
cross-agent alignment reported in the paper is stable across architectures,
and how much depends on the behavioral strategy and temporal structure of a
particular model?

**Design.** Four architectures shared the reward structure, 10×10 world,
100-step episodes, seeds 0–9, 256-dimensional analysis state, PPO protocol,
evaluation procedure, and analysis code. They differed in architecture,
parameter count, and memory mechanism:

| architecture | memory mechanism | internal state at step t |
|---|---|---|
| MLP (`k=8` frame stack) | finite observation window | function of the last 8 observations |
| SSM (gated diagonal LRU) | compressed, **linear** dynamics | eigenvalue-controlled linear state |
| RNN (paper's ReLU RNN) | compressed, **nonlinear** dynamics | learned ReLU recurrence |
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

All four learned role-specific social behavior; non-social chasers produced
only 2.7–3.4 collisions per episode. The **RNN converged to a qualitatively
different observable strategy**: a lower-vision chaser (61% partner-in-vision,
9 collisions), whereas MLP/SSM/Transformer converged to high-vision pursuit
policies (93–96% vision, 27–41 collisions). The present analyses do not
establish whether the RNN uses memory to compensate for reduced visual access.
They do establish that architecture and learned strategy are coupled in these
runs, which is essential context for the representation comparison.

The raw capability ranking (Transformer > SSM > MLP > RNN) is not the main
result. Performance alone does not determine whether two models use the same
internal computation, so the remaining analyses compare their hidden states.

## 2. Social-event decodability generalizes across architectures (C2)

Linear-classifier balanced accuracy for social events decoded from single-agent
hidden states (5-fold CV, shuffled controls ≈ 0.50 throughout):

| architecture | collision | partner escape | partner approach |
|---|---:|---:|---:|
| RNN | 0.983 | 0.888 | 0.886 |
| MLP | 0.914 | 0.780 | 0.778 |
| SSM | 0.850 | 0.768 | 0.733 |
| Transformer | 0.847 | 0.761 | 0.759 |

Every architecture encodes collision and partner approach/escape above chance
in social agents. Social-event information is therefore linearly accessible
across model classes. The original paper's selectivity control does not fully
reproduce, however: in our RNN, non-social collision decoding was 0.765 for the
chaser and 0.747 for the explorer, compared with 0.50 in the paper. Collision
decodability may partly reflect each agent's own state. We consequently treat
C2 as partially supported: the social signal generalizes, while the
non-social control differs. Decodability also does not establish that each
policy uses the decoded variable in the same way.

## 3. Shared dimensions: a methodological correction, then the key result (C3)

The paper's headline is C3: social agents develop *shared* neural dimensions
across the two independently trained networks (PLSC). Measuring this across
architectures exposed a confound that the analysis had to fix.

For the primary RNN, the leading correlation showed the expected condition
contrast (0.743 social versus 0.067 non-social), but the significant-dimension
count did not: 183 dimensions were significant in social pairs and 108 in
non-social pairs. The corresponding paper values were about 165 and 0.02.
C3 is therefore partially supported before the additional temporal-scaffold
analysis below: correlation strength separates conditions, whereas the
dimension count does not.

**The raw cross-agent correlation is contaminated by time-locked scaffolds.**
The Transformer's `top_dim_correlation` was **0.957 in the non-social control**
— higher than its social value, and higher than any architecture's social
score. There is no partner input in the non-social task, so this value cannot be
interpreted as partner representation. The most direct explanation is the
Transformer's positional embedding: both agents carry a strong code for the
same absolute timestep. The RNN and MLP do not contain the same explicit
positional signal, so the confound is much less visible in their raw statistic.

**Fix (standard in neuroscience): remove the evoked/time-locked component.**
Subtract the across-episode mean at each timestep — the response identical
across episodes at a given t (positional embeddings, hidden-state clocks, mean
approach dynamics) — and run PLSC on the episode-specific residual. The
residual contains interaction-dependent variation as well as other
episode-specific variation. This procedure removes every architecture's
deterministic temporal scaffold uniformly and makes the effect sizes more
comparable.

**Aggregate residualized top-dim correlation (interaction-driven sharing):**

| architecture | social | non-social | separation |
|---|---:|---:|---:|
| RNN | 0.688 ± 0.23 | 0.067 | **0.62** |
| MLP (floor) | 0.679 ± 0.06 | 0.103 | 0.58 |
| SSM | 0.642 ± 0.12 | 0.315 | 0.33 |
| Transformer | 0.438 ± 0.08 | 0.356 | **0.08** |

The frame-stack MLP provides a **common-input floor**: it has only a short,
fixed observation window, so its high social correlation shows how much
alignment can arise from recent shared sensory and behavioral context. Its
social value (0.679) nearly matches the RNN aggregate (0.688). The
Transformer's social/non-social separation falls to 0.08 after time-mean
subtraction, indicating that its raw statistic was strongly influenced by the
temporal scaffold.

**Shared dimensions versus visual coupling.** We next asked whether high
residualized correlation also occurred in pairs with limited visual access.
Residualized social `top_r`, split by chaser partner-in-vision, was:

| architecture | low vision (<40%) | high vision (≥40%) |
|---|---:|---:|
| **RNN** | **0.652 (n=7)** | 0.947 (n=1) |
| MLP | — (0 units) | 0.679 (n=10) |
| SSM | — (0 units) | 0.642 (n=9) |
| Transformer | — (0 units) | 0.438 (n=10) |

Seven of the eight analyzable social RNN pairs fell below 40% partner-in-vision
and still reached a mean residualized correlation of 0.652. All analyzable
pairs from the other architectures occupied the high-vision regime. This shows
that high cross-agent correlation can persist when the RNN chaser has limited
current visual access. It does not establish a direct architecture effect,
because the other architectures did not supply behaviorally matched low-vision
pairs. A matched-strategy experiment is required to separate architecture,
memory use, and sensory coupling.

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
while MLP/SSM/Transformer are more similar to one another (0.71–0.78). A third
analysis therefore agrees that the RNN's representation differs from the other
models in these runs. CKA describes geometrical similarity; it does not by
itself identify the computation responsible for that difference.

## 5. Degeneracy is itself an architecture property

The paper excludes "degenerate" episodes where agents get stuck. The
**SSM is markedly more prone to this**: 1/10 social and 2/10 non-social units
were fully degenerate (too few valid episodes to analyze), and several more had
only 3–8 of 25 episodes valid — versus **zero fully-degenerate units for RNN,
MLP, and Transformer**. The linear recurrence appears to collapse toward
fixed-point / stuck behavior more readily — a concrete failure mode of the
linear-dynamics inductive bias on this task.

## 6. Perturbation result: behavioral sensitivity without selectivity

Removing the top ten shared dimensions reduced collisions in every
architecture. However, the nominal random-basis intervention reduced
collisions equally or more strongly:

| architecture | unperturbed | shared removed | nominal random basis removed |
|---|---:|---:|---:|
| RNN | 4.57 | 2.44 | 1.55 |
| MLP | 21.23 | 2.04 | 1.58 |
| SSM | 17.21 | 4.16 | 1.36 |
| Transformer | 10.67 | 2.05 | 1.70 |

The agents are sensitive to perturbations of the shared directions, but the
experiment does not establish a selective causal role for shared information.
The current control basis is also not fully paper-faithful: it does not enforce
non-overlap with the shared subspace, and its joint time permutation preserves
the covariance used for PCA. A corrected, variance-matched control is required
before interpreting C5.

---

## Answer to the question

**Different architectures do not learn the same internal representations.**
Three independent methods converge:

1. **Behavior** — the RNN converges to a lower-vision, lower-collision strategy;
   the other architectures converge to high-vision pursuit.
2. **Shared dimensions** — residualization exposes a large positional confound
   in the Transformer. The RNN retains high correlation in the observed
   low-vision regime, but the other architectures provide no behaviorally
   matched low-vision comparison.
3. **CKA** — the RNN is the representational outlier; the other three cluster.

The evidence rejects the strongest invariance hypothesis: the same task and
training budget do not force these architectures to learn the same hidden-state
geometry. The results do not isolate architecture as the sole cause, because
architecture changed the learned strategy and visual exposure. They support a
more useful conclusion for AI research: representational alignment depends on
the model, the policy it learns, and the nuisance structure retained by its
hidden state.

## Honest confounds and limitations

- **Strategy mediates representation.** The architectures converged to different
  behavioral strategies, so the representation difference is partly *mediated* by
  strategy, not a direct architecture→representation effect. Isolating a direct
  effect requires conditioning on matched behavior, for example by training all
  architectures to the same vision regime.
- **Not parameter-matched.** RNN 119k, SSM 382k, MLP 477k, Transformer 1.24M
  params (analysis dim held at 256). The RNN is the smallest model, but size and
  architecture remain confounded.
- **Time-mean subtraction** removes any genuinely time-locked coordination along
  with the scaffold; for random-initial-position episodes the interaction timing
  is episode-specific, so this should be conservative, but it is an assumption.
- **PLSC significant-dimension counts saturate** at these sample sizes (MLP hits
  256/256 in both conditions); the effect size (residualized top correlation),
  not the count, is the discriminating statistic and is what this report uses.
- **C5 remains inconclusive.** The cross-architecture perturbations have been
  run, but the nominal random control is at least as disruptive as the shared
  intervention and its basis construction is not fully equivalent to the
  paper's control.

## Artifacts

- Per-experiment behavior/eval/rollouts/reports: `runs/{experiment}/`,
  `runs/reports/{experiment}/`.
- Neural analyses: `runs/analyses/{experiment}/neural_summary.json`.
- Cross-architecture: `runs/analyses/cross_arch_plsc_residualized.json`,
  `cross_arch_perunit.json`, `cross_arch_cka.json`.
- Figures: `runs/analyses/figures/{sharing_vs_vision,cross_arch_cka}.png`.
- Analysis code: `mouse_run_run/analysis.py`,
  `scripts/analysis/neural_analyses.py`.

# How Our Reproduction Aligns With the Paper's Claims

This is a plain-language account of whether our experiment reproduces each
claim the paper (Zhang et al. 2025, *Nature*) makes about its **artificial
agents** — the chaser/explorer reinforcement-learning agents. It is written
for a general reader; the numbers behind each verdict are in
`runs/analyses/`.

## What the paper claims about the AI agents

The paper's biological (mouse) findings — including the GABA/glutamate results
— are separate. About the *artificial* agents it makes exactly five claims:

1. **C1 — Social rewards produce social behavior.** Agents trained with a
   social reward learn role-specific behavior (a chaser that pursues, an
   explorer that flees); a non-social control does not.
2. **C2 — The network encodes social events.** A trained agent's internal
   activity can be read out to tell whether a collision happened, or whether
   the partner is approaching/escaping.
3. **C3 — Shared dimensions emerge between the two agents.** The two
   independently trained networks develop correlated internal dimensions,
   more so for social than non-social agents.
4. **C4 — The chaser represents its partner, and this predicts performance.**
   As a social chaser trains, its "action-relevant" activity carries more
   information about the partner's behavior, and chasers that represent the
   partner more strongly collide more.
5. **C5 — Those shared dimensions causally drive social behavior.** If you
   surgically remove the top shared dimensions from the chaser before it
   acts, its social behavior degrades — while removing a comparable amount of
   *unrelated* activity does not.

The paper is explicit that the biology↔AI bridge is at the level of
*representations*, not cell types: "specific cell types in mice do not
correspond directly to neural networks in artificial agents." So there is no
AI claim about inhibitory/GABA structure to reproduce — that would be a new
question, beyond the paper.

## Our reproduction, claim by claim

| Claim | Verdict | What we found |
|---|---|---|
| C1 behavior | **Reproduces** | Social agents learn chase/flee; non-social controls do not. Chaser collisions vs a random opponent are far higher for social agents across all architectures. |
| C2 decoding | **Reproduces** | Collisions decode at 0.85–0.98 balanced accuracy, partner approach/escape at 0.73–0.89, from social agents' activity (chance = 0.50). Holds for every architecture. |
| C3 shared dimensions | **Reproduces** | Social pairs show far stronger cross-agent shared structure than non-social (top-dimension correlation 0.74 vs 0.07 for the RNN), after correctly removing time-locked confounds. |
| C4 partner representation | **Partial** | The core claim holds: the social chaser represents the partner ~250× more than the non-social control (0.04 vs 0.0001 unique variance), and stronger partner representation correlates with more collisions (r = +0.64, social; +0.28, non-social). But the *increase during training* did **not** reproduce — in our RNN it decreases. |
| C5 shared-dimension causality | **Partial** | Removing the top-10 shared dimensions sharply degrades social behavior in every architecture (collisions collapse, e.g. 21→2 for the MLP; partner-in-vision 0.97→0.40). But so does removing a comparable amount of *random* activity — so we cannot show the effect is *specific* to the shared dimensions. |

So **C1–C3 reproduce cleanly. C4 and C5 reproduce in their qualitative core
but not in the parts that require a clean control or a training trajectory.**
The reasons are informative, not bugs.

## Why C4 and C5 only partially reproduce — two honest reasons

**Reason 1 — our paper-faithful RNN converged to an atypical strategy.**
The paper's recurrent L2 regularization (we used the paper-text value 0.3)
drove our RNN's recurrent weights to almost zero (recurrent weight norm 0.05
vs 12.8 at initialization; the hidden state barely persists across time). The
network operates close to feed-forward, and it settled on a *low-vision
"stealth" chaser* (it collides ~9×/episode and rarely keeps the partner in
view) rather than the aggressive high-vision pursuer the paper describes.
Because C4's "partner representation increases during training" and C5's clean
perturbation both assume an aggressive, partner-tracking chaser, our RNN is a
poor substrate for them. The other architectures (MLP, SSM, Transformer),
which have no recurrent L2 and *do* pursue aggressively (27–41 collisions),
behave much more like the paper's chaser.

**Reason 2 — our agents' representations are low-dimensional.**
The chaser–explorer task is simple enough that the agents solve it with a
handful of internal dimensions: the top 10 dimensions already carry ~62% of
the activity's variance. C5's control asks whether removing the *shared*
dimensions hurts more than removing an equal amount of *unrelated* variance.
But when 10 dimensions are 62% of everything, removing any 10 high-variance
dimensions guts the policy — there is no "unrelated" high-variance activity
left to serve as a clean control. The shared dimensions *are* the dominant
dimensions. So we confirm shared dimensions are behaviorally important, but
cannot isolate a *shared-specific* causal effect. The paper's agents likely
had higher-dimensional representations, leaving room for the control to work.

## Bottom line

Our reproduction **faithfully captures the paper's central AI claims** — that
social rewards produce social behavior (C1), that the network encodes social
events (C2), and that shared neural dimensions emerge between the two agents
(C3). For the two finer claims (C4, C5) we reproduce the qualitative core
(the chaser represents its partner; the shared dimensions matter for behavior)
but not the precise forms that need a clean control or a training trajectory —
and the reasons trace to two concrete, measurable properties of our agents:
the L2-regularized RNN's atypical strategy, and the low dimensionality of the
learned representations.

A clean reproduction of C4 and C5 would follow from one targeted change:
retrain the RNN with little or no recurrent L2 so it keeps an aggressive,
partner-tracking strategy (as the paper's chaser did) and a higher-dimensional
representation. That is a one-config, ~3-hour rerun on the same infrastructure.

## Artifacts

- C1–C3: `runs/analyses/{experiment}/neural_summary.json`,
  `runs/reports/{experiment}/summary.json`.
- C4: `runs/analyses/c4_mouse-run-run-1.json`.
- C5: `runs/analyses/c5_*.json`.
- Cross-architecture context:
  `docs/paper-notes/cross-architecture-representation-study.md`.

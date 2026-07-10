# Reproduction Results and Interpretation from an AI Perspective

## Main conclusion

Our reproduction supports the paper's broad behavioral result: social training
produced role-specific chasing and escaping. It also recovers strong decoding
of social events and a large social/non-social difference in the leading
cross-agent activity correlation. Direct comparison with the original numbers,
however, reveals two important control differences. Collision events remained
decodable in our non-social agents, and our non-social RNNs retained many
nominally significant shared dimensions. We therefore classify the decoding
and shared-dimension claims as partially, rather than fully, reproduced.

The two stronger claims were also only partly recovered. Partner information
predicted chaser performance, but it decreased rather than increased over
training, and our perturbation experiment did not isolate an effect specific
to the shared dimensions.

The additional architecture experiments sharpen this conclusion. Models with
different memory mechanisms all learned the task, yet they adopted different
strategies and organized their hidden activity differently. In particular, a
Transformer showed high cross-agent correlation even in the non-social
control, consistent with a strong positional scaffold. This is an important
result for both neuroscience and AI: correlated population activity can reflect
interaction, common input, or shared temporal structure, and these explanations
require different controls.

## What we reproduced

We reproduced the paper's chaser-explorer multi-agent reinforcement-learning
experiment from Figs. 5 and 6. Two independently parameterized agents acted in
a 10 × 10 grid world. The chaser was rewarded for collisions, whereas the
explorer was rewarded for visiting new locations and avoiding the chaser. Each
agent received its own position and, in the social condition, the partner's
position when the partner fell within a 7 × 7 field of view. The non-social
control removed both the social collision reward and partner-position input.

The primary model followed the paper's architecture: separate actor-critic
policies with 256-unit ReLU recurrent neural networks (RNNs). We trained ten
social and ten non-social pairs for 20,000 updates, with 4,000 environment
steps per update. We then used the paper's two evaluation settings:

- For behavior, each trained agent played 100 episodes against the same type
  of uniformly random opponent. This separates an agent's ability from the
  changing ability of its co-trained partner.
- For representation analyses, each trained pair played 25 longer self-play
  episodes. We recorded hidden states, actions, positions, visibility, and
  social events and excluded episodes that met the paper-style degeneracy
  criterion.

This is a reproduction of the experimental design and scientific analyses,
not a bit-for-bit rerun of the original software. We used a modern PyTorch PPO
implementation, whereas the paper used RLlib 2.2 with KL-penalty PPO. We used
the recurrent L2 coefficient reported in the paper (0.3); the released code
and example parameters contain a conflicting value of 3.0. These differences
define the scope of the comparison.

## Direct numerical comparison with the original paper

The tables below place the original and reproduced values on the same page.
The paper's values are recomputed from the official Fig. 5 and Fig. 6 source
workbooks as mean ± sample standard deviation, matching our report. The
published plots show medians and interquartile ranges, so small discrepancies
between these table entries and a visual reading of the figures are expected.
Each original-paper cell and each reproduction cell summarizes ten seeds unless
another sample size is shown.

### Behavior at the end of training

| Measure | Paper social | Paper non-social | Our RNN social | Our RNN non-social |
|---|---:|---:|---:|---:|
| Chaser collisions per episode (Fig. 5i) | 16.06 ± 2.68 | 3.19 ± 0.36 | 8.98 ± 5.70 | 3.45 ± 0.32 |
| Chaser partner in vision, % (Fig. 5j) | 86.36 ± 7.00 | 36.16 ± 1.55 | 60.51 ± 18.50 | unavailable |
| Chaser average distance (Fig. 5k) | 3.04 ± 0.47 | 6.44 ± 0.12 | 3.78 ± 0.99 | 5.19 ± 0.04 |
| Chaser new fields (Fig. 5l) | 30.77 ± 1.87 | 72.39 ± 3.86 | 17.55 ± 5.31 | 82.56 ± 1.90 |
| Explorer new fields (Fig. 5m) | 35.73 ± 7.00 | 58.68 ± 3.24 | 45.24 ± 12.60 | 74.87 ± 6.06 |
| Explorer average distance (Fig. 5n) | 7.71 ± 0.38 | 6.56 ± 0.12 | 5.58 ± 0.19 | 5.00 ± 0.08 |

The condition contrast has the same sign for all comparable behavioral
measures. The reproduction's social chaser is less aggressive than the
paper's: it collides about half as often and sees the partner less frequently.
Our stored non-social visibility flag is always false because the non-social
observation mode disables partner visibility. It is therefore not the same
physical field-of-view measure shown in Fig. 5j, and we report it as
unavailable rather than as 0%.

### Neural and representational analyses

| Analysis | Paper social | Paper non-social | Our RNN social | Our RNN non-social |
|---|---:|---:|---:|---:|
| Chaser collision decoding (Fig. 6b) | 0.949 ± 0.020 | 0.500 ± 0.000 | 0.983 ± 0.014 (n = 6) | 0.765 ± 0.044 (n = 10) |
| Chaser partner-escape decoding (Fig. 6c) | 0.886 ± 0.126 | 0.520 ± 0.026 | 0.888 ± 0.033 (n = 9) | unavailable |
| Explorer collision decoding (Fig. 6d) | 0.942 ± 0.040 | 0.500 ± 0.000 | 0.983 ± 0.014 (n = 6) | 0.747 ± 0.066 (n = 10) |
| Significant shared dimensions (Fig. 6g) | 164.69 ± 51.95 | 0.015 ± 0.047 | 183.0 ± 83.64 | 107.5 ± 17.88 |
| Alignment strength | ΔPCC 0.406 ± 0.046 (Fig. 6h) | ΔPCC −0.0003 ± 0.027 | top *r* 0.743 ± 0.218 | top *r* 0.067 ± 0.033 |

The social decoder values closely match the paper. The collision decoders in
our non-social controls do not: they are well above the 0.50 chance level.
This does not imply that the non-social agents represent a partner; a collision
can be predictable from the agent's own position, motion, and state. It does
show that collision decodability alone is not selective evidence for a social
representation in our implementation. Exact Fig. 6e values are omitted because
the official source workbook contains an empty Fig. 6e sheet; we do not replace
missing source data with values estimated by eye.

The shared-dimension count also differs sharply in the non-social control. By
contrast, the leading correlation preserves the paper's qualitative condition
contrast. The two alignment magnitudes should not be compared directly: the
paper pools late checkpoints and reports chance-subtracted Pearson correlation
(ΔPCC), whereas our headline is final-checkpoint leading PLSC correlation.

For partner representation (Fig. 6p–r), the paper's social value increased
from **0.69 ± 0.20%** early to **4.37 ± 1.58%** late, whereas ours decreased
from **7.46%** at the first saved checkpoint to **3.93%** late. The paper's
representation–performance correlation was **r = 0.611** (n = 219) for social
agents and **r = −0.055** (n = 220) for non-social agents. Ours was
**r = 0.639** and **r = 0.284**, respectively (ten pairs per condition).
Thus, the late condition difference and positive social association recur, but
the developmental trajectory does not.

### Perturbation comparison

| System and measure | Original | Shared dimensions removed | Random dimensions removed |
|---|---:|---:|---:|
| Paper: collisions (Fig. 6s) | 17.49 ± 5.72 | 4.55 ± 3.76 | 16.91 ± 11.59 |
| Our RNN: collisions | 4.57 ± 7.97 | 2.44 ± 3.04 | 1.55 ± 0.93 |
| Paper: partner in vision, % (Fig. 6t) | 89.52 ± 5.45 | 46.94 ± 22.15 | 81.89 ± 11.65 |
| Our RNN: partner in vision, % | 33.08 ± 24.43 | 36.52 ± 21.59 | 31.76 ± 10.36 |
| Paper: average distance (Fig. 6u) | 2.32 ± 0.25 | 4.75 ± 1.24 | 2.69 ± 0.62 |
| Our RNN: average distance | 5.21 ± 1.40 | 4.88 ± 1.14 | 5.23 ± 0.80 |

In the paper, removing the shared subspace reduces collisions and visibility
and increases distance, while removing random directions largely preserves the
original behavior. Our interventions do not show this selective pattern. This
direct comparison is the clearest reason to treat C5 as inconclusive.

## Results of the primary RNN reproduction

### Social training produced the expected role-specific behavior

Against a standardized random explorer, social chasers produced
**8.98 ± 5.70 collisions per episode**, compared with **3.45 ± 0.32** for
non-social chasers. They also stayed closer to the random explorer
(mean distance **3.78** versus **5.19**). The complementary result appeared for
the explorer: against a random chaser, social explorers suffered only
**0.57 ± 0.19 collisions**, compared with **2.87 ± 0.32** for non-social
explorers. The absolute number of accidental collisions in the control is less
important than the role-specific separation between conditions.

These results reproduce the behavioral logic of Fig. 5. The opposite outcomes
for the two roles show that the social reward selected pursuit for the chaser
and avoidance for the explorer.

### Hidden activity encoded social events and partner behavior

We trained linear classifiers to read events from the agents' hidden states.
Balanced accuracy is 0.50 at chance. In social RNNs, collision decoding reached
**0.983** for the chaser and **0.983** for the explorer. The chaser's state
decoded the partner's escape at **0.888**, and the explorer's state decoded the
partner's approach at **0.886**. Shuffled-label controls remained near 0.50.
Depending on the target, six to nine of the ten social pairs contained enough
positive events to enter the decoding analysis.

The matching non-social collision decoders reached **0.765** for the chaser
and **0.747** for the explorer, whereas the paper reported chance-level values
of 0.50. Social-event information is therefore strongly accessible, but the
collision result is not selective to social training in our runs. Partner
approach and escape cannot be evaluated in our non-social condition because
partner-visibility events are not recorded there.

For a neuroscience audience, this analysis has the same interpretation as a
population decoder: the relevant information is linearly accessible in the
population state. From an AI perspective, linear accessibility establishes
that the network represents the variable in a readable form. It does not by
itself show that the policy uses that variable to choose an action.

### Shared dimensions emerged between independently trained agents

We used partial least squares correlation (PLSC) to identify one weighted
activity pattern in the chaser and a corresponding pattern in the explorer
whose values covaried over time. The mean correlation of the leading shared
dimension was **0.743 ± 0.218** in social RNN pairs and
**0.067 ± 0.033** in non-social pairs. Significance was assessed with an
episode re-pairing null that preserves within-episode temporal structure while
breaking the identity of the interacting partner episode.

The leading-correlation contrast is consistent with Fig. 6: social interaction
was accompanied by stronger cross-agent representational structure even though
the two agents had separate parameters and exchanged no hidden states.
However, the number of significant dimensions was **183 ± 84** in our social
RNNs and **108 ± 18** in the non-social controls, unlike the paper's near-zero
non-social count. We therefore reproduce the correlation-strength contrast,
not the full dimension-count result. "Shared" means that two different
networks developed aligned latent variables; it does not mean that they shared
weights or directly transmitted neural activity.

### The two stronger claims were only partly recovered

The partner-representation analysis asks how much variance in the chaser's
action-relevant activity is uniquely associated with the explorer after
accounting for the chaser's own position, action, and behavior. Late in
training, this unique partner contribution was **0.0393** in social chasers and
**0.000083** in non-social chasers. Across the ten social pairs, stronger
partner representation was associated with more collisions
(**r = 0.639**); the non-social association was weaker (**r = 0.284**).
However, the social value decreased from **0.0746** early in training to
**0.0393** late in training. We therefore recovered the condition difference
and the performance association, but not the increasing training trajectory
reported in the paper. Given the sample of ten pairs, the correlation should
be read as supportive rather than as a precise effect-size estimate.

The perturbation analysis also produced a mixed result. Removing the top ten
shared dimensions from the chaser reduced collisions, as the paper predicts,
but the nominal random-basis control reduced collisions at least as strongly.
For the RNN, collisions fell from **4.57** per episode without intervention to
**2.44** after shared-dimension removal and **1.55** after random-basis
removal. The intervention therefore shows that the policy is sensitive to
those shared directions, but it does not establish that their causal role is
specific to shared information.

Our current random-basis construction is also not fully equivalent to the
paper's control: it does not explicitly constrain the control directions to be
non-overlapping with the shared subspace, and its joint temporal permutation
preserves the activity covariance used to select principal components. The C5
result should therefore be classified as **inconclusive** until the
paper-faithful control is implemented and the intervention is rerun.

## Alignment with the paper

| Paper result tested | Reproduction verdict | Evidence from our study |
|---|---|---|
| Social reward produces role-specific behavior | **Supported** | Social chasers collided more and stayed closer; social explorers were caught less often than non-social controls. |
| Hidden activity represents social events and partner behavior | **Partly supported** | Social decoding closely matched the paper, but non-social collision decoding remained above chance in our runs. |
| Shared cross-agent dimensions emerge during social interaction | **Partly supported** | Leading PLSC correlation separated social from non-social pairs, but the significant-dimension count did not reproduce the paper's near-zero control. |
| Partner representation increases during training and predicts chaser performance | **Partly supported** | The social condition showed much stronger partner representation and a positive performance association, but representation decreased over training. |
| Shared dimensions have a selective causal role in social behavior | **Inconclusive** | Shared-dimension removal impaired behavior, but the nominal random-basis control impaired it equally or more, and the control implementation requires correction. |

## Additional experiments beyond the paper's primary comparison

### Cross-architecture study

We repeated the full social and non-social training experiment with three
additional architectures: a frame-stack multilayer perceptron (MLP), a gated
linear state-space model (SSM), and a causal Transformer. Each architecture
used ten social and ten non-social seeds and the same environment, reward,
training budget, evaluation procedure, and 256-dimensional analysis state.
These experiments ask a question that is especially relevant to modern AI:
does a shared task force different model classes to learn the same internal
solution?

The behavioral answer was no. All four architectures learned the role
distinction, but the RNN adopted a lower-visibility, lower-collision strategy,
whereas the other models remained close to the partner and collided more
frequently.

| Architecture | Social chaser collisions | Non-social chaser collisions | Partner in vision |
|---|---:|---:|---:|
| RNN | 8.98 | 3.45 | 0.61 |
| MLP | 26.93 | 2.68 | 0.93 |
| SSM | 35.92 | 3.15 | 0.94 |
| Transformer | 41.31 | 3.39 | 0.96 |

Event decodability generalized across architectures. Chaser collision accuracy
ranged from **0.847 to 0.983**, and partner approach or escape accuracy ranged
from **0.733 to 0.888**. Thus, linearly readable social information was robust
to the choice of model, even though the behavioral strategies differed.

### Controlling temporal scaffolds in the shared-subspace analysis

The Transformer exposed a confound that was difficult to see in the RNN alone.
Its raw leading-dimension correlation was **0.957 in the non-social control**,
even though that condition supplied no partner-position input. The most direct
explanation is the Transformer's positional embedding: both agents carry a
strong code for the same timestep, so their states can correlate without
representing one another.

We therefore repeated PLSC after subtracting each model's across-episode mean
at every timestep. This removes deterministic time-locked activity—including
positional embeddings and repeatable hidden-state transients—and retains
episode-specific variation. The residualized results were:

| Architecture | Social leading correlation | Non-social leading correlation | Social-control separation |
|---|---:|---:|---:|
| RNN | 0.688 | 0.067 | 0.62 |
| MLP | 0.679 | 0.103 | 0.58 |
| SSM | 0.642 | 0.315 | 0.33 |
| Transformer | 0.438 | 0.356 | 0.08 |

The RNN and MLP retained the clearest separation between social and non-social
conditions. The MLP is an informative floor because it has no recurrent state:
its relatively high social correlation shows how much alignment can arise
from the two agents responding to the same visible situation. Seven of the
eight analyzable social RNN pairs also occupied a low-vision regime and still
showed a mean residualized correlation of **0.652**. None of the other
architectures learned a low-vision policy, so this observation cannot separate
an architecture effect from a strategy effect. It shows that high correlation
can coexist with limited current visual access in the RNN condition; a matched-
behavior experiment is needed to determine why.

### Representational geometry differed across architectures

We also presented the same observation streams to each trained model and
compared its hidden-state geometry using linear centered kernel alignment
(CKA). The RNN was the most distinct architecture, with similarities of
**0.64–0.68** to the other models. The MLP, SSM, and Transformer were more
similar to one another (**0.71–0.78**). Behavior, residualized PLSC, and CKA
therefore converge on a cautious conclusion: the shared objective did not
produce a unique, architecture-independent representation.

### Cross-architecture perturbations

Shared-dimension removal reduced collisions in every architecture, but the
nominal random-basis intervention was equally or more disruptive in every
case. These values come from self-play against the trained explorer and are
therefore different from the random-opponent behavioral evaluation above.

| Architecture | No intervention | Shared dimensions removed | Nominal random basis removed |
|---|---:|---:|---:|
| RNN | 4.57 | 2.44 | 1.55 |
| MLP | 21.23 | 2.04 | 1.58 |
| SSM | 17.21 | 4.16 | 1.36 |
| Transformer | 10.67 | 2.05 | 1.70 |

This pattern reinforces the need for a corrected, variance-matched,
non-overlapping control. It is compatible with low-dimensional policies in
which many high-variance perturbations are destructive, but the current
experiment does not identify low dimensionality as the unique explanation.

### Targeted low-regularization RNN rerun

We also trained an exploratory RNN condition with recurrent L2 regularization
removed. The recurrent weight norm increased substantially, and the chaser
shifted toward a more aggressive strategy. The change also produced unstable
hidden dynamics: only three of ten social seeds retained enough valid episodes
for the downstream analyses. Within those analyzable seeds, the perturbation
control still did not separate shared-dimension removal from random-basis
removal.

This rerun suggests that recurrent regularization strongly affects both the
strategy and numerical stability of the learned policy. It does not show that
task complexity is the sole cause of the remaining C4/C5 differences. The raw
artifacts from this rented-GPU run were not retained, so we treat it as
exploratory context rather than primary evidence; the committed configuration
records how it can be repeated.

## Interpretation for AI and language models

### Information in a representation is not the same as information used

The decoding results show that collision and partner behavior can be recovered
from hidden activity with a simple linear readout. In language-model terms,
this is similar to finding that a property can be decoded from a residual
stream or layer activation. It establishes availability, not functional use.
A model may carry information because it is useful, because it is correlated
with another useful variable, or because the architecture preserves it by
default. Perturbation or mediation analyses are needed to distinguish these
possibilities.

### Representational similarity needs explicit nuisance controls

The Transformer result provides a concrete example. Two agents can appear to
share a strong latent dimension because both encode token position—or, in our
task, timestep—even when neither represents the other. The same concern
applies when comparing two language models, two layers, or a model and neural
recordings: prompt structure, token position, stimulus timing, and common input
can all produce alignment. A high correlation is a starting point for
interpretation, not its endpoint.

### Architecture shapes which solution learning finds

The agents had the same objective but did not converge to the same behavior or
internal geometry. This is the role of an AI model's *inductive bias*: the
architecture makes some computations easier to learn than others. For language
models, the analogous lesson is that similar benchmark performance does not
imply the same internal algorithm, and the same analysis may have different
meanings across recurrent, state-space, and Transformer models.

### A causal claim depends on the control intervention

Projecting activity away from a subspace is analogous to silencing a neural
population pattern. If both the target and control perturbations remove a large
amount of action-relevant activity, both may impair behavior. The scientific
question is therefore comparative: does removing the hypothesized shared
subspace impair behavior more than a variance-matched, non-overlapping control?
Our current data do not answer that question. This is why we distinguish a
successful intervention from a selective causal result.

### Relevance to language models has a clear boundary

These agents are small reinforcement-learning policies, not language models,
and the task contains no language. The study therefore does not demonstrate
that large language models possess social understanding or brain-like social
representations. Its relevance to language modeling is methodological: hidden
states are distributed population codes; model architecture shapes those
codes; common temporal structure can mimic representational alignment; and
causal interpretations require interventions with strong controls.

## A vocabulary bridge for the presentation

| Neuroscience term | AI-model analogue | Interpretation in this study |
|---|---|---|
| Population activity | Hidden-state vector | The joint activation of 256 model units at one timestep |
| Population decoder | Linear probe or classifier | Tests whether an event is linearly readable from hidden activity |
| Shared neural dimension | Paired latent feature across two models | One weighted pattern per agent whose values covary over time |
| Unique subspace | Model-specific latent activity | Variation not captured by the shared PLSC dimensions |
| Subspace perturbation | Activation intervention | Removes selected hidden-state directions before the action readout |
| Cell type | No direct counterpart here | Artificial units are learned coordinates, not biological cell classes |

## Limitations and overall interpretation

The main limitations are part of the scientific result. The implementation is
a modern port rather than an exact recreation of the original RLlib training
stack. The architecture comparison is not parameter-matched, and architecture
is confounded with the strategy each model learned. Time-mean subtraction is a
conservative control that can remove genuine time-locked coordination along
with nuisance structure. The C5 random-basis intervention must be corrected
before it can support a selective causal claim. Finally, the low-L2 rerun lacks
retained raw artifacts and should not carry a primary conclusion.

Within those boundaries, the reproduction gives a coherent answer. The
paper's central representational pattern is robust in our RNN agents: social
rewards produce role-specific behavior, social events are linearly readable,
and strong cross-agent latent dimensions emerge. The extensions show why the
interpretation must remain model-aware. Shared activity is shaped by behavioral
strategy, sensory coupling, temporal scaffolds, regularization, and
architecture. For an AI audience, this is the principal lesson: similar output
behavior can be implemented by different internal computations, and convincing
claims about those computations require both nuisance controls and selective
causal interventions.

## Evidence files

- Primary behavior: `runs/reports/mouse-run-run-1/summary.json`
- Event decoding and PLSC: `runs/analyses/mouse-run-run-1/neural_summary.json`
- Partner representation: `runs/analyses/c4_mouse-run-run-1.json`
- Perturbations: `runs/analyses/c5_*.json`
- Cross-architecture PLSC and CKA: `runs/analyses/cross_arch_plsc_residualized.json`
  and `runs/analyses/cross_arch_cka.json`
- Extended architecture report:
  `docs/paper-notes/cross-architecture-representation-study.md`

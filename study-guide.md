# Study guide — machine learning as estimation, geometry, and control

These chapters are intended for a mathematically mature reader. Linear algebra, multivariable calculus, probability, and basic programming are assumed; each chapter introduces its model-specific notation. Follow the playlist order on a first pass, then use the connections below for a second, more synthetic reading.

## A route through the material

| Stage | Chapters | Central question | Check your understanding |
|---|---|---|---|
| Statistical estimation | [1](lecture-01.html), [2](lecture-02.html), [3](lecture-03.html), [4](lecture-04.html), [5](lecture-05.html) | What distribution or decision rule does the objective estimate? | Derive squared-loss and logistic gradients; distinguish conditional from joint modeling. |
| Generalization and computation | [6](lecture-06.html), [7](lecture-07.html), [8](lecture-08.html) | Which errors arise from data, capacity, optimization, and implementation? | Keep validation independent and calculate a backpropagation step by hand. |
| Latent variables and representations | [9](lecture-09.html), [10](lecture-10.html), [11](lecture-11.html), [12](lecture-12.html), [13](lecture-13.html) | How do hidden structure and learned geometry support prediction or generation? | Derive the EM bound, PCA projection, diffusion forward marginal, and contrastive loss. |
| Sequence models and adaptation | [14](lecture-14.html), [16](lecture-16.html) | How is a conditional token distribution represented and adapted? | Track tensor shapes, causal masks, cache bytes, and response-only supervision. |
| Decision-making and learning from outcomes | [18](lecture-18.html), [20](lecture-20.html) | How do actions affect future data and reward? | Derive the Bellman equation, score-function gradient, and baseline identity. |

Lecture numbers 15, 17, and 19 are absent from the supplied playlist. The chapter numbering preserves that fact.

## Connection 1: conditional expectations recur everywhere

Under squared loss, the population-optimal predictor is $f^*(x)=\mathbb E[Y\mid X=x]$. A value function is also a conditional expectation, this time of future return. In diffusion denoising, squared-error training estimates a conditional noise expectation. The targets differ, but the orthogonal-projection interpretation is shared.

**Independent derivation.** For any square-integrable $Y$ and predictor $f(X)$, expand $Y-f(X)=[Y-\mathbb E(Y\mid X)]+[\mathbb E(Y\mid X)-f(X)]$. The cross term has expectation zero because the first bracket has conditional mean zero. Therefore

$$\mathbb E[(Y-f(X))^2]=\mathbb E[\operatorname{Var}(Y\mid X)]+\mathbb E[(\mathbb E[Y\mid X]-f(X))^2].$$

This separates irreducible conditional variation from approximation error. It does not, by itself, account for finite-sample estimation or optimization error.

## Connection 2: normalization creates probabilities and geometry

The exponential family's log-partition function normalizes a density. Softmax normalizes class scores, attention weights, and contrastive candidate scores. Derivatives of a log-sum-exp produce normalized weights; second derivatives produce a covariance structure. Recognizing this pattern makes several apparently unrelated gradient calculations familiar.

Normalization also changes meaning. An attention row sums to one across positions; a contrastive row sums to one across candidates; a vocabulary softmax sums to one across token types. Their axes and events are different, so their numerical values cannot be interpreted interchangeably.

## Connection 3: latent-variable learning is alternating inference and fitting

EM alternates computing a posterior over latent assignments and fitting parameters to weighted sufficient statistics. Diffusion uses a constructed noising process to make particular conditional distributions tractable. Both rely on carefully chosen auxiliary distributions, but they do not have the same optimization algorithm or the same guarantees.

When reading a new generative-model paper, identify which distributions are defined, which are learned, which can be sampled, and which can be evaluated pointwise. Many confusing derivations become clear once these four roles are separated.

## Connection 4: the objective does not define the entire system

Two models with the same loss can differ because of data, architecture, initialization, optimization, inference, and evaluation. Two agent systems with identical model weights can differ because of retrieval, tools, verification, and search. The retrieval and reinforcement-learning chapters therefore expand the unit of study from a predictor to a procedure that acquires information and acts.

## Three original research projects

1. **A controlled representation experiment.** Generate data with two independent factors, train a classifier on only one factor, then compare linear probing and fine-tuning on the other. Vary how much information the representation preserves. Distinguish insufficient representation from insufficient downstream labels.
2. **A numerically checked policy-gradient lab.** Enumerate all trajectories of a tiny finite MDP and compute the exact gradient. Compare Monte Carlo estimators with full return, reward-to-go, constant baseline, state baseline, and own-sample group centering. Measure bias and variance separately.
3. **Retrieval with an explicit bottleneck analysis.** Build a small fixed document corpus and questions with known supporting passages. Compare lexical, embedding, and hybrid retrieval; then compare generated answers using retrieved versus oracle evidence. Measure retrieval recall and claim support independently.

These are proposed experiments, not results claimed by the lectures. Begin with small deterministic cases and preserve a final test set before tuning any method.

## Practical study procedure

For each chapter, derive one displayed equation without looking, reproduce one numerical example, and answer the five exercises. Follow a timestamp link when the motivation is unclear. Read the linked primary paper after you can explain which problem the lecture's method solves and under which assumptions. The recording is authoritative for what was taught; primary research is authoritative for the attributed research method; the authored notes make their own derivations and corrections explicit.

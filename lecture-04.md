# Lecture 4 — Exponential families, generalized linear models, and softmax

A property service predicts a sale price, the probability of a defect, the number of maintenance requests, and one of several property types. Writing an unrelated learning algorithm for each output would hide their common structure. Exponential families explain why different observation models can share a gradient rule while retaining different means, variances, and support. Softmax then turns the same construction into the multiclass output layer used throughout modern machine learning.

## 1. Normalization organizes an exponential family [02:30](https://www.youtube.com/watch?v=8gVi4Rk21Eg&t=150s)

### Define the distribution before the predictor

For now consider one observation $Y$, without input features. Its possible values form a support set $\mathcal Y$. A density or probability mass belongs to the chosen exponential-family representation when it can be written

$$p_\eta(y)=b(y)\exp\left[\eta^\top T(y)-A(\eta)\right].$$

Here $T(y)\in\mathbb R^m$ is a vector of statistics, $\eta\in\mathbb R^m$ is the **natural parameter**, and $b(y)\ge0$ is a fixed base factor independent of $\eta$. The scalar $A(\eta)$ is the **log-partition function**. The inner product requires matching dimensions, and its exponent must be dimensionless. The representation is a statement about a family of distributions; a generalized linear model will later relate $\eta$ to property features.

Let $\nu$ be the underlying counting measure for discrete outcomes or an appropriate continuous measure for densities. Normalization requires $\int_{\mathcal Y}p_\eta(y)\,d\nu(y)=1$, so

$$A(\eta)=\log\int_{\mathcal Y}b(y)e^{\eta^\top T(y)}\,d\nu(y).$$

The **natural parameter domain** consists of those $\eta$ for which this integral is finite. It is not automatically all of $\mathbb R^m$. For example, an exponential distribution on nonnegative values has a negative natural parameter when its statistic is $T(y)=y$. Allowing a positive parameter would make the integral diverge.

The log-partition function depends on the parameters, even though it does not depend on the realized observation. Consequently, it cannot generally be discarded during parameter estimation. Dropping it would reward increasing the unnormalized score without accounting for how all other outcomes also gain mass. This is the normalization term that supplies both useful moments and curvature.

### What sufficient statistics preserve

For independent observations sharing $\eta$, the joint likelihood factors as $[\prod_i b(y_i)]\exp[\eta^\top\sum_iT(y_i)-nA(\eta)]$. All parameter-dependent interaction with the data passes through $\sum_iT(y_i)$. This is the relevant **sufficiency** statement: once that aggregate statistic is known, the remaining data factor does not alter relative likelihoods of parameter values within this family.

Sufficiency is relative to the chosen model. If the service models maintenance counts as Poisson, the sum of counts is sufficient for a shared rate. That does not imply that timing patterns or correlations are irrelevant to a richer model. A compressed statistic can be sufficient under an inaccurate family while discarding evidence that the family is inaccurate.

The representation is also not necessarily unique. Redundant statistics can create parameter directions that leave the distribution unchanged. Later, softmax will provide a concrete example: adding the same constant to every class score changes no probability. Convexity and identifiability therefore need separate examination.

## 2. Familiar distributions reveal the common pattern [07:00](https://www.youtube.com/watch?v=8gVi4Rk21Eg&t=420s)

### Bernoulli: the natural parameter is log odds

For a binary defect label $y\in\{0,1\}$ with probability $q\in(0,1)$, write the mass as $q^y(1-q)^{1-y}$. Taking its logarithm and collecting the term multiplying $y$ gives $y\log[q/(1-q)]+\log(1-q)$. Thus $T(y)=y$, $b(y)=1$, $\eta=\log[q/(1-q)]$, and $A(\eta)=\log(1+e^\eta)$.

Solving the natural-parameter equation yields $q=e^\eta/(1+e^\eta)$. At $\eta=0$, both outcomes have equal probability. At $\eta=\log 3$, the defect probability is $3/4$. Finite natural parameters represent probabilities strictly inside $(0,1)$; deterministic endpoints occur only as limits. The domain and boundary behavior matter when a fitted classifier becomes extremely confident.

### Gaussian: fixed variance belongs in the bookkeeping

For a real-valued price with mean $\mu$ and known variance $s^2\gt0$, expand the Gaussian exponent:

$$-\frac{(y-\mu)^2}{2s^2}
=-\frac{y^2}{2s^2}+\frac{\mu y}{s^2}-\frac{\mu^2}{2s^2}.$$

One valid representation uses $T(y)=y$, $\eta=\mu/s^2$, $b(y)=\exp[-y^2/(2s^2)]/\sqrt{2\pi s^2}$, and $A(\eta)=s^2\eta^2/2$. The familiar identity $\eta=\mu$ is the special case $s^2=1$. When variance is also unknown, the statistic must include both $y$ and $y^2$, and the natural parameter becomes two-dimensional. A parameter being held fixed is not the same as a parameter disappearing from the model.

This distinction also prevents conflating a scalar Gaussian response with a multivariate Gaussian over many features. The family representation can handle both, but the statistic dimensions and covariance parameters change. In the running case, we are currently modeling one output conditional on a property, not the full distribution of property features.

### Poisson: counts have a mean-variance relationship

For a nonnegative integer count $y$ and positive rate $\lambda$, the Poisson mass is $e^{-\lambda}\lambda^y/y!$. Rearranging gives $T(y)=y$, $\eta=\log\lambda$, $b(y)=1/y!$, and $A(\eta)=e^\eta$. The natural parameter may be any real number, while the rate remains positive.

This gives the property service a coherent count model: the output cannot be a negative integer, and the fitted mean can vary continuously. But Poisson also imposes variance equal to mean. If maintenance requests cluster because storms affect many properties simultaneously, the observed variability may substantially exceed the mean. A different family or a latent-rate model may then be appropriate. The output type narrows plausible families; it does not uniquely choose one.

| Output in the running case | Statistic | Natural parameter | Log partition | Conditional mean |
|---|---|---|---|---|
| Defect indicator | $y$ | $\log[q/(1-q)]$ | $\log(1+e^\eta)$ | sigmoid of $\eta$ |
| Price, fixed variance $s^2$ | $y$ | $\mu/s^2$ | $s^2\eta^2/2$ | $s^2\eta$ |
| Maintenance count | $y$ | $\log\lambda$ | $e^\eta$ | $e^\eta$ |

The table is a map among parameterizations, not a declaration that all three observations have additive Gaussian noise. Bernoulli and Poisson variation are defined by their own conditional distributions.

## 3. Derivatives of the log partition produce moments [15:00](https://www.youtube.com/watch?v=8gVi4Rk21Eg&t=900s)

### The first derivative is an expectation

Assume the support is independent of $\eta$ and differentiation can pass through the normalization integral. These regularity conditions hold for the elementary examples above at interior natural parameters. Differentiate the logarithm using the chain rule:

$$\nabla A(\eta)=
\frac{\int T(y)b(y)e^{\eta^\top T(y)}\,d\nu(y)}
{\int b(y)e^{\eta^\top T(y)}\,d\nu(y)}
=\mathbb E_\eta[T(Y)].$$

The numerator weights each statistic by its unnormalized mass. Dividing by the partition function converts those weights into probabilities. The result is the mean of the statistic, not necessarily the mean of the raw observation. They coincide when $T(y)=y$.

For Bernoulli, $A'(\eta)=e^\eta/(1+e^\eta)=q$. For the fixed-variance Gaussian, $A'(\eta)=s^2\eta=\mu$. For Poisson, $A'(\eta)=e^\eta=\lambda$. The same derivative mechanism recovers three different mean maps. It is a unifying identity, not a claim that every partition function is computationally cheap to evaluate.

### The second derivative is a covariance

The derivative of the log mass is $\nabla_\eta\log p_\eta(y)=T(y)-\nabla A(\eta)$. Differentiating the expectation of $T$ therefore gives

$$\nabla^2A(\eta)
=\mathbb E_\eta[T(Y)T(Y)^\top]
-\mathbb E_\eta[T(Y)]\mathbb E_\eta[T(Y)]^\top
=\operatorname{Cov}_\eta(T(Y)).$$

For any vector $v$, multiplying this matrix on both sides gives $\operatorname{Var}(v^\top T(Y))\ge0$. The Hessian is positive semidefinite, so $A$ is convex on its natural domain. It is strictly convex only when no nonzero parameter direction corresponds to an almost-surely constant linear combination of the statistics. Redundancy produces flat directions.

For a defect probability $q=3/4$, the curvature is $q(1-q)=3/16$. For a Poisson rate 4, the curvature is 4. For the fixed-variance Gaussian it is $s^2$, constant in the natural parameter. These are useful numerical checks because a wrong sign or a missing scale factor immediately breaks the variance identity.

The **Fisher information** is the expected outer product of the score $\nabla_\eta\log p_\eta(Y)$. The score expression above shows that Fisher information equals this same covariance. Thus local distinguishability of distributions, fluctuations of their sufficient statistics, and curvature of normalization are the same object in natural coordinates. This connection is developed systematically in [Wainwright and Jordan's treatment of exponential families](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf).

### Cumulants require a shift

The moment-generating function of $T(Y)$ under a fixed parameter $\eta$ is $M_\eta(t)=\mathbb E_\eta[e^{t^\top T(Y)}]$, when finite. Substituting the family density gives $M_\eta(t)=\exp[A(\eta+t)-A(\eta)]$. Its logarithm, the **cumulant-generating function**, is therefore $K_\eta(t)=A(\eta+t)-A(\eta)$.

This shifted difference, rather than $A$ by itself, is the precise cumulant-generating function at the chosen distribution. Its first derivative at zero is the mean and its second is the covariance. Higher cumulants are not generally raw moments: the third scalar cumulant is the third central moment, while the fourth subtracts three times the squared variance from the fourth central moment. Keeping moments, central moments, and cumulants distinct prevents an appealing shorthand from becoming a false identity.

## 4. A generalized linear model connects features to a distribution [28:00](https://www.youtube.com/watch?v=8gVi4Rk21Eg&t=1680s)

### Three parameter objects play different roles

Let $x_i\in\mathbb R^p$ be property features including an intercept, and let $\theta\in\mathbb R^p$ be shared fitted coefficients. In the scalar canonical construction, set the observation's natural parameter to $\eta_i=x_i^\top\theta$. Its mean statistic is $\mu_i=A'(\eta_i)$. The coefficient vector, the input-specific natural parameter, and the mean are three different objects.

For defects, the mean is a probability. For maintenance counts, the mean is a positive rate. The canonical **link function** maps the mean back to the linear natural parameter: $g(\mu_i)=\eta_i$. The inverse link maps a score to the mean. The term “generalized linear model” is broader than this canonical-link construction, but the canonical case is the one that yields the particularly simple gradient used here.

For conditionally independent observations, the averaged negative log likelihood is

$$J(\theta)=\frac1n\sum_i
\left[A(x_i^\top\theta)-T(y_i)x_i^\top\theta-\log b(y_i)\right].$$

When $T(y)=y$, differentiating gives $\nabla J=n^{-1}\sum_i(\mu_i-y_i)x_i$. Differentiating again gives $\nabla^2J=n^{-1}\sum_iA''(\eta_i)x_ix_i^\top$. The observation model determines the mean response and variance; the outer computational structure remains prediction error times features.

This formula assumes a scalar statistic and a canonical natural-parameter link. It is not a universal update for every distribution under every parameterization. A vector statistic requires a matching matrix of coefficients, and a noncanonical link adds chain-rule factors. Stating those restrictions is more useful than memorizing a rule whose apparent universality fails when a model changes.

### A Poisson update for maintenance requests

Suppose three equally exposed properties have counts 0, 2, and 4, and use only an intercept. The mean rate is $\lambda=e^\theta$. At $\theta=0$, the model predicts rate 1 for every property. The averaged gradient is $1-(0+2+4)/3=-1$. A descent step of size $0.1$ increases $\theta$ to $0.1$ and the rate to $e^{0.1}\approx1.10517$.

The exact maximum-likelihood rate is the sample mean 2, so the optimum is $\theta=\log2$. To derive it, the coefficient-dependent loss is $e^\theta-2\theta$; its derivative vanishes at $e^\theta=2$, and its positive second derivative establishes the unique minimum. If all observed counts were zero, the rate would instead approach zero as $\theta\to-\infty$, illustrating another boundary optimum.

If properties have different observation durations, the expected count should scale with exposure. An exposure $e_i\gt0$, measured in a consistent time unit, can enter as an offset $\eta_i=\log e_i+x_i^\top\theta$. The coefficient on the log exposure is fixed at one. Ignoring exposure would confuse a long observation window with an intrinsically high maintenance rate.

## 5. Softmax is a coupled categorical model [36:00](https://www.youtube.com/watch?v=8gVi4Rk21Eg&t=2160s)

### Scores compete through one normalization

Suppose a property belongs to exactly one of $K$ categories. Represent label $j$ by a **one-hot vector** $e_j\in\mathbb R^K$, with a one in coordinate $j$ and zeros elsewhere. Give each class a coefficient vector $\theta_j\in\mathbb R^p$, and define its score $z_j=\theta_j^\top x$. Stack the scores into $z\in\mathbb R^K$. The categorical log partition is $A(z)=\log\sum_k e^{z_k}$, giving

$$p_j=\frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}.$$

These positive probabilities sum to one and lie in the **probability simplex**, the set of nonnegative vectors whose coordinates sum to one. The model expresses one mutually exclusive category. If a document may simultaneously concern plumbing and electrical issues, independent binary labels or a richer joint model may be appropriate instead. A softmax would incorrectly force those labels to compete for a single unit of probability mass.

Take scores $(\log2,0,0)$ for three property categories. Exponentiating gives $(2,1,1)$ and normalizing gives $(1/2,1/4,1/4)$. If the true category is the first, negative log likelihood is $\log2$. If it is the second, loss is $\log4$. The probabilities depend on every class score, not only the selected class's score.

### Invariance reveals the actual decision boundary

For any scalar $c$, softmax of $z+c\mathbf1$ equals softmax of $z$, because the common factor $e^c$ cancels. Here $\mathbf1$ is the all-ones vector. Similarly, adding the same coefficient vector to every $\theta_j$ changes no probabilities. One can remove this redundancy by fixing a reference class's coefficients to zero or imposing a sum-to-zero constraint.

The boundary between classes $j$ and $k$ is where their probabilities, equivalently their scores, are equal: $(\theta_j-\theta_k)^\top x=0$. An individual zero-score hyperplane $\theta_j^\top x=0$ is generally not a multiclass decision boundary. This distinction clarifies the geometric drawings in the recording: class-specific score contours and boundaries between winning classes are different objects.

For two classes, divide numerator and denominator of $p_1$ by $e^{z_1}$ to obtain $p_1=1/[1+e^{-(z_1-z_2)}]$. Binary logistic regression is therefore the two-class case with effective coefficients $\theta_1-\theta_2$. The identity explains why only the difference is identifiable.

### Derive the training gradient and its null direction

Let $q\in\mathbb R^K$ be a target probability vector, initially the one-hot observed label. Cross entropy is $L=-\sum_jq_j\log p_j=A(z)-q^\top z$, since $\sum_jq_j=1$. Differentiating gives $\nabla_zL=p-q$. The gradient with respect to class coefficients is $(p_j-q_j)x$.

The Hessian in score coordinates is $\operatorname{diag}(p)-pp^\top$, where $\operatorname{diag}(p)$ is the diagonal matrix with entries $p_j$. It is the covariance of a one-hot categorical variable. Multiplying by $\mathbf1$ gives zero, exactly matching common-shift invariance. The sum of gradient entries also vanishes: probability mass is redistributed among classes rather than created or destroyed.

For prediction $(1/2,1/4,1/4)$ and target $(1,0,0)$, the score gradient is $(-1/2,1/4,1/4)$. A descent step raises the correct class score and lowers both alternatives. Although the one-hot loss contains only $-\log p_1$, the normalization ensures that all class coefficients receive a gradient.

### Stable code and independent checks

Subtract the largest score before exponentiating. Shift invariance preserves the exact mathematical probabilities, while all exponentials become at most one. Compute log probabilities directly from shifted scores rather than taking the logarithm of rounded probabilities; a very unlikely class can otherwise underflow to zero.

```python
import numpy as np


def categorical_loss_gradient(scores, target):
    scores = np.asarray(scores, dtype=float)
    target = np.asarray(target, dtype=float)
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("at least two class scores are required")
    if target.shape != scores.shape:
        raise ValueError("target and score shapes must match")
    if not np.isfinite(scores).all() or not np.isfinite(target).all():
        raise ValueError("inputs must be finite")
    if (target < 0).any() or not np.isclose(target.sum(), 1):
        raise ValueError("target must be a probability vector")
    with np.errstate(over="raise", invalid="raise"):
        shifted = scores - scores.max()
        log_p = shifted - np.log(np.exp(shifted).sum())
        probabilities = np.exp(log_p)
        loss = -target @ log_p
    return float(loss), probabilities - target, probabilities


scores = np.array([np.log(2), 0.0, 0.0])
target = np.array([1.0, 0.0, 0.0])
loss, gradient, probabilities = categorical_loss_gradient(scores, target)
assert np.allclose(probabilities, [0.5, 0.25, 0.25])
assert np.isclose(loss, np.log(2))
assert np.allclose(gradient, [-0.5, 0.25, 0.25])
shifted_result = categorical_loss_gradient(scores + 1000, target)
assert np.allclose(shifted_result[2], probabilities)
assert np.isclose(gradient.sum(), 0)
```

The function returns a scalar loss, its score gradient, and the probability vector. It has no file or network effects and does not mutate its inputs. It rejects invalid target distributions and nonfinite arrays. The large common-shift check would fail in a naive implementation that exponentiates the raw scores. Extremely large finite score differences can still exceed floating-point range; the arithmetic error is surfaced rather than silently treated as a valid result.

## 6. Label smoothing changes the target distribution [62:00](https://www.youtube.com/watch?v=8gVi4Rk21Eg&t=3720s)

### A regularized target has a finite preferred confidence

With a hard target $e_y$, reducing cross entropy for an isolated example drives its predicted class probability toward one. **Label smoothing** replaces that target by $q=(1-\epsilon)e_y+\epsilon u$, where $\epsilon\in[0,1]$ is a chosen smoothing amount and $u$ is uniform over the $K$ classes. Under this convention the correct-class target is $1-\epsilon+\epsilon/K$ and every other target is $\epsilon/K$.

Another convention assigns $\delta$ to each incorrect class and $1-(K-1)\delta$ to the correct class. The conventions agree when $\delta=\epsilon/K$, but using the same numerical symbol in both formulas would give different targets. The three-class picture in the recording uses the per-incorrect-class form. An implementation must state which convention its parameter controls.

For $K=3$ and $\epsilon=0.15$, the target is $(0.9,0.05,0.05)$. At prediction $(0.5,0.25,0.25)$ the gradient becomes $(-0.4,0.2,0.2)$. At prediction $(0.99,0.005,0.005)$, the correct-class gradient is positive, so descent reduces its excessive confidence relative to the smoothed target. This is a change to the objective, not merely a numerically stabilized version of the original likelihood.

To see the preferred distribution, write cross entropy as $H(q,p)=H(q)+D_{\rm KL}(q\|p)$. Here $H(q)=-\sum_jq_j\log q_j$ is entropy and $D_{\rm KL}(q\|p)=\sum_jq_j\log(q_j/p_j)$ is Kullback–Leibler divergence. The latter is nonnegative and vanishes when the distributions agree. For an unconstrained single-example prediction, the minimum is therefore $p=q$. If every $q_j$ is positive, finite relative logits can attain that target.

[Müller, Kornblith, and Hinton](https://arxiv.org/abs/1906.02629) studied label smoothing's effects on generalization, calibration, and learned representations. Their experiments also found disadvantages for some knowledge-distillation settings. These findings support treating smoothing as an empirically testable modeling choice, not a universal improvement. Uniform smoothing does not encode which two classes are semantically similar and does not by itself estimate the actual annotation-error process.

```python
epsilon = 0.15
smoothed = (1 - epsilon) * target + epsilon / len(target)
assert np.allclose(smoothed, [0.9, 0.05, 0.05])
_, soft_gradient, _ = categorical_loss_gradient(scores, smoothed)
assert np.allclose(soft_gradient, [-0.4, 0.2, 0.2])
_, optimum_gradient, optimum_p = categorical_loss_gradient(
    np.log(smoothed), smoothed
)
assert np.allclose(optimum_p, smoothed)
assert np.allclose(optimum_gradient, 0)
```

This continuation checks the target convention and the claimed unconstrained optimum. It does not show that a shared linear model can fit every smoothed target simultaneously. Conflicting examples and restricted features can make the training optimum a compromise.

## Research extension: the same curvature appears in statistical mechanics

For discrete physical states with energies $E_j$, a Gibbs distribution assigns probability proportional to $e^{-\beta E_j}$, where $\beta$ has inverse-energy units. Its partition function is $Z(\beta)=\sum_j e^{-\beta E_j}$. This is the same normalization as softmax with scores $z_j=-\beta E_j$. Differentiation gives $d\log Z/d\beta=-\mathbb E[E]$ and $d^2\log Z/d\beta^2=\operatorname{Var}(E)$.

Thus a response derivative is a fluctuation: the sensitivity of mean energy to inverse temperature is minus its variance. This is the scalar counterpart of the covariance Hessian already derived. [Jaynes's original information-theoretic treatment](https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620) provides a primary bridge between maximum entropy and statistical mechanics. The analogy does not imply that learned logits are physical energies or that a model's sampling temperature is a thermodynamic temperature. The common structure is constrained normalization and its derivatives.

For the property service, this viewpoint suggests a useful diagnostic: directions with nearly zero categorical variance carry little local information and make curvature-based estimation difficult. Redundant classes or saturated probabilities can therefore create numerical problems even when the normalization formula is exact.

## Limitations, exercises, and worked solutions

The derivative identities require regularity and an interior natural parameter. Exact partition functions can be expensive in large structured models. Canonical-link convexity concerns fixed features, not an arbitrary neural representation. Softmax assumes mutually exclusive outcomes, and smoothing changes the target rather than certifying label quality. These are boundaries of the construction, not implementation details to be hidden by an automatic-differentiation library.

1. **Derive.** Write an exponential distribution with rate $\lambda\gt0$ on $y\ge0$ in exponential-family form. Find its natural domain, mean, and variance from $A$.
2. **Compute.** For Poisson counts 0, 2, and 4 with only an intercept, find the maximum-likelihood natural parameter and Hessian there.
3. **Counterexample.** Why is $\theta_j^\top x=0$ generally not the boundary between softmax classes $j$ and $k$? Construct two scores showing this.
4. **Derive.** Show that adding a common constant to scores leaves probabilities and cross entropy unchanged, and connect that invariance to the Hessian null vector.
5. **Design.** A document can discuss both plumbing and electrical defects. Choose a target representation and loss, and explain why neither softmax nor uniform label smoothing automatically solves the problem.

<details><summary>Solutions</summary>

1. The density is $\lambda e^{-\lambda y}$ for $y\ge0$. Choose $T(y)=y$, $b(y)=1$ on that support, and $\eta=-\lambda\lt0$. The normalization integral is $\int_0^\infty e^{\eta y}dy=-1/\eta$, so $A(\eta)=-\log(-\eta)$. Differentiating gives mean $-1/\eta=1/\lambda$ and variance $1/\eta^2=1/\lambda^2$. At nonnegative $\eta$ the integral diverges, establishing why the domain restriction is essential.

2. The average count is 2, and the coefficient-dependent objective is $e^\theta-2\theta$. Its derivative vanishes at $\theta=\log2$. The second derivative is $e^\theta$, so the Hessian at the optimum is 2. This agrees with the model's Poisson variance and with the general covariance-curvature identity. If observation durations differed, this calculation would need exposure offsets rather than the unadjusted average count.

3. Consider scalar input $x$ and scores $z_j=x$ and $z_k=2$ using an intercept for the second class. Class $j$'s zero-score point is $x=0$, but equality between the two scores occurs at $x=2$. Softmax ranks classes by their relative scores, so the true pairwise boundary is $z_j-z_k=0$. A third class can further determine whether either class wins in part of that pairwise boundary, but it does not change the equality condition itself.

4. Replacing each $z_j$ by $z_j+c$ multiplies numerator and denominator of each probability by $e^c$, which cancels. Alternatively, $A(z+c\mathbf1)=A(z)+c$ and $q^\top(z+c\mathbf1)=q^\top z+c$ because $q$ sums to one. Their difference is unchanged. The Hessian satisfies $[\operatorname{diag}(p)-pp^\top]\mathbf1=p-p(p^\top\mathbf1)=0$. Its singularity is therefore an exact symmetry, not necessarily a numerical failure.

5. Use a binary vector such as $(y_{\rm plumbing},y_{\rm electrical})\in\{0,1\}^2$. Independent sigmoid outputs with a sum of binary cross-entropies are a simple baseline that permits both entries to be one. They model marginal probabilities without requiring exclusivity; a richer joint model may be needed if label dependencies matter. A softmax over the two labels forces their probabilities to sum to one, excluding the intended simultaneous-positive interpretation. Smoothing that target only spreads one unit of mass and retains the wrong output semantics. Evaluate each label and simultaneous-label cases on held-out documents.

</details>

## Primary sources

- [Original recording](https://www.youtube.com/watch?v=8gVi4Rk21Eg): exponential families, GLMs, multiclass probabilities, and label smoothing.
- [Wainwright and Jordan, Graphical Models, Exponential Families, and Variational Inference](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf): mathematical development of normalization, moments, and geometry.
- [Müller, Kornblith, and Hinton, When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629): empirical benefits and limitations of soft targets.
- [Jaynes, Information Theory and Statistical Mechanics](https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620): primary source for the maximum-entropy/statistical-mechanics connection.

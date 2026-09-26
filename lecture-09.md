# Lecture 9 — Clustering, Gaussian mixtures, and the geometry of lower bounds

The property service has a large archive of sensor measurements but only a small number of labeled inspections. It wants to explore recurring patterns before deciding what to label. Clustering can organize observations, but it cannot supply the missing meaning by itself. A mathematical objective must first say what counts as a coherent group.

The recording develops hard clustering with k-means, replaces hard assignments with Gaussian-mixture probabilities, and introduces Jensen's inequality as the foundation for expectation–maximization. This chapter follows that sequence. The property examples, numerical iterations, code, and counterexamples are original. Research extensions are linked to primary publications, and model assumptions are kept distinct from claims about physical populations.

## 1. A cluster is defined by an objective and a representation [00:00](https://www.youtube.com/watch?v=bSmIGBCoffA&t=0s)

In supervised classification, observed labels define a target distinction. In clustering, the observations $x_1,\ldots,x_n\in\mathbb R^d$ arrive without those labels. We choose an integer $K$, centers $\mu_1,\ldots,\mu_K$, and assignments $c_i\in\{1,\ldots,K\}$.

For k-means, the objective is the sum of squared Euclidean distances to assigned centers:

$$\Phi(c,\mu)=\sum_{i=1}^n\|x_i-\mu_{c_i}\|^2.$$

Equivalently, after optimizing assignments for fixed centers,

$$\Phi(\mu)=\sum_i\min_{1\le j\le K}\|x_i-\mu_j\|^2.$$

The word “means” refers to the arithmetic means that minimize squared-distance error within fixed groups. It is not an assertion that every population naturally consists of spherical clusters.

For the service, take a one-dimensional sensor index with observations $0,1,9,10$. With $K=2$, centers $0.5$ and $9.5$ give total squared error 1. This is a useful compression of the measurements. It does not establish that the two groups correspond to safe and unsafe buildings, nor that there are exactly two physical causes.

Cluster names are arbitrary. Permuting the center indices and assignment labels leaves the objective unchanged. “Cluster 1” in one run need not correspond to “cluster 1” in another. Comparing runs requires matching centers or comparing partitions in a label-invariant way.

### The representation chooses which differences matter

Suppose one coordinate is area in square meters and another is a dimensionless moisture score. Raw squared Euclidean distance adds quantities with incompatible units. Numerically large coordinates can dominate. Standardization, a scientifically chosen metric, or a learned embedding changes what proximity means.

An invertible feature rescaling generally changes k-means assignments. Rotations and translations preserve Euclidean distances, and a common positive scalar multiplies the objective by a constant, but coordinate-specific scaling changes relative distances. Preprocessing is therefore part of the model rather than an innocuous formatting step.

Even uniform data can be partitioned into $K$ groups. A successful algorithmic run is not evidence that latent categories exist. For exploration, inspect representative points and uncertainty near boundaries, test stability under reasonable perturbations, and compare with independent variables that were not used to manufacture the partition.

The number of clusters is also a modeling choice. The optimal training objective cannot increase as $K$ increases: the larger model can reuse the smaller model's centers and ignore an extra one. At $K=n$, placing a center at every observation gives zero error. Minimizing this unpenalized training score over $K$ therefore does not identify a meaningful number of groups.

## 2. Lloyd's algorithm alternates two exact conditional minimizations [08:00](https://www.youtube.com/watch?v=bSmIGBCoffA&t=480s)

Given current centers, the assignment step chooses

$$c_i\leftarrow\arg\min_j\|x_i-\mu_j\|^2.$$

Each observation can be assigned independently because the objective is a sum. Squaring does not alter which nonnegative distance is smallest. It matters in the center-update step because squared distance has the arithmetic mean as its minimizer.

For a nonempty cluster $C_j=\{i:c_i=j\}$, the center update is

$$\mu_j\leftarrow\frac1{|C_j|}\sum_{i\in C_j}x_i.$$

To prove this, let $\bar x_j$ be that mean. For any proposed center $\mu$,

$$\sum_{i\in C_j}\|x_i-\mu\|^2
=\sum_{i\in C_j}\|x_i-\bar x_j\|^2
+|C_j|\|\bar x_j-\mu\|^2.$$

The cross-term vanishes because $\sum_{i\in C_j}(x_i-\bar x_j)=0$. The second term is nonnegative and is zero exactly at $\mu=\bar x_j$. This identity supplies both the update and its monotonicity guarantee.

Each assignment step cannot increase $\Phi$, and each center update cannot increase it. Under exact arithmetic, consistent tie handling, and a specified empty-cluster policy, the standard finite-assignment argument explains why the procedure stabilizes. Monotonicity alone, for an arbitrary real-valued iterative process, would not prove finite termination. Practical implementations also impose an iteration limit and a numerical stopping tolerance.

### Follow one complete run

Start the observations $0,1,9,10$ at centers 0 and 1. The first point goes to center 0, and the remaining three go to center 1. The initial objective is $0+0+64+81=145$.

The updated centers are 0 and $20/3$. Holding those assignments fixed, the objective becomes $146/3\approx48.6667$. On the next assignment step, both 0 and 1 join the first center, while 9 and 10 join the second. Updating again yields centers $0.5$ and $9.5$ and objective 1. The next iteration leaves the assignments and centers unchanged.

The following function performs one assignment-and-update cycle. Inputs are a finite data matrix and a finite, compatible matrix of centers. It returns the chosen labels and new centers without modifying inputs. Ties select the first center. An empty cluster raises an explicit error so the caller must choose a policy rather than silently accepting a division by zero.

```python
import numpy as np


def lloyd_step(data, centers):
    data = np.asarray(data, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if data.ndim != 2 or centers.ndim != 2:
        raise ValueError("Data and centers must be matrices")
    if not len(data) or not len(centers) or not data.shape[1]:
        raise ValueError("Data and centers must be nonempty")
    if data.shape[1] != centers.shape[1]:
        raise ValueError("Feature dimensions must agree")
    if not np.isfinite(data).all() or not np.isfinite(centers).all():
        raise ValueError("Inputs must be finite")
    distance2 = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    labels = distance2.argmin(axis=1)
    if np.any(np.bincount(labels, minlength=len(centers)) == 0):
        raise ValueError("Empty cluster: choose a reseeding policy")
    updated = np.stack([
        data[labels == j].mean(axis=0) for j in range(len(centers))
    ])
    before = distance2[np.arange(len(data)), labels].sum()
    after = ((data - updated[labels]) ** 2).sum()
    assert after <= before + 1e-10
    return labels, updated


data = np.array([[0.0], [1.0], [9.0], [10.0]])
labels, centers = lloyd_step(data, [[0.0], [1.0]])
assert np.allclose(centers[:, 0], [0, 20 / 3])
labels, centers = lloyd_step(data, centers)
assert np.allclose(centers[:, 0], [0.5, 9.5])
assert np.isclose(((data - centers[labels]) ** 2).sum(), 1)
```

This transparent calculation allocates an $n\times K\times d$ temporary difference array. It is appropriate for the tiny teaching case. Large datasets should compute distances in blocks or use an established implementation rather than allocate this tensor.

An empty cluster has no empirical mean. Retaining its previous center, reseeding it at a poorly represented observation, or reducing $K$ are different policies. Any intervention should be evaluated against the stated objective and recorded as part of the algorithm. The mere fact that the formula contains a mean does not define behavior when the denominator is zero.

## 3. Initialization changes optimization, not the meaning of the objective [16:00](https://www.youtube.com/watch?v=bSmIGBCoffA&t=960s)

Lloyd's algorithm reaches a fixed point of its alternating updates, not necessarily a global optimum. A concrete example uses four rectangle corners:

$$(-2,-1),\;(-2,1),\;(2,-1),\;(2,1).$$

With centers $(0,-1)$ and $(0,1)$, assignments separate the bottom and top rows. Recomputing their means changes nothing. Every point has squared distance 4 to its center, so the objective is 16.

With centers $(-2,0)$ and $(2,0)$, assignments separate left and right columns. This is also a fixed point, but every squared distance is 1, giving objective 4. Monotone improvement cannot leave the first configuration if neither conditional update improves it.

Random restarts explore different basins. Comparing their final values of the same objective helps select a better optimization result for a fixed representation and $K$. It does not resolve whether that objective answers the intended scientific question.

[Arthur and Vassilvitskii's k-means++ paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) improves the initialization. Choose the first center uniformly from the observations. Then choose each subsequent center with probability proportional to its squared distance $D(x)^2$ from the nearest already chosen center. This preferentially covers regions poorly represented by existing centers.

For the original squared-distance problem, the paper proves an expected initialization cost bound

$$\mathbb E[\Phi]\le8(\log K+2)\Phi_{\rm opt}.$$

The expectation is over the seeding randomness; this is not a deterministic guarantee for every run. Subsequent Lloyd updates cannot worsen that starting cost. The result concerns objective approximation, not recovery of true latent classes.

The exact global k-means optimization problem is computationally hard in general, but that does not imply every small instance is difficult or that the objective has no minimum. Complexity statements concern families of instances. They should not be used as a substitute for analyzing the specific data and algorithm at hand.

Choosing $K$ can draw on domain knowledge, stability, compression needs, and downstream utility. An elbow in the objective curve is a diagnostic, not an identified physical law. A probabilistic model adds likelihood-based comparisons, but it introduces additional assumptions rather than making model selection assumption-free.

For the property service, one useful evaluation is whether clusters reduce the cost of selecting diverse inspections. A partition can be operationally useful even if it does not correspond to a unique set of natural building types. The evaluation criterion should reflect that actual use.

## 4. A Gaussian mixture specifies a generative story [29:00](https://www.youtube.com/watch?v=bSmIGBCoffA&t=1740s)

A Gaussian mixture introduces an unobserved component indicator $Z\in\{1,\ldots,K\}$. Let $\pi_j=P(Z=j)$, with $\pi_j\ge0$ and $\sum_j\pi_j=1$. Conditional on $Z=j$, draw

$$X\mid Z=j\sim\mathcal N(\mu_j,\Sigma_j).$$

The joint density and marginal density are

$$p_\theta(x,Z=j)=\pi_j\mathcal N(x;\mu_j,\Sigma_j),\qquad
p_\theta(x)=\sum_{j=1}^K\pi_j\mathcal N(x;\mu_j,\Sigma_j).$$

The parameter collection $\theta$ contains weights, means, and covariance matrices. Positive definite covariance matrices define ordinary nonsingular component densities. The latent indicator is unobserved; it is not a label provided by a human inspector.

The mixing weights are prior probabilities of component membership before seeing $x$, even if estimated from data. This use of “prior” differs from a Bayesian prior distribution over the model parameters themselves. Learning $\pi$ does not stop it from being a prior over $Z$ relative to observing $X$.

Unlike k-means, a general mixture can represent components with different spreads, orientations, and frequencies. Those extra degrees of freedom can describe overlapping populations more naturally, but they also require more data and create more ways to overfit.

For independent observations, the observed-data log likelihood is

$$\ell(\theta)=\sum_{i=1}^n
\log\left[\sum_{j=1}^K\pi_j
\mathcal N(x_i;\mu_j,\Sigma_j)\right].$$

The sum inside the logarithm marginalizes the missing indicator. If the indicators were observed, the complete-data log likelihood would instead separate into component-specific terms, much like labeled Gaussian fitting in Lecture 5. The missing assignments are the new obstacle.

### Soft assignments are posterior probabilities

Given a current parameter estimate, the responsibility of component $j$ for observation $i$ is

$$r_{ij}=P_\theta(Z_i=j\mid X_i=x_i)
=\frac{\pi_j\mathcal N(x_i;\mu_j,\Sigma_j)}
{\sum_{k=1}^K\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)}.$$

For each observation, responsibilities sum to one. Saying “60% in each of two components” would violate this constraint; a two-component uncertainty statement might instead be 60% and 40%.

Take scalar means 0 and 2, both with variance 1, and mixing weights $0.75$ and $0.25$. At $x=1$, the component densities are equal, so responsibilities are exactly the mixing weights. Geometric equidistance does not imply posterior probability $1/2$ when frequencies differ.

At $x=2$, the ratio of the second density to the first is $e^2$. The posterior odds for component 2 are $e^2/3$, giving probability $e^2/(3+e^2)\approx0.7112$. The observation favors component 2 enough to overcome its smaller mixing weight, but not enough to make membership certain.

These probabilities describe membership under the fitted mixture. A poorly specified Gaussian model can produce precise-looking but misleading assignments. In particular, one elongated or curved physical population may be represented by several Gaussian components. Component count and physical-source count need not coincide.

## 5. Expectation and maximization alternate inference and estimation [50:00](https://www.youtube.com/watch?v=bSmIGBCoffA&t=3000s)

The expectation step, or E-step, computes current responsibilities. The maximization step, or M-step, treats those numbers as fixed weights and refits the component parameters. Define the effective component count $N_j=\sum_i r_{ij}$. For a full Gaussian mixture,

$$\pi_j^{\rm new}=\frac{N_j}{n},\qquad
\mu_j^{\rm new}=\frac1{N_j}\sum_i r_{ij}x_i,$$

$$\Sigma_j^{\rm new}
=\frac1{N_j}\sum_i r_{ij}
(x_i-\mu_j^{\rm new})(x_i-\mu_j^{\rm new})^\top.$$

The covariance is centered at the new mean. Its denominator is the effective count, following maximum likelihood, rather than an unbiased-estimation correction. Lecture 10 derives these updates from a common objective; here their weighted-count interpretation connects directly to k-means.

When responsibilities are exactly zero or one, the mean update becomes the arithmetic mean of assigned points. For common fixed spherical variance $\sigma^2I$ and fixed positive mixing weights, responsibilities concentrate on the unique nearest center as $\sigma^2\downarrow0$. This provides a precise limiting connection to hard assignments. Ties and changing covariance models require separate care.

### One soft iteration by hand

Use observations $x_1=0,x_2=2$, initial means 0 and 2, equal mixing weights, and unit variances. At $x=0$, responsibility for the first component is

$$r=\frac1{1+e^{-2}}\approx0.880797.$$

By symmetry, the responsibility matrix is $\begin{pmatrix}r&1-r\\1-r&r\end{pmatrix}$. Each effective count is 1, so the weights stay equal. The new means are $2(1-r)\approx0.238406$ and $2r\approx1.761594$. Each new variance is $4r(1-r)\approx0.419974$.

Unlike k-means, each observation influences both means. The uncertain assignment pulls each mean slightly toward the other point. Refitting variances then changes how sharply the next E-step distinguishes the components.

The implementation below evaluates one-dimensional mixture responsibilities and observed log likelihood in log space. It accepts nonempty finite vectors, strictly positive variances and weights, and weights summing to one. It returns an $n\times K$ responsibility matrix and a scalar log likelihood. There are no external effects.

```python
def mixture_estep(points, means, variances, weights):
    points, means, variances, weights = [
        np.asarray(v, dtype=float)
        for v in (points, means, variances, weights)
    ]
    if any(v.ndim != 1 or not v.size for v in
           (points, means, variances, weights)):
        raise ValueError("Expected nonempty vectors")
    if means.shape != variances.shape or means.shape != weights.shape:
        raise ValueError("Component parameter shapes must match")
    if not all(np.isfinite(v).all() for v in
               (points, means, variances, weights)):
        raise ValueError("Inputs must be finite")
    if np.any(variances <= 0) or np.any(weights <= 0):
        raise ValueError("Variances and weights must be positive")
    if not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must sum to one")
    log_joint = np.log(weights)[None, :] - 0.5 * (
        np.log(2 * np.pi * variances)[None, :]
        + (points[:, None] - means[None, :]) ** 2 / variances[None, :]
    )
    log_density = np.logaddexp.reduce(log_joint, axis=1)
    responsibility = np.exp(log_joint - log_density[:, None])
    return responsibility, float(log_density.sum())


points = np.array([0.0, 2.0])
responsibility, old_ll = mixture_estep(points, [0, 2], [1, 1], [0.5, 0.5])
counts = responsibility.sum(axis=0)
means = (responsibility.T @ points) / counts
variances = (
    responsibility * (points[:, None] - means[None, :]) ** 2
).sum(axis=0) / counts
weights = counts / len(points)
_, new_ll = mixture_estep(points, means, variances, weights)
assert np.allclose(responsibility.sum(axis=1), 1)
assert np.allclose(means, [0.238405844, 1.761594156])
assert np.allclose(variances, 0.419974342)
assert new_ll >= old_ll
```

This is a checked single iteration, not a robust unrestricted mixture fitter. With only two observations, repeatedly estimating two free variances can drive them toward degeneracy. The example is deliberately small enough to audit while exposing a genuine model failure.

### Likelihood can improve while the model degenerates

Put one component mean exactly at an observation and let its covariance shrink to zero, while another component maintains nonzero density on the remaining observations. The shrinking component's density at its mean diverges, so the total log likelihood can become arbitrarily large. An unconstrained Gaussian-mixture likelihood need not have a finite maximizer.

Covariance floors, constrained covariance families, or probabilistic regularization can prevent this behavior, but they change the optimization problem. A component with negligible effective count also requires an explicit policy. Stable log-space arithmetic prevents avoidable underflow; it does not solve statistical degeneracy.

The historical primary reference for the general alternating method is [Dempster, Laird, and Rubin's EM paper](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x). Its generality will matter more in the next chapter, where the latent variable need not be a cluster indicator.

## 6. Jensen's inequality constructs a lower bound that can touch the likelihood [60:00](https://www.youtube.com/watch?v=bSmIGBCoffA&t=3600s)

A set is convex if it contains every line segment joining two of its points. A function $f$ on a convex domain is convex if

$$f(\lambda a+(1-\lambda)b)
\le\lambda f(a)+(1-\lambda)f(b),\qquad 0\le\lambda\le1.$$

The graph lies below its chords. More precisely, the epigraph $\{(x,t):t\ge f(x)\}$ is a convex set; the graph itself generally is not. This distinction makes the geometric picture precise.

For a twice-differentiable scalar function on an interval, $f''\ge0$ characterizes convexity. A nonnegative first derivative characterizes monotonicity instead. Strong convexity requires a uniform positive curvature lower bound, not merely a pointwise statement that curvature happens to be positive. For example, $e^x$ has positive second derivative everywhere but is not strongly convex on all of $\mathbb R$ because that derivative approaches zero as $x\to-\infty$.

Jensen's inequality extends the chord inequality to expectations:

$$f(\mathbb E[X])\le\mathbb E[f(X)]$$

under the relevant integrability conditions. For a concave function, the direction reverses. The logarithm is concave on positive numbers because its second derivative is $-1/x^2$.

For values 1 and 9 with equal weights,

$$\log\left(\frac{1+9}{2}\right)=\log5
\ge\frac12\log1+\frac12\log9=\log3.$$

The gap is $\log(5/3)>0$. Replacing a log of an average by an average of logs creates a lower bound, not an equality.

### The lower bound becomes useful when its gap can be controlled

For one observation, choose any positive distribution $q(z)$ over latent states covering the support of $p_\theta(x,z)$. Then

$$\log p_\theta(x)
=\log\sum_z q(z)\frac{p_\theta(x,z)}{q(z)}
\ge\sum_zq(z)\log\frac{p_\theta(x,z)}{q(z)}
\equiv\mathcal L(q,\theta).$$

The expression $\mathcal L$ is an evidence lower bound, or ELBO. It is the expected complete-data log density plus the entropy of $q$. The support condition prevents the importance-weight rewrite from silently dropping probability mass.

Because the logarithm is strictly concave, equality holds when $p_\theta(x,z)/q(z)$ is constant over the relevant support. Choosing $q(z)=p_\theta(z\mid x)$ makes that ratio equal to $p_\theta(x)$, so the bound touches the likelihood.

At iteration $t$, EM chooses the current posterior $q_t$, then improves the bound as a function of $\theta$. If that improvement is exact or at least nondecreasing,

$$\ell(\theta_{t+1})\ge
\mathcal L(q_t,\theta_{t+1})\ge
\mathcal L(q_t,\theta_t)=\ell(\theta_t),$$

where the multi-observation version sums individual bounds. This is the mathematical content of the recording's touching-curve picture.

The lower bound need not be globally concave in every latent-variable model or parameterization. EM needs a tractable or improving M-step, not a universal concavity theorem. Monotonic likelihood also does not guarantee a global optimum, finite parameter convergence, or immunity to the mixture singularity above.

For our property archive, EM alternates a model-based explanation of which component could have generated each measurement with a weighted re-estimation of those components. The procedure is internally coherent. Whether its explanations correspond to useful building populations remains an external validation question.

## Exercises

1. **Derive.** Prove the mean-update identity for a fixed nonempty cluster, and explain why it fails to define an update for an empty cluster.
2. **Compute.** Verify the two fixed points of Lloyd's algorithm on the rectangle corners and their costs. Explain why a monotonic algorithm can remain at the worse one.
3. **Compute.** In the two-component mixture with means 0 and 2, unit variances, and weights $0.75,0.25$, find the second component's posterior probability at 1 and 2.
4. **Counterexample.** Explain why increasing $K$ can improve the training objective without discovering more meaningful categories. Give a separate example in which increasing Gaussian-mixture likelihood reflects a pathology.
5. **Derive and design.** State the equality condition in the ELBO construction. Then propose a practical diagnostic protocol for fitting a mixture to the property archive.

<details><summary>Solutions</summary>

1. Write $x_i-\mu=(x_i-\bar x)+(\bar x-\mu)$ and expand the squared norms. Summing the cross-term gives $2(\bar x-\mu)^\top\sum_i(x_i-\bar x)=0$. The remaining center-dependent term is $|C|\|\bar x-\mu\|^2$, minimized at the mean. For an empty cluster, $|C|=0$ and the objective contributes zero for every center, so there is no data-defined mean or unique minimizer. A policy must be supplied separately.

2. With horizontal centers $(0,-1),(0,1)$, each point chooses its own row because its squared distances are 4 versus 8. The means remain unchanged, and total cost is 16. With vertical centers $(-2,0),(2,0)$, squared distances are 1 versus 17, giving total cost 4 and another fixed point. Moving from the first configuration to the second requires a coordinated change that the two conditional minimizations do not initiate from that fixed point.

3. At 1, equal component densities cancel, giving probability $0.25$. At 2, the density ratio is $e^2$ in favor of component 2. Multiplying by prior odds $1/3$ and converting back gives $e^2/(3+e^2)\approx0.7112$. The result combines geometric compatibility with mixture frequency.

4. For any finite dataset, assigning one center per observation yields zero k-means distortion, even if observations are samples from one continuous unimodal population. More centers can improve quantization without identifying additional physical categories. Separately, a Gaussian component centered on one observation with variance tending to zero creates arbitrarily large density at that point; another component can explain the remaining observations. The likelihood increase then reflects covariance collapse rather than a useful population model.

5. Equality requires a constant ratio $p_\theta(x,z)/q(z)$ over the relevant support, achieved by the exact posterior $q(z)=p_\theta(z\mid x)$. In practice, record initialization, covariance constraints, stopping criteria, and preprocessing; inspect the likelihood trajectory and effective component counts; check covariance eigenvalues for collapse; compare restarts after accounting for label permutations; and evaluate held-out density and downstream inspection usefulness. Use independent metadata to test proposed interpretations. A smooth likelihood trace verifies one aspect of optimization, not the semantic validity of the discovered groups.

</details>

## Primary references and evidence boundaries

The [recording](https://www.youtube.com/watch?v=bSmIGBCoffA) provides the progression from clustering to mixtures and Jensen's inequality. [Arthur and Vassilvitskii](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) support the seeding algorithm and expected approximation guarantee. [Dempster, Laird, and Rubin](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x) provide the primary EM reference. The property case, fixed-point example, soft iteration, code, and convexity clarifications are original exposition. The next chapter develops the full variational identity and then connects unsupervised structure to principal component analysis.

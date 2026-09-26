# Lecture 11 — Diffusion models: forward noise, reverse dynamics, and a trajectory ELBO

Suppose the property service wants a model of inspection images, rather than a classifier of predefined defects. The statistical objective is to learn a distribution from which new images can be sampled. A diffusion model approaches this by defining an easy corruption process and learning many small reverse transitions.

The recording develops the noising process, its Gaussian marginal, reverse-time intuition, a continuous-time connection, and a variational training objective. This chapter follows that order. A scalar sensor model carries the numerical calculations so every coefficient can be checked before interpreting the same equations in image space. Research extensions are linked to primary papers; broad claims about which architecture currently dominates an application are not needed for the derivation.

## 1. A generative model specifies a distribution, not one reconstruction [00:00](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=0s)

Let $X_0\in\mathbb R^d$ follow the data distribution $q_{\rm data}$. Training provides samples from this distribution. The desired model is a distribution $p_\theta(X_0)$ that can generate new observations with appropriate structure.

The subscript 0 labels the clean endpoint. Increasing time indices mean more corruption. The learned sampler will start at a large index $T$ and move toward 0. These indices describe a designed stochastic process, not the chronological age of the image.

Choose a noise schedule $\beta_1,\ldots,\beta_T$ with $0<\beta_t<1$. Define independent standard normal innovations $\epsilon_t\sim\mathcal N(0,I_d)$, independent of $X_0$, and the forward recursion

$$X_t=\sqrt{1-\beta_t}\,X_{t-1}
+\sqrt{\beta_t}\,\epsilon_t.$$

Equivalently, its transition density is

$$q(x_t\mid x_{t-1})
=\mathcal N(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_tI_d).$$

The notation $\mathcal N(x;\mu,\Sigma)$ denotes a density evaluated at $x$; $\mathcal N(\mu,\Sigma)$ denotes the distribution itself. This distinction is useful when multiplying densities or taking their logarithms.

The square roots are essential. The coefficients multiply random vectors, so their squares determine covariance contributions. The recording corrects missing square roots during the derivation; we use the consistent variance convention throughout.

The forward process is fixed by design. In this formulation its schedule and transition rule are not learned neural networks. Learning concerns the reverse model. The term “denoising” should not imply that the model deterministically retrieves a unique original image from pure noise. It samples a plausible predecessor distribution at each stage.

### Why the process shrinks the signal while adding noise

Let $\alpha_t=1-\beta_t$. Independence gives

$$\mathbb E[X_t]=\sqrt{\alpha_t}\,\mathbb E[X_{t-1}],$$

$$\operatorname{Cov}(X_t)
=\alpha_t\operatorname{Cov}(X_{t-1})+\beta_tI_d.$$

If the initial covariance is identity, every later covariance is identity. Otherwise, each step moves covariance toward identity. This is why the construction is called variance-preserving in a standard diffusion convention: it avoids the unbounded variance growth of repeatedly adding noise without signal attenuation.

Identity covariance does not imply a Gaussian distribution. A vector can be non-Gaussian despite zero mean and identity covariance. The approach to a Gaussian comes from the explicit marginal formula developed next, not from covariance alone.

For the image application, this forward process is a training device. It is not a physical model claiming that a real camera repeatedly shrinks pixel values and injects independent Gaussian noise. Its usefulness comes from creating analytically manageable corrupted versions of the training data.

## 2. Unrolling the chain gives direct sampling at any noise level [14:00](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=840s)

Define the cumulative signal factor

$$\bar\alpha_t=\prod_{s=1}^t\alpha_s,\qquad \bar\alpha_0=1.$$

Unrolling the recursion gives

$$X_t=\sqrt{\bar\alpha_t}\,X_0+
\sum_{s=1}^t\sqrt{\beta_s\prod_{r=s+1}^t\alpha_r}\,\epsilon_s.$$

The second term is Gaussian because it is a linear combination of independent Gaussian vectors. Its covariance is the sum of the squared coefficients. That sum telescopes:

$$\sum_{s=1}^t\beta_s\prod_{r=s+1}^t\alpha_r
=\sum_{s=1}^t
\left(\prod_{r=s+1}^t\alpha_r-\prod_{r=s}^t\alpha_r\right)
=1-\bar\alpha_t.$$

Consequently,

$$\boxed{q(x_t\mid x_0)=
\mathcal N(x_t;\sqrt{\bar\alpha_t}\,x_0,
(1-\bar\alpha_t)I_d).}$$

For a chosen time, we can sample directly as

$$X_t=\sqrt{\bar\alpha_t}\,X_0+
\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad
\epsilon\sim\mathcal N(0,I_d).$$

This avoids simulating all preceding steps when constructing a training example. It will make the noise-prediction objective in Lecture 12 efficient.

For a scalar sensor, take $\beta_1=0.1$, $\beta_2=0.2$, and clean value $x_0=2$. Then $\bar\alpha_2=0.9(0.8)=0.72$, so

$$X_2\mid X_0=2\sim
\mathcal N(2\sqrt{0.72},0.28).$$

The conditional mean is about 1.69706 and conditional variance is 0.28. Expanding the two-step recursion gives noise $\sqrt{0.08}\epsilon_1+\sqrt{0.2}\epsilon_2$, whose variance is $0.08+0.2=0.28$. The direct and sequential constructions agree for this marginal.

### A marginal construction is not automatically a trajectory construction

The effective noises at two different times along the same forward trajectory share innovations. They are correlated. Drawing an independent fresh $\epsilon$ in the direct formula for each time gives correct individual marginals but generally the wrong joint trajectory.

In the two-step example, write the first effective noise as $\epsilon_1$ and the second as

$$\hat\epsilon_2=
\frac{\sqrt{0.08}\epsilon_1+\sqrt{0.2}\epsilon_2}{\sqrt{0.28}}.$$

Their correlation is $\sqrt{0.08/0.28}=\sqrt{2/7}\approx0.5345$. This does not matter for an objective sampling one time at a time, but it matters for joint path calculations and for interpreting reverse transitions.

The marginal formula also gives

$$\operatorname{Cov}(X_t)
=\bar\alpha_t\operatorname{Cov}(X_0)
+(1-\bar\alpha_t)I_d.$$

As $\bar\alpha_t\to0$, the signal contribution vanishes and the marginal approaches the standard normal under ordinary finiteness conditions. At a finite terminal time, this is an approximation unless the data distribution has a special matching form.

The product of numbers below one does not automatically tend to zero. For example, if $\beta_t=1/(t+1)^2$, then

$$\bar\alpha_T=\prod_{t=1}^T
\frac{t(t+2)}{(t+1)^2}
=\frac{T+2}{2(T+1)}\longrightarrow\frac12.$$

That schedule never removes all signal. An exact condition for the product to vanish is $\sum_t-\log(1-\beta_t)=\infty$; the usual small-step intuition is a sufficiently large accumulated noise budget. A finite schedule should be checked through its actual terminal signal-to-noise level.

## 3. The reverse conditional contains uncertainty about the past [28:00](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=1680s)

The full forward joint distribution is

$$q(x_{0:T})=q_{\rm data}(x_0)
\prod_{t=1}^Tq(x_t\mid x_{t-1}).$$

It induces a reverse conditional $q(x_{t-1}\mid x_t)$. Knowing $x_t$ does not reveal which previous state and innovation produced it. Reversal is therefore a conditional-distribution problem, not algebraic subtraction of known noise.

A learned generative chain uses a simple terminal prior and parameterized reverse transitions:

$$p_\theta(x_{0:T})=p(x_T)
\prod_{t=1}^Tp_\theta(x_{t-1}\mid x_t),\qquad
p(x_T)=\mathcal N(x_T;0,I_d),$$

$$p_\theta(x_{t-1}\mid x_t)=
\mathcal N(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I_d).$$

Here $\mu_\theta$ is a neural network that receives both the noisy observation and its time or noise level. For this chapter, $\sigma_t^2$ is a chosen variance. Later variants can learn or otherwise change that parameterization.

At sampling time, a reverse step can be written

$$X_{t-1}=\mu_\theta(X_t,t)+\sigma_t\xi_t,$$

with a new Gaussian draw $\xi_t$. Adding random noise during a denoising step is not contradictory: the mean moves the sample according to a learned conditional distribution, while the variance expresses unresolved uncertainty about the previous state.

### Forward Gaussian noise does not make every finite reverse conditional Gaussian

Suppose the clean scalar $X_0$ is equally likely to be $-2$ or 2. After one Gaussian corruption, the noisy observation has a mixture density. Conditional on a noisy value, $X_0$ still has two possible discrete values with posterior weights. Its reverse conditional is not a single Gaussian.

At later finite times, a multimodal data distribution can likewise induce non-Gaussian reverse conditionals. Gaussian reverse kernels are a model choice and a useful small-step approximation. The continuous-time diffusion theory motivates local Gaussian increments under suitable assumptions; it does not say that arbitrary finite-time transitions or arbitrary stochastic processes are Gaussian.

If a single forward step erased the entire image, its reverse conditional would need to represent the entire complex data distribution at once. Many small corruptions create a sequence of more local inference tasks. This is a useful explanation for the construction, not a proof that any schedule or neural architecture will learn successfully.

[Ho, Jain, and Abbeel's DDPM paper](https://arxiv.org/abs/2006.11239) develops the discrete diffusion formulation and a practical training parameterization. The equations here isolate the probabilistic structure before choosing the network architecture.

## 4. Reverse-time SDEs explain the role of the score [43:00](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=2580s)

For a continuous-time connection, consider the stochastic differential equation

$$dX_t=f(X_t,t)\,dt+g(t)\,dW_t,$$

where $W_t$ is standard Brownian motion and $g(t)$ is a scalar diffusion coefficient independent of position. Brownian increments over a short interval of length $h$ have covariance $hI_d$. Thus an Euler step is approximately

$$X_{t+h}=X_t+f(X_t,t)h+g(t)\sqrt h\,\epsilon.$$

For the variance-preserving process, expanding $\sqrt{1-\beta(t)h}$ to first order gives drift $f(x,t)=-\tfrac12\beta(t)x$ and diffusion coefficient $g(t)=\sqrt{\beta(t)}$.

Let $q_t(x)$ denote the time-$t$ marginal density. Its score is the vector

$$s_t(x)=\nabla_x\log q_t(x).$$

It points in the local direction of increasing log density and is not a class score or a predicted probability.

To avoid a sign ambiguity, introduce an increasing reverse clock $\tau=T-t$ and define $Y_\tau=X_{T-\tau}$. Under suitable smoothness, positivity, and time-reversal regularity assumptions, the reverse process has the form

$$\boxed{dY_\tau=
\left[-f(Y_\tau,T-\tau)
+g(T-\tau)^2s_{T-\tau}(Y_\tau)\right]d\tau
+g(T-\tau)\,d\bar W_\tau.}$$

The score correction depends on the marginal data distribution. Merely negating the forward drift omits the information needed to infer likely predecessors.

A useful sign check is the stationary standard-normal variance-preserving process. Its score is $s_t(x)=-x$. With constant $\beta$, the reverse drift becomes $+\tfrac12\beta x-\beta x=-\tfrac12\beta x$, matching the forward stationary drift. A formula that gives outward drift in this stationary example has mixed time conventions.

This theory traces to work including [Anderson's 1982 reverse-time diffusion paper](https://www.sciencedirect.com/science/article/pii/0304414982900515). [Song and colleagues](https://arxiv.org/abs/2011.13456) connect reverse SDEs and score estimation to modern generative modeling. The theorem uses assumptions on the process and density; it is not a central-limit argument that all continuous-time processes become Gaussian.

### A deterministic flow can share the same one-time marginals

As a researched connection, the associated probability-flow ordinary differential equation uses

$$\frac{dX_t}{dt}=f(X_t,t)-\frac12g(t)^2s_t(X_t).$$

With the exact score and suitable regularity, it has the same time-indexed marginals as the forward SDE. It does not have the same random trajectories or joint path distribution.

The factor of one half can be understood from the Fokker–Planck equation. The SDE density evolves as

$$\partial_tq_t=-\nabla\cdot(fq_t)
+\frac12g(t)^2\Delta q_t.$$

Because $q_ts_t=\nabla q_t$, the deterministic velocity above produces the same density equation through $\partial_tq_t=-\nabla\cdot(vq_t)$. This is a statement about marginal transport. Numerical integration and score-estimation errors still affect a practical sampler.

## 5. Treating the entire noisy path as latent gives a tractable bound [55:00](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=3300s)

The observed object is $x_0$; the latent object is the whole path $x_{1:T}$. Choose the fixed forward conditional $q(x_{1:T}\mid x_0)$ as the variational distribution. The ELBO identity from Lecture 10 gives

$$\log p_\theta(x_0)\ge
\mathbb E_q\left[
\log p_\theta(x_{0:T})-\log q(x_{1:T}\mid x_0)
\right].$$

This $q$ is an explicitly designed noising distribution, not generally the exact posterior $p_\theta(x_{1:T}\mid x_0)$ of the learned generative model. Training therefore optimizes a variational objective. It is not ordinary exact EM with an E-step recomputing the learned model's posterior until the bound is tight.

Factor the forward conditional in reverse order while retaining its conditioning on the clean observation:

$$q(x_{1:T}\mid x_0)=q(x_T\mid x_0)
\prod_{t=2}^Tq(x_{t-1}\mid x_t,x_0).$$

The learned joint factors as the terminal prior times reverse transitions. Substituting both factorizations and grouping log ratios gives the negative ELBO:

$$-\operatorname{ELBO}(x_0)=
\underbrace{\operatorname{KL}(q(x_T\mid x_0)\|p(x_T))}_{\text{terminal prior term}}
+\sum_{t=2}^T
\mathbb E_{q(x_t\mid x_0)}
\left[
\operatorname{KL}\!\left(
q(x_{t-1}\mid x_t,x_0)\|
p_\theta(x_{t-1}\mid x_t)
\right)\right]
-\mathbb E_{q(x_1\mid x_0)}
[\log p_\theta(x_0\mid x_1)].$$

The expectation around each conditional KL is essential. A conditional divergence depends on the conditioning value; the chain rule averages it under the appropriate marginal distribution.

There are three qualitatively different terms. The terminal term checks how well the final forward distribution matches the chosen prior. Intermediate terms train one-step denoising distributions. The final reconstruction term handles the clean endpoint. It must not be silently replaced by the same positive-variance Gaussian formula used for all intermediate times.

For a fixed schedule and fixed terminal prior, the terminal term has no $\theta$ dependence, so it can be omitted when differentiating with respect to the reverse network. That does not make its value zero or make terminal-prior mismatch disappear at sampling time.

The apparent advantage is now visible: although $q(x_{t-1}\mid x_t)$ depends on the unknown data distribution, the conditional $q(x_{t-1}\mid x_t,x_0)$ is Gaussian and computable. Training has access to the clean observation $x_0$ and can use that easier conditional as a target.

## 6. Completing the square produces the reverse training target [74:00](https://www.youtube.com/watch?v=dqUMCzWjZSI&t=4440s)

For $t\ge2$, apply Bayes' rule with $x_0$ held fixed:

$$q(x_{t-1}\mid x_t,x_0)\propto
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).$$

Let $y=x_{t-1}$ for the algebra. Up to constants independent of $y$, the negative log density is

$$\frac1{2\beta_t}\|x_t-\sqrt{\alpha_t}y\|^2
+\frac1{2(1-\bar\alpha_{t-1})}
\|y-\sqrt{\bar\alpha_{t-1}}x_0\|^2.$$

Collecting quadratic terms gives precision

$$\left[\frac{\alpha_t}{\beta_t}
+\frac1{1-\bar\alpha_{t-1}}\right]I_d
=\frac{1-\bar\alpha_t}
{\beta_t(1-\bar\alpha_{t-1})}I_d.$$

Collecting linear terms and multiplying by the inverse precision gives

$$q(x_{t-1}\mid x_t,x_0)=
\mathcal N(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_tI_d),$$

$$\boxed{\tilde\beta_t=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,}$$

$$\boxed{\tilde\mu_t=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t.}$$

The mean combines the known clean endpoint and noisy observation. Its coefficients need not sum to one because the forward process also rescales the signal; it is not simply an ordinary convex average of two measurements in identical observation models.

For the two-step sensor schedule $\beta_1=0.1$, $\beta_2=0.2$, we have $\bar\alpha_1=0.9$, $\bar\alpha_2=0.72$, and posterior variance $\tilde\beta_2=1/14$. With $x_0=2$ and $x_2=1$, posterior mean is

$$\tilde\mu_2=
\frac{\sqrt{0.9}(0.2)}{0.28}(2)
+\frac{\sqrt{0.8}(0.1)}{0.28}(1)
\approx1.67470.$$

Independently, the prior precision for $X_1\mid x_0$ is $1/0.1=10$, and the observation contributes $\alpha_2/\beta_2=4$. Total precision is 14, confirming the variance and completing-square calculation.

The function below computes this conditional for an intermediate step. It requires matching finite arrays, a step variance in $(0,1)$, and a cumulative previous signal factor strictly between zero and one. Excluding the value one deliberately excludes the degenerate clean-endpoint case.

```python
import numpy as np


def posterior_parameters(x0, xt, beta, alpha_bar_previous):
    x0, xt = np.asarray(x0, dtype=float), np.asarray(xt, dtype=float)
    if x0.shape != xt.shape or not x0.size:
        raise ValueError("Clean and noisy values must have matching shapes")
    if not np.isfinite(x0).all() or not np.isfinite(xt).all():
        raise ValueError("Values must be finite")
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("Step variance must lie between zero and one")
    if not np.isfinite(alpha_bar_previous) or not 0 < alpha_bar_previous < 1:
        raise ValueError("Use an intermediate step with a noisy predecessor")
    alpha = 1 - beta
    denominator = 1 - alpha * alpha_bar_previous
    variance = beta * (1 - alpha_bar_previous) / denominator
    mean = (
        np.sqrt(alpha_bar_previous) * beta * x0
        + np.sqrt(alpha) * (1 - alpha_bar_previous) * xt
    ) / denominator
    return mean, variance


mean, variance = posterior_parameters([2], [1], 0.2, 0.9)
precision = 1 / 0.1 + 0.8 / 0.2
independent_mean = (
    np.sqrt(0.9) * 2 / 0.1 + np.sqrt(0.8) * 1 / 0.2
) / precision
assert np.isclose(variance, 1 / 14)
assert np.allclose(mean, independent_mean)
assert np.allclose(mean, [1.67470014])
```

### Matching Gaussian means turns a KL into a regression loss

For Gaussian distributions with means $m_1,m_2$ and the same positive definite covariance $\Sigma$,

$$\operatorname{KL}(\mathcal N(m_1,\Sigma)\|
\mathcal N(m_2,\Sigma))
=\frac12(m_1-m_2)^\top\Sigma^{-1}(m_1-m_2).$$

The determinant and covariance-trace terms cancel; the mean displacement remains. If the learned intermediate variance is chosen as $\sigma_t^2=\tilde\beta_t$, its KL training term is

$$\frac1{2\tilde\beta_t}
\|\mu_\theta(x_t,t)-\tilde\mu_t(x_t,x_0)\|^2.$$

In the scalar example, a mean error of 0.2 gives KL $0.2^2/(2/14)=0.28$. The same mean error is penalized more strongly when posterior uncertainty is smaller. This explains why variance-dependent coefficients cannot automatically be discarded when combining losses across different times.

At $t=1$, $\bar\alpha_0=1$ and $\tilde\beta_1=0$: conditioning on the clean endpoint makes the predecessor exactly $x_0$. This is a point mass, not an ordinary positive-variance Gaussian density. The endpoint reconstruction term in the ELBO handles this case separately.

The next lecture reparameterizes the learned mean in terms of predicted noise. That transformation yields a convenient regression target sampled directly from the forward marginal, while preserving an explicit distinction between the exact weighted variational objective and a simplified training loss.

## Exercises

1. **Derive.** Prove the telescoping variance identity for the forward chain and explain why effective noises at different times on one path are not independent.
2. **Compute.** For $\beta_1=0.1$, $\beta_2=0.2$, and $x_0=2$, obtain the conditional law of $X_2$. Then compute the posterior law of $X_1$ given $X_2=1$ and $X_0=2$.
3. **Counterexample.** Explain why a forward Gaussian corruption does not imply a Gaussian reverse conditional, using a two-point clean distribution.
4. **Derive.** Starting from the trajectory factorizations, identify the three types of terms in the negative ELBO and state which expectations accompany the intermediate KL terms.
5. **Check assumptions.** A derivation says that every $\beta_t>0$ guarantees complete signal destruction, that a reverse SDE must negate only the drift, and that $t=1$ can use the intermediate Gaussian KL unchanged. Diagnose all three claims.

<details><summary>Solutions</summary>

1. Replace $\beta_s$ by $1-\alpha_s$ inside each coefficient variance. Each summand becomes $\prod_{r=s+1}^t\alpha_r-\prod_{r=s}^t\alpha_r$, so adjacent products cancel, leaving $1-\bar\alpha_t$. Effective noise at a later time contains earlier innovations with nonzero coefficients, so it shares random terms with earlier effective noise. Correct one-time marginals do not specify the joint trajectory.

2. The direct marginal is $\mathcal N(2\sqrt{0.72},0.28)$. For the reverse conditional, prior precision is 10 and likelihood precision is 4, so the posterior variance is $1/14$. The precision-weighted linear term is $2\sqrt{0.9}/0.1+\sqrt{0.8}/0.2$. Dividing by 14 gives mean approximately 1.67470. This posterior conditions on the clean endpoint; it is not the unknown marginal reverse conditional that would average over possible clean data.

3. If $X_0$ is equally likely to be $-2$ or 2, observing a noisy $X_1=x$ changes the probabilities of those two values but does not turn them into a continuous Gaussian variable. Bayes' rule gives a two-point posterior with weights proportional to the corresponding Gaussian likelihoods. Finite-step reverse conditionals can remain multimodal or discrete despite Gaussian forward noise.

4. Reverse-factor the fixed conditional path as $q(x_T\mid x_0)\prod_{t=2}^Tq(x_{t-1}\mid x_t,x_0)$, and compare it with $p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}\mid x_t)$. The terminal ratio gives a prior KL, intermediate ratios give conditional KLs averaged over $q(x_t\mid x_0)$, and the unmatched final generative factor gives $-\mathbb E_{q(x_1\mid x_0)}\log p_\theta(x_0\mid x_1)$. All are conditioned on the training observation before averaging over the dataset.

5. Positive noise increments can have finite total strength: $\beta_t=1/(t+1)^2$ leaves cumulative signal factor tending to $1/2$. The increasing reverse clock requires drift $-f+g^2\nabla\log q_t$, not merely $-f$, under the stated scalar-diffusion assumptions. Finally, the clean-endpoint posterior has zero variance, so substituting it into a positive-variance Gaussian KL introduces a division by zero and ignores the separate reconstruction term.

</details>

## Primary references and evidence boundaries

The [recording](https://www.youtube.com/watch?v=dqUMCzWjZSI) grounds the sequence and the variational construction. Primary references are [Ho et al., DDPM](https://arxiv.org/abs/2006.11239), [Song et al., score-based SDE modeling](https://arxiv.org/abs/2011.13456), and [Anderson's reverse-time diffusion theory](https://www.sciencedirect.com/science/article/pii/0304414982900515). The scalar schedule, telescoping counterexample, posterior calculation, probability-flow derivation, and exercises are original exposition. Gaussian reverse kernels, finite-time prior matching, and exact-posterior EM are explicitly distinguished rather than treated as interchangeable assumptions.

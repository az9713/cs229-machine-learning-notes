# Lecture 12 — Learning to denoise, then learning to adapt

A property-inspection service has two scarce resources: clean examples of unusual damage and labeled examples of the decisions an inspector must make. A generative model can learn a distribution of images, while a representation model can turn images into features for a downstream decision. These are different statistical tasks. Success at generating plausible cracks does not establish that a classifier detects dangerous cracks.

This chapter completes the diffusion model from Lecture 11, then asks how a pretrained model should change when a new task arrives. The recurring object is a learned function with parameters $\theta$: first a noise predictor, later a feature extractor. In both cases, the important engineering decision is what the function predicts, what loss trains it, and which parameters are allowed to move. Caption-grounded topics retain their recording order. Numerical constructions and derivations are original teaching material; each research extension is identified beside its primary source. Captions, rather than visual board inspection, ground the lecture coverage.

## 1. Closing the diffusion objective at its endpoints [02:15](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=135s)

Let $x_0\in\mathbb R^d$ be a clean image represented by $d$ real coordinates. The forward chain adds Gaussian noise with variances $\beta_t\in(0,1)$ at integer steps $t=1,\ldots,T$. Write $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Its marginal is

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I_d).$$

Here $I_d$ is the identity matrix. The reverse model uses
$p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2 I_d)$,
with positive prescribed variance $\sigma_t^2$ for the steps considered below. The forward distribution is denoted $q$. It is fixed by the data distribution and noise schedule; it is not another trainable network.

The negative variational lower bound separates into a terminal prior mismatch, intermediate reverse-transition mismatches, and a clean-data reconstruction term:

$$\mathcal L_{\rm VLB}
=\operatorname{KL}(q(x_T\mid x_0)\|p(x_T))
+\sum_{t=2}^T\mathbb E_{q(x_t\mid x_0)}
\operatorname{KL}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t))
-\mathbb E_{q(x_1\mid x_0)}\log p_\theta(x_0\mid x_1).$$

The Kullback–Leibler divergence $\operatorname{KL}(q\|p)$ is the expectation under $q$ of $\log(q/p)$. The expectation over clean data is implicit in the training objective. Keeping these expectations matters: the terms compare conditional distributions at randomly sampled noisy inputs, rather than at one privileged trajectory.

### A small terminal mismatch is still a mismatch

With $p(x_T)=\mathcal N(0,I_d)$, the Gaussian KL evaluates exactly to

$$\frac12\left[
\bar\alpha_T\|x_0\|^2-d\bar\alpha_T-d\log(1-\bar\alpha_T)
\right].$$

This follows by inserting mean $\sqrt{\bar\alpha_T}x_0$ and covariance $(1-\bar\alpha_T)I_d$ into the Gaussian KL formula. As $\bar\alpha_T\to0$, the expression tends to zero for fixed $x_0$. At finite $T$, it generally remains positive. For $d=1$, $x_0=2$, and $\bar\alpha_T=0.01$, it is approximately $0.020025$. This term is independent of $\theta$ for a fixed schedule and fixed prior, so omitting it during parameter optimization differs from claiming that it vanishes exactly.

At the clean endpoint, the conditional $q(x_0\mid x_1,x_0)$ is a point mass. It cannot be inserted into an ordinary positive-variance Gaussian KL calculation. Instead choose a continuous Gaussian decoder with variance $\sigma_1^2>0$. Its negative log density is

$$-\log p_\theta(x_0\mid x_1)
=\frac{\|x_0-\mu_\theta(x_1,1)\|^2}{2\sigma_1^2}
+\frac d2\log(2\pi\sigma_1^2).$$

The quadratic has a positive sign because this is a negative log density. The decoder is a modeling choice. Discrete pixel likelihoods require an appropriate discrete or discretized distribution, as discussed in [Ho, Jain, and Abbeel's DDPM paper](https://arxiv.org/abs/2006.11239). An algebraic resemblance between endpoint reconstruction and intermediate losses does not make their probability measures identical.

## 2. Why predicting noise is a reparameterization [19:14](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=1154s)

For $t\ge2$, Lecture 11 derived the analytic posterior mean

$$\tilde\mu_t(x_t,x_0)
=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.$$

To see what a network must learn, solve the forward marginal for the clean input:

$$x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\varepsilon}
{\sqrt{\bar\alpha_t}}.$$

Substitute this expression into $\tilde\mu_t$ and use
$\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$. The coefficient of $x_t$ simplifies because
$\beta_t+\alpha_t(1-\bar\alpha_{t-1})=1-\bar\alpha_t$. Therefore

$$\tilde\mu_t=
\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon\right).$$

All coefficients are known. Parameterize the reverse mean by replacing the sampled noise with a predictor $\varepsilon_\theta(x_t,t)$ of the same dimension:

$$\mu_\theta=
\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\varepsilon_\theta(x_t,t)\right).$$

Subtracting cancels the known $x_t$ term. With fixed reverse variance, the part of the Gaussian KL that depends on the mean becomes

$$w_t\|\varepsilon-\varepsilon_\theta(x_t,t)\|^2,\qquad
w_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}.$$

If reverse and analytic posterior covariances differ but are fixed, their remaining KL contribution is independent of the mean predictor. Learned variances require retaining the corresponding variance-dependent terms.

### Equal-looking losses can train different models

The simplified DDPM loss averages unweighted noise errors. The variational objective weights them by $w_t$. These are different objectives when one network shares parameters across timesteps. For a toy shared scalar prediction $c$, suppose two deterministic noise targets are $0$ and $2$. Equal weights give minimizer $c=1$; weights $9$ and $1$ give $c=0.2$, obtained by setting $2[9c+(c-2)]=0$. Removing a timestep-dependent coefficient is an empirical design decision, not cancellation of a global constant. [DDPM](https://arxiv.org/abs/2006.11239) provides the primary formulation for this distinction.

Nor should the network be expected to recover every sampled noise vector exactly. For a fixed noisy observation $x_t$, multiple clean images and noise realizations may be compatible. Conditional squared-error decomposition shows that the optimal predictor is

$$\varepsilon^*(x_t,t)=\mathbb E[\varepsilon\mid x_t,t].$$

For any prediction $a$, expand $\|\varepsilon-a\|^2$ around this conditional mean. The cross term has conditional expectation zero, leaving irreducible conditional variance plus $\|\varepsilon^*-a\|^2$. Zero prediction error requires a degenerate conditional distribution, not merely successful optimization.

An instructive research extension connects this predictor to the score, the spatial gradient of the noisy log density. Differentiating the Gaussian corruption kernel and averaging over the conditional clean data gives

$$\nabla_x\log q_t(x)
=-\frac{\mathbb E[\varepsilon\mid x_t=x]}{\sqrt{1-\bar\alpha_t}}.$$

Thus noise prediction estimates a scaled score. Differentiation under the integral requires regularity and integrability. [Song et al.](https://arxiv.org/abs/2011.13456) develops the score-based stochastic-differential-equation framework. The identity explains a shared mathematical object; it does not make every numerical sampler interchangeable.

## 3. One training example and one reverse step [26:52](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=1612s)

A training iteration samples a clean image, a timestep, and a fresh standard Gaussian vector. Construct $x_t$ directly from the marginal; simulating all earlier steps is unnecessary. Evaluate the predictor and backpropagate the squared noise error. Fresh noise for independently sampled training examples is compatible with the correlated effective noises along a single complete forward trajectory.

For uniform $t$, the stochastic loss estimates $T^{-1}\sum_t\mathbb E[\ell_t]$. To estimate that same average using timestep probabilities $p_t>0$, use $\ell_t/(Tp_t)$. Its expectation is $\sum_t p_t\mathbb E[\ell_t]/(Tp_t)$, which cancels the sampling distribution. Sampling changes without this correction change the intended objective.

Consider a scalar image coordinate $x_0=2$, $\bar\alpha_t=0.64$, and sampled noise $-1$. The noisy coordinate is $1$. A prediction $-0.5$ has squared error $0.25$, and implies clean estimate

$$\hat x_0=\frac{1-0.6(-0.5)}{0.8}=1.625.$$

At generation time the clean input is unavailable. Begin from the terminal prior and repeatedly compute the reverse mean, then add $\sigma_t z$ for fresh $z\sim\mathcal N(0,I_d)$ according to the chosen sampler. Confusing variance with standard deviation changes the transition distribution.

The following deterministic kernel checks both reconstruction and a reverse mean. Inputs are equal-shaped finite NumPy arrays and scalar schedule factors in $(0,1)$. Outputs are new arrays; there is no mutation, random sampling, file access, or training. It rejects invalid inputs. It intentionally leaves final-step likelihood and sampler choices to the caller.

```python
import numpy as np


def diffusion_estimates(xt, noise_prediction, beta, alpha_bar):
    xt = np.asarray(xt, dtype=float)
    prediction = np.asarray(noise_prediction, dtype=float)
    if xt.shape != prediction.shape or not xt.size:
        raise ValueError("Input and prediction shapes must match")
    if not np.isfinite(xt).all() or not np.isfinite(prediction).all():
        raise ValueError("Arrays must be finite")
    if not np.isfinite([beta, alpha_bar]).all():
        raise ValueError("Schedule must be finite")
    if not (0 < beta < 1 and 0 < alpha_bar < 1):
        raise ValueError("Schedule factors must lie strictly inside (0, 1)")
    clean = (xt - np.sqrt(1 - alpha_bar) * prediction) / np.sqrt(alpha_bar)
    mean = (
        xt - beta * prediction / np.sqrt(1 - alpha_bar)
    ) / np.sqrt(1 - beta)
    return clean, mean


clean, reverse_mean = diffusion_estimates([1], [-0.5], 0.2, 0.64)
assert np.allclose(clean, [1.625])
assert np.allclose(reverse_mean, [(7 / 6) / np.sqrt(0.8)])
assert np.isclose((-1 - (-0.5)) ** 2, 0.25)
```

For our inspection service, these checks verify algebra, not image quality or downstream diagnostic validity. Long reverse chains cost many network evaluations. A diffusion model's decomposition makes each local prediction better specified, but accumulated approximation and discretization error remain.

## 4. A pretrained representation changes the statistical task [36:51](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=2211s)

Pretraining fits parameters on a broad source dataset; adaptation uses the resulting model for another task. The [foundation-model report](https://arxiv.org/abs/2108.07258) characterizes models trained on broad data at scale and adaptable to many downstream tasks. Breadth is an intended use pattern, not a theorem that every useful feature will emerge. Self-supervised data still contain targets derived from the observations; “unlabeled” means that task-specific human labels were not supplied.

For the inspection case, let $\phi_\theta:\mathcal X\to\mathbb R^m$ map an image to an $m$-dimensional representation. A learned vector is also called an embedding. The architecture specifies how to interpret $\theta$; a checkpoint is meaningful only with compatible architecture and preprocessing. Let $(x_i,y_i)_{i=1}^n$ be labeled downstream examples, with a scalar target $y_i$. A linear head predicts

$$f_{w,\theta}(x)=w^\top\phi_\theta(x),\qquad w\in\mathbb R^m.$$

A constant representation coordinate can supply an intercept. Linear probing holds pretrained $\hat\theta$ fixed and minimizes

$$L(w,\hat\theta)=\frac1{2n}\sum_{i=1}^n
\big(w^\top\phi_{\hat\theta}(x_i)-y_i\big)^2.$$

Writing $\Phi$ for the matrix with rows $\phi_{\hat\theta}(x_i)^\top$, the problem is ordinary least squares in $\Phi$. The predictor may be highly nonlinear in the original image while remaining linear in $w$.

### Information present is not information linearly accessible

Suppose an original scalar measurement takes values $x=-1,0,1$, with target $y=x^2$. An affine predictor in $x$ cannot fit all three: the middle point forces intercept zero, while the endpoints require contradictory slopes. Features $\phi(x)=(1,x,x^2)$ permit the exact head $w=(0,0,1)$. Representation learning aims to make relevant structure accessible to a simple readout.

Conversely, if $\phi(x_a)=\phi(x_b)$ for two examples with different required labels, no deterministic head can recover both. Pretraining can discard task-relevant information. A successful linear probe establishes decodability under a tested distribution; it does not establish that the original network uses the feature causally. A failed linear probe need not establish absence of information, because a nonlinear readout may succeed.

Few-shot and zero-shot specify the number of task examples available under a protocol. They do not by themselves specify whether weights change. A few-shot prompt and few-shot gradient training are different adaptation procedures. Claims that supervised downstream data are no longer used should not be read as universal facts; the decision depends on task, evidence, and deployment requirements.

## 5. Fine-tuning changes the representation as well as the head [54:49](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=3289s)

Full fine-tuning minimizes $L(w,\theta)$ over both objects, initialized from the pretrained representation. Unlike a frozen linear probe, the problem is generally nonconvex. Pretraining can influence the trajectory and the eventual function even when the written downstream objective is unchanged.

The mechanism is visible in the gradient. For one example, let residual $r=w^\top\phi_\theta(x)-y$ and representation Jacobian $J_\theta(x)=\partial\phi_\theta(x)/\partial\theta$. With loss $r^2/2$,

$$\nabla_\theta L=rJ_\theta(x)^\top w.$$

A poorly matched initial head sends a correspondingly poorly matched update into the pretrained features. Linear-probe-then-fine-tune, abbreviated LP-FT, first fits $w$ with $\theta=\hat\theta$, then jointly updates both from that fitted head.

For a scalar illustration, take $\phi_\theta(x)=\theta x$, $\hat\theta=1$, training input $x=1$, and target $y=2$. A random head $w=-1$ gives residual $-3$ and feature gradient $(-3)(-1)=3$; one step of size $0.1$ moves $\theta$ down to $0.7$. The fitted probe is $w=2$, whose residual and immediate feature gradient are zero. This calculation illustrates early feature disturbance. It does not prove which method generalizes better, and a zero residual can also halt useful adaptation.

[Kumar et al.](https://arxiv.org/abs/2202.10054) provides a research extension: analysis and experiments show settings where full fine-tuning improves in-distribution accuracy while damaging performance under distribution shift; LP-FT can mitigate this. The result depends on the pretrained features and shift. It is not a universal ranking of methods.

For inspection deployment, evaluate on different buildings, cameras, and time periods, not only random photographs from the same inspection. Compare in-distribution accuracy, shifted accuracy, label cost, and computational cost separately. Better training fit cannot choose among these dimensions without an explicit operational objective.

## 6. LoRA restricts the update, not the base computation [62:26](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=3746s)

Suppose a layer computes $Wx$ with pretrained matrix $W_0\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$. Low-rank adaptation freezes $W_0$ and learns

$$W=W_0+sBA,\quad
B\in\mathbb R^{d_{\rm out}\times r},\quad
A\in\mathbb R^{r\times d_{\rm in}}.$$

Here $r$ is the chosen rank limit and $s$ a fixed scale. Since every column of $BA$ lies in the column space of $B$, $\operatorname{rank}(BA)\le r$. This restricts the change in $W$, not the rank of $W$ itself. The two factors need not be transposes, so the update need not be symmetric.

The trainable parameter count is $r(d_{\rm in}+d_{\rm out})$, versus $d_{\rm in}d_{\rm out}$ for a full matrix. For a $4096\times4096$ layer and $r=8$, this is $65{,}536$ instead of $16{,}777{,}216$, a factor of 256. Parameter count is not the intrinsic dimension of the rank-$r$ manifold: transformations $B\mapsto BR$, $A\mapsto R^{-1}A$ for invertible $R$ leave the product unchanged.

### Initialization is a gradient condition

Let $G=\partial L/\partial W$ be the upstream matrix gradient. Matrix differentials give

$$\nabla_B L=sGA^\top,\qquad
\nabla_A L=sB^\top G.$$

Initializing $B=0$ with random nonzero $A$ preserves $W_0$ while allowing $B$ to move. Setting both factors to zero makes both gradients zero. The optimization is then stuck under ordinary first-order updates, even if the full-matrix gradient is nonzero.

The following check implements a two-dimensional layer with a rank-one update. It has no randomness or side effects and assumes the displayed shapes. It checks initialization, the first nonzero update, and exact merging into the base matrix; this is an algebraic experiment, not a training library.

```python
W0 = np.eye(2)
A = np.array([[1.0, 0.0]])
B = np.zeros((2, 1))
x = np.array([1.0, 2.0])
target = np.array([2.0, 2.0])
residual = (W0 + B @ A) @ x - target
G = np.outer(residual, x)
dB, dA = G @ A.T, B.T @ G
assert np.allclose(dB, [[-1], [0]])
assert np.allclose(dA, 0)
B = B - 0.5 * dB
unmerged = W0 @ x + B @ (A @ x)
merged = (W0 + B @ A) @ x
assert np.allclose(unmerged, merged)
assert np.allclose(merged, [1.5, 2])
assert np.isclose(0.5 * np.sum((merged - target) ** 2), 0.125)
```

[Hu et al.'s LoRA paper](https://arxiv.org/abs/2106.09685) supplies the primary method and empirical evidence. Fewer trainable weights reduce gradient and optimizer-state storage. The frozen weights still occupy memory, and the forward product $W_0x$ still runs. Backpropagation through earlier trainable layers can still require $W_0^\top$ operations. Activation memory also remains. Exact savings depend on precision, selected layers, optimizer, and parallelism; a universal “six times” or “twelve times” memory multiplier is not justified.

For one fixed adapter, merging $sBA$ into $W_0$ recovers an ordinary dense layer, up to numerical precision. Serving many tasks can instead share the base and store separate small adapters. This is useful for multiple inspection clients, but neither low rank nor low storage guarantees isolation of task behavior or improved accuracy. Sharding parameters, gradients, and optimizer states is a separate systems choice; it should not be conflated with ordinary replicated data parallelism.

## Limitations and design judgment

Diffusion training and downstream adaptation solve different problems. Noise-prediction accuracy does not establish generative coverage, and visually plausible generation does not validate a safety-related classifier. A useful evaluation records what is held fixed: data split, schedule, architecture, base checkpoint, adaptation budget, and distribution shift. Choose a frozen probe when it answers the task well with little data; use fuller adaptation when evidence shows that the representation must change. Treat LoRA rank and layer selection as hypotheses to validate.

## Exercises

1. Compute the terminal Gaussian KL for $d=1$, $x_0=0$, and $\bar\alpha_T=0.5$. Why can this term be omitted in optimizing a fixed-schedule predictor without calling it zero?
2. Two timesteps have expected losses $2$ and $6$. Sample them with probabilities $0.75$ and $0.25$. Construct an unbiased estimate of their equally weighted average and calculate the uncorrected expectation.
3. For the scalar denoising example, suppose the noise predictor is exact. Recover $x_0$. Explain why exact prediction is not generally attainable for all noisy observations.
4. Construct a representation that preserves $x$ but makes $y=x^2$ inaccessible to an affine probe on $x\in\{-1,0,1\}$. Then repair the representation.
5. A rank-one adapter has $A=B=0$ and a nonzero full-matrix gradient. Derive its factor gradients. Explain both a repair and a task for which rank one remains insufficient.

<details>
<summary>Fully worked solutions</summary>

1. The KL is $\tfrac12[-0.5-\log(0.5)]\approx0.096574$. It is positive because the terminal variance is $0.5$, not the prior's variance $1$. With fixed schedule and prior its derivative with respect to predictor parameters is zero. Constants can be removed from an optimization without changing the minimizers, while still contributing to the value of a likelihood bound.
2. The target is $(2+6)/2=4$. Divide an observed timestep loss by $2p_t$. The two possible corrected values are $4/3$ and $12$; their expectation is $0.75(4/3)+0.25(12)=4$. The uncorrected expectation is $0.75(2)+0.25(6)=3$, so it trains a different weighting.
3. The exact prediction is $-1$. Therefore $\hat x_0=[1-0.6(-1)]/0.8=2$. But when clean data are random, one noisy value may arise from several clean/noise pairs. Conditional variance remains, and the squared-loss optimum predicts the conditional mean of noise rather than each hidden realization.
4. Choose $\phi(x)=x$, which preserves the entire scalar input. An affine probe $a+bx$ must have $a=0$ at $x=0$, $b=1$ at $x=1$, and $b=-1$ at $x=-1$, a contradiction. Append $x^2$ to the representation. A head selecting that coordinate then fits exactly. Thus retained information and linear accessibility are different properties.
5. Since $\nabla_B L=sGA^\top$ and $\nabla_A L=sB^\top G$, both vanish. Make one factor random and the other zero to preserve the initial model while permitting a first update. This does not remove the rank restriction: if a layer must change by the two-dimensional identity matrix, a rank-one update cannot match it on every input because that target update has rank two.
</details>

## Primary sources

- Ho, Jain, and Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).
- Bommasani et al., [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258).
- Kumar et al., [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054).
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

# Lecture 5 — Generative classification, Gaussian geometry, and naive Bayes

Research extensions are identified through links to primary papers; numerical examples and executable checks are original teaching constructions.

An inspection service must decide which properties need an urgent visit. It has continuous measurements, such as moisture and temperature, and short reports containing words such as “crack” and “routine.” Logistic regression can predict the urgency label directly. A different strategy is to learn what measurements and reports look like in each class, then invert those models with Bayes' rule. This chapter develops that strategy and identifies the assumptions that make it economical.

The recording moves from generative classification through Gaussian geometry and maximum likelihood to naive Bayes. The inspection service, numerical datasets, code, and independence counterexample below are original teaching constructions. Links mark the corresponding recorded discussions; researched extensions are identified where introduced. We use $n$ observations, $d$ features, a column vector $x\in\mathbb R^d$, and a binary label $Y$, with $Y=1$ denoting urgency.

## 1. Learning a joint distribution to answer a conditional question [00:00](https://www.youtube.com/watch?v=zRdE8A4UZes&t=0s)

A discriminative model specifies $P(Y\mid X)$ directly. A generative classifier specifies a joint distribution, commonly through the factorization $p(x,y)=p(x\mid y)P(y)$. Here “generative” means that the model can sample a class and then sample observations conditional on that class. It does not imply a neural network, an image generator, or a particular training scale.

Let $\pi=P(Y=1)$ be the class prior, and let $f_k(x)=p(x\mid Y=k)$ be the class-conditional density. Bayes' rule gives

$$P(Y=1\mid x)=\frac{\pi f_1(x)}
{\pi f_1(x)+(1-\pi)f_0(x)}.$$

This expression separates three questions. How common is urgency? How compatible is this observation with urgent cases? How compatible is it with ordinary cases? A large value of $f_1(x)$ alone does not answer the decision question. We need the competing density and the prior.

The odds form makes the separation particularly clear:

$$\frac{P(Y=1\mid x)}{P(Y=0\mid x)}
=\frac{f_1(x)}{f_0(x)}\cdot\frac{\pi}{1-\pi}.$$

In ordinary notation, posterior odds equal likelihood ratio times prior odds. If urgent cases have prior probability $0.1$ and a measurement is four times as likely in urgent cases, the posterior odds are $4/9$, giving posterior probability $4/13\approx0.308$. Evidence favoring a rare event need not make that event more likely than its alternative.

For continuous features, $f_k(x)$ is a density, not the probability that $X$ equals exactly $x$. It can exceed one, and its numerical value changes when measurement units change. Probabilities come from integrating densities over regions. A common invertible change of coordinates introduces the same Jacobian factor into both class densities, so their likelihood ratio remains unchanged. This is why the posterior is meaningful even when individual density values depend on units.

Classification also needs a loss function. Under equal costs, choose class 1 when its posterior exceeds $1/2$. If a missed urgent case costs $C_{\rm FN}$ and an unnecessary visit costs $C_{\rm FP}$, with zero cost for correct decisions, the expected losses are $C_{\rm FN}p$ for predicting 0 and $C_{\rm FP}(1-p)$ for predicting 1. Therefore choose 1 when

$$p>\frac{C_{\rm FP}}{C_{\rm FP}+C_{\rm FN}}.$$

Changing the action threshold is not the same as changing the model's posterior. Preserving that distinction allows an inspection service to use the same calibrated probability estimates under different staffing constraints or safety costs.

### Prior shift is a concrete form of distribution change

Suppose the class-conditional densities remain fixed but urgency becomes more common in a new region. Training on one prior and deploying under another changes the posterior even though the feature mechanism within each class is unchanged. Under this explicit assumption, the new log odds equal the old log odds plus

$$\log\frac{\pi_{\rm new}}{1-\pi_{\rm new}}
-\log\frac{\pi_{\rm old}}{1-\pi_{\rm old}}.$$

This correction is derived from Bayes' rule, not a general cure for distribution shift. If new construction materials also change moisture measurements within each class, adjusting only the prior leaves a misspecified likelihood. The generative factorization is useful precisely because it exposes which part of the model is being held fixed.

## 2. A Gaussian is a geometry, not just a bell curve [09:00](https://www.youtube.com/watch?v=zRdE8A4UZes&t=540s)

For a random vector $X$, define its mean $\mu=\mathbb E[X]$ and covariance

$$\Sigma=\mathbb E[(X-\mu)(X-\mu)^\top].$$

The diagonal entry $\Sigma_{jj}$ is the variance of feature $j$; an off-diagonal entry measures linear co-movement. Covariance matrices are symmetric positive semidefinite because, for every vector $v$,

$$v^\top\Sigma v
=\mathbb E[(v^\top(X-\mu))^2]\ge0.$$

Positive semidefinite does not mean that every matrix entry is positive. A negative covariance is entirely legitimate. For a nonsingular Gaussian density on all of $\mathbb R^d$, we need positive definiteness, meaning strict positivity for every nonzero $v$.

The multivariate Gaussian density is

$$f(x)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left[-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right].$$

Here $|\Sigma|$ is the determinant. Its square root adjusts the normalization for the volume occupied by the distribution. The quadratic form determines the shape of equal-density contours. Calling $\Omega=\Sigma^{-1}$ the precision matrix, the squared Mahalanobis distance is $(x-\mu)^\top\Omega(x-\mu)$.

To see the geometry, diagonalize $\Sigma=U\Lambda U^\top$, where $U$ is orthogonal and $\Lambda$ contains positive eigenvalues $\lambda_j$. In rotated coordinates $r=U^\top(x-\mu)$, the quadratic form becomes $\sum_jr_j^2/\lambda_j$. A contour at squared distance $c$ therefore has semiaxis lengths $\sqrt{c\lambda_j}$, not $\lambda_j$. Variance has squared measurement units; an axis length has ordinary measurement units.

For the inspection service, take

$$\Sigma=\begin{pmatrix}2&1\\1&2\end{pmatrix}.$$

The unit vectors along $(1,1)$ and $(1,-1)$ have eigenvalues 3 and 1. A joint increase in both standardized sensors is relatively common; a disagreement between them is less common. The unit-Mahalanobis contour has semiaxes $\sqrt3$ and 1. A Euclidean distance rule would miss this directional distinction.

Whitening maps $x$ to $z=\Lambda^{-1/2}U^\top(x-\mu)$. Then $\operatorname{Cov}(z)=I$ and the Mahalanobis quadratic becomes $\|z\|^2$. Whitening is a coordinate transformation, not evidence that the original sensors were physically independent.

### Marginal independence and conditional independence are different

The spoken discussion of inverse covariance needs a precise distinction. For a jointly Gaussian vector, $\Sigma_{ij}=0$ means features $i$ and $j$ are marginally independent. By contrast, $\Omega_{ij}=0$ means they are conditionally independent given all remaining features. These statements concern different experiments.

An original counterexample makes the distinction visible. Let $Z,\epsilon_1,\epsilon_2$ be independent standard normal variables, and define $X_1=Z+\epsilon_1$, $X_2=Z+\epsilon_2$. The pair has covariance 1 because both respond to the same latent environmental condition. Yet, after conditioning on $Z$, their remaining noises are independent. For the ordered vector $(X_1,X_2,Z)$,

$$\Sigma=\begin{pmatrix}2&1&1\\1&2&1\\1&1&1\end{pmatrix},
\qquad
\Omega=\begin{pmatrix}1&0&-1\\0&1&-1\\-1&-1&3\end{pmatrix}.$$

Thus $\Sigma_{12}\ne0$ while $\Omega_{12}=0$. Conditioning removes the common cause's uncertainty.

The precision criterion also follows directly from the Gaussian exponent. Holding all other coordinates fixed leaves a quadratic in $x_i,x_j$ with a cross-term proportional to $\Omega_{ij}x_ix_j$. When that coefficient is zero, the conditional density factors into a function of $x_i$ times a function of $x_j$. This graphical-model connection extends the lecture's geometry; [Wainwright and Jordan's monograph](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf) develops the broader relationship between exponential families and conditional independence. Outside the Gaussian family, zero covariance does not generally imply independence.

## 3. Fitting Gaussian discriminant analysis [24:00](https://www.youtube.com/watch?v=zRdE8A4UZes&t=1440s)

Gaussian discriminant analysis, abbreviated GDA, makes three modeling commitments:

$$Y\sim\operatorname{Bernoulli}(\pi),\qquad
X\mid Y=0\sim\mathcal N(\mu_0,\Sigma),\qquad
X\mid Y=1\sim\mathcal N(\mu_1,\Sigma).$$

The two classes have different means but share one covariance matrix. The shared covariance says that the pattern of within-class variability is the same in both classes. It does not say that the unconditional covariance of the entire population equals $\Sigma$: mixing populations with different means adds between-class variability.

For independent labeled observations, the joint log likelihood separates into a prior term and Gaussian terms. Write $n_k$ for the number of observations in class $k$, and define residuals $r_i=x_i-\mu_{y_i}$. Up to constants independent of the parameters,

$$\ell=n_1\log\pi+n_0\log(1-\pi)
-\frac n2\log|\Sigma|
-\frac12\sum_i r_i^\top\Sigma^{-1}r_i.$$

Differentiating the prior term gives $n_1/\pi-n_0/(1-\pi)=0$, hence $\hat\pi=n_1/n$. For a class mean, the gradient is $\Sigma^{-1}\sum_{i:y_i=k}(x_i-\mu_k)$. When both classes are represented and $\Sigma$ is invertible, the stationary point is the class sample mean:

$$\hat\mu_k=\frac1{n_k}\sum_{i:y_i=k}x_i.$$

The covariance derivative is easier in precision coordinates. Let $S=\sum_i r_ir_i^\top$. The relevant objective is

$$\ell(\Omega)=\frac n2\log|\Omega|-\frac12\operatorname{tr}(\Omega S).$$

Its derivative is $(n/2)\Omega^{-1}-S/2$. Setting it to zero yields $\hat\Sigma=S/n$. Substituting the fitted means gives

$$\hat\Sigma=\frac1n\sum_i
(x_i-\hat\mu_{y_i})(x_i-\hat\mu_{y_i})^\top.$$

The denominator is $n$ because this is maximum likelihood. A pooled unbiased covariance estimator under the correctly specified model uses $n-2$ when two class means are estimated. Neither denominator is a typographical variant of the other: the estimators optimize different statistical criteria.

For a one-dimensional worked example, use ordinary cases $x=-1,1$ and urgent cases $x=1,3$. The fitted means are 0 and 2, the prior is $1/2$, and all four squared within-class residuals are 1. Thus the maximum-likelihood variance is 1; the unbiased pooled estimate is 2. Their decision boundaries happen to coincide under equal priors, but their probability estimates differ because the scale of the log odds changes.

### When the fitted covariance is singular

Within each class, the residual vectors sum to zero. Consequently, the pooled residual matrix has rank at most $n-2$. If $d>n-2$, the empirical covariance cannot be invertible. Even below this threshold, collinear features can cause singularity. An inverse routine failing here is reporting a structural problem, not merely an inconvenient implementation detail.

One remedy is covariance regularization, for example replacing $\hat\Sigma$ by $\hat\Sigma+\lambda I$ with $\lambda>0$ after choosing meaningful feature scales. This changes the estimator and suppresses extreme precision in directions with little estimated variance. Another is a diagonal covariance model, which removes within-class correlations. These remedies make different assumptions and should be evaluated on held-out data.

The parameter count helps explain the tradeoff. Binary GDA with shared full covariance has $2d+d(d+1)/2+1$ free parameters. Binary logistic regression with an intercept has $d+1$. Generative fitting is not automatically a smaller model in parameter count; its advantage, when present, comes from how its structural assumptions constrain the estimation problem.

## 4. Why a Gaussian generative model produces logistic probabilities [45:00](https://www.youtube.com/watch?v=zRdE8A4UZes&t=2700s)

Posterior log odds are

$$a(x)=\log\frac{\pi}{1-\pi}+\log f_1(x)-\log f_0(x).$$

With shared covariance, the two Gaussian normalization constants cancel. Expand the two quadratic forms. The terms $x^\top\Sigma^{-1}x$ cancel as well, leaving

$$a(x)=w^\top x+b,\qquad
w=\Sigma^{-1}(\mu_1-\mu_0),$$

$$b=\log\frac{\pi}{1-\pi}
-\frac12\left(\mu_1^\top\Sigma^{-1}\mu_1
-\mu_0^\top\Sigma^{-1}\mu_0\right).$$

Converting odds back into a probability gives $P(Y=1\mid x)=\sigma(w^\top x+b)$, where $\sigma(a)=1/(1+e^{-a})$. The equal-cost decision boundary is a hyperplane. This is the algebraic reason shared-covariance GDA is also called linear discriminant analysis in classification settings.

Return to means 0 and 2 with variance 1, but deploy at prior $\pi=0.1$. The log odds are $2x-2-\log9$. The boundary moves from $x=1$ to

$$x_*=1+\frac12\log9\approx2.0986.$$

At $x=2$, the urgent-class mean, the posterior is only $\sigma(2-\log9)\approx0.4509$. Being at one class's mean does not override a strong prior favoring the other class.

The following computation accepts specified class means, a symmetric positive definite covariance, a prior strictly between zero and one, and a finite observation vector. It returns posterior log odds. It has no file or network effects. Cholesky factorization rejects a non-positive-definite covariance; solving linear systems avoids forming an explicit inverse.

```python
import numpy as np


def gda_log_odds(x, mean0, mean1, covariance, prior):
    x, mean0, mean1 = [
        np.asarray(v, dtype=float) for v in (x, mean0, mean1)
    ]
    covariance = np.asarray(covariance, dtype=float)
    if x.ndim != 1 or mean0.shape != x.shape or mean1.shape != x.shape:
        raise ValueError("Observation and means must be matching vectors")
    if not x.size or covariance.shape != (x.size, x.size):
        raise ValueError("Covariance must match a nonempty observation")
    if not all(np.isfinite(v).all() for v in (x, mean0, mean1, covariance)):
        raise ValueError("All inputs must be finite")
    if not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("Prior must be strictly between zero and one")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("Covariance must be symmetric")
    np.linalg.cholesky(covariance)
    weight = np.linalg.solve(covariance, mean1 - mean0)
    intercept = np.log(prior) - np.log1p(-prior)
    intercept -= 0.5 * (mean1 + mean0) @ weight
    return float(x @ weight + intercept)


boundary = 1 + np.log(9) / 2
assert abs(gda_log_odds([boundary], [0], [2], [[1]], 0.1)) < 1e-12
assert np.isclose(gda_log_odds([2], [0], [2], [[1]], 0.1),
                  2 - np.log(9))
```

### The same functional form does not imply the same estimator

GDA induces a logistic conditional model, but fitting the joint likelihood is not equivalent to fitting conditional logistic likelihood. GDA spends statistical effort explaining the feature distribution within classes. Logistic regression uses only the conditional likelihood of labels given observed features. On finite or misspecified data, they generally produce different coefficients.

The converse also fails: logistic conditional probabilities do not imply Gaussian class-conditional densities. Choose any feature marginal $p(x)$ and define $P(Y=1\mid x)=\sigma(w^\top x+b)$. Their product specifies a joint model whose class-conditional distributions need not be Gaussian.

[Ng and Jordan's comparison of generative and discriminative classifiers](https://proceedings.neurips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) studies differing finite-sample and asymptotic behavior, including naive Bayes versus logistic regression. Its message is a tradeoff under specified models, not a universal rule that generative classification wins with little data. Correctness of assumptions, dimension, regularization, and the chosen performance metric all matter.

If each class has a different covariance $\Sigma_k$, the cancellation of quadratic terms no longer occurs. The log odds contain

$$-\frac12x^\top(\Sigma_1^{-1}-\Sigma_0^{-1})x
-\frac12\log\frac{|\Sigma_1|}{|\Sigma_0|}$$

in addition to linear and constant terms. This is quadratic discriminant analysis. A curved boundary can represent more structure, but requires estimating more parameters and can create surprising classification regions far from the observations.

Neither linear nor quadratic posterior confidence is an out-of-distribution detector. A posterior normalizes over the classes in the model. Both densities can be tiny while their ratio is enormous. An unfamiliar measurement can therefore receive an extreme posterior even though neither class model describes it well.

## 5. From generative classification to generative pretraining [58:00](https://www.youtube.com/watch?v=zRdE8A4UZes&t=3480s)

The broader connection is that learning observation structure can support downstream prediction. A language model learns a distribution over sequences; an image model learns a distribution over images or latent representations. Their learned features can then help supervised tasks. This is a conceptual bridge to later lectures, not a claim that a modern pretrained model satisfies GDA's Gaussian assumptions.

There is a useful exact extension of the Gaussian cancellation argument. Suppose class-conditional densities belong to an exponential family with the same base measure and sufficient statistic:

$$p(x\mid Y=k)=b(x)\exp\{\eta_k^\top T(x)-A(\eta_k)\}.$$

Then posterior log odds are

$$\log\frac{\pi}{1-\pi}
+(\eta_1-\eta_0)^\top T(x)-A(\eta_1)+A(\eta_0).$$

They are affine in $T(x)$. They are affine in the raw observation $x$ only when the relevant statistic is linear in $x$. For a Gaussian with class-specific covariance, $T(x)$ includes quadratic products, recovering the quadratic discriminant. Thus the link between generative and discriminative models is mediated by sufficient statistics, not by a blanket assertion that exponential-family models always have linear boundaries in their original inputs.

For our inspection service, a pretrained text representation could replace hand-built word indicators, followed by a small discriminative classifier. Whether that improves performance is an empirical question requiring the same split discipline as any other estimator. The fact that the representation was learned generatively does not exempt its downstream predictions from calibration, distribution-shift, or leakage checks.

## 6. Naive Bayes trades dependence modeling for statistical economy [73:00](https://www.youtube.com/watch?v=zRdE8A4UZes&t=4380s)

To model reports, choose a fixed vocabulary of $d$ words and encode $x_j=1$ if word $j$ appears, otherwise 0. This is a multi-hot representation: many entries can be one. A one-hot vector would identify exactly one category and cannot describe a report containing several vocabulary words.

An unrestricted distribution over all binary word-presence vectors has $2^d$ possible outcomes and $2^d-1$ free probabilities per class. Naive Bayes replaces that joint table with the assumption that the features are conditionally independent given the class:

$$P(X=x\mid Y=k)
=\prod_{j=1}^d\theta_{jk}^{x_j}(1-\theta_{jk})^{1-x_j},
\qquad
\theta_{jk}=P(X_j=1\mid Y=k).$$

There are now only $2d+1$ parameters for two classes, including the prior. Conditional independence does not imply marginal independence: mixing urgent and ordinary reports can correlate words even when the model makes them independent within each class.

The maximum-likelihood estimate of $\theta_{jk}$ is the fraction of class-$k$ reports containing word $j$. If that fraction is zero or one, a single contrary feature can force a class likelihood to zero. With an independent uniform $\operatorname{Beta}(1,1)$ prior on each Bernoulli parameter, the posterior predictive probability is

$$\tilde\theta_{jk}=\frac{c_{jk}+1}{n_k+2},$$

where $c_{jk}$ counts reports in class $k$ containing the word. This is Laplace smoothing. It is a posterior predictive mean, not the maximum-likelihood estimate. The denominator has 2 because there are two Bernoulli outcomes; a multinomial count model uses a different normalization involving vocabulary size.

### Both present and absent words contribute evidence

Let the vocabulary be “crack” and “routine.” Suppose urgent reports have presence probabilities $(0.8,0.2)$ and ordinary reports have $(0.2,0.6)$, with urgency prior $0.1$. For a report containing “crack” and not “routine,” the urgent likelihood is $0.8(1-0.2)=0.64$, while the ordinary likelihood is $0.2(1-0.6)=0.08$. Bayes' rule gives

$$P(Y=1\mid x)=\frac{0.1(0.64)}
{0.1(0.64)+0.9(0.08)}=\frac8{17}\approx0.4706.$$

The absence of “routine” is part of the evidence. Omitting absence terms would silently switch away from the specified Bernoulli model.

The next block evaluates these fixed parameters in log space. It is a numerical check of the model, not a text tokenizer or a training pipeline. All probabilities in this example lie strictly inside $(0,1)$, as smoothing would ensure for finite counts.

```python
prior = 0.1
features = np.array([1, 0])
theta = np.array([[0.2, 0.6], [0.8, 0.2]])
log_joint = np.log([1 - prior, prior])
log_joint += (
    features * np.log(theta) + (1 - features) * np.log1p(-theta)
).sum(axis=1)
log_normalizer = np.logaddexp(log_joint[0], log_joint[1])
posterior = np.exp(log_joint - log_normalizer)
assert np.isclose(posterior[1], 8 / 17)
assert np.isclose(posterior.sum(), 1)
```

Now duplicate the “crack” feature into a second column with identical values. A naive Bayes fit treats the duplicate as another independent piece of evidence. In our example, it multiplies posterior odds by another factor of $0.8/0.2=4$, moving probability from $8/17$ to $32/41\approx0.7805$. Nothing new was observed. The rise in confidence is an artifact of a false conditional-independence assumption.

This failure explains why naive Bayes can classify reasonably while producing poor probabilities. Several correlated indicators may preserve the winning class while exaggerating the evidence for it. Negation creates another problem: word presence alone cannot distinguish “crack detected” from “no crack detected” without additional features. Better representations address linguistic structure; smoothing addresses zero counts. They solve different problems.

## Exercises

1. **Derive.** For shared variance $\sigma^2$ and scalar means $\mu_1>\mu_0$, derive the equal-cost decision threshold as a function of the prior. Explain its direction of movement as urgency becomes rarer.
2. **Compute.** Using ordinary observations $-1,1$ and urgent observations $1,3$, find the maximum-likelihood prior, means, and shared variance. Compute the posterior at $x=1$ and $x=2$.
3. **Counterexample.** In the common-cause construction $X_1=Z+\epsilon_1$, $X_2=Z+\epsilon_2$, verify the displayed covariance and precision matrices and explain which independence statement each relevant zero does or does not establish.
4. **Derive and compute.** A word appears in none of three urgent reports. Find its Bernoulli MLE and its Laplace-smoothed posterior predictive probability. Explain why a multinomial vocabulary denominator is incorrect here.
5. **Design.** An inspection classifier is highly confident on a new type of building absent from training. Propose an evaluation that distinguishes prior shift, feature shift within classes, duplicated evidence, and changes in decision cost.

<details><summary>Solutions</summary>

1. Expanding the two scalar Gaussian log densities gives $a(x)=(\mu_1-\mu_0)[x-(\mu_1+\mu_0)/2]/\sigma^2+\log[\pi/(1-\pi)]$. Setting this to zero yields $x_*=(\mu_1+\mu_0)/2+\sigma^2\log[(1-\pi)/\pi]/(\mu_1-\mu_0)$. As $\pi$ falls, the logarithm increases, so a larger observation is needed to classify urgency. This movement follows from the prior; it does not require moving either class mean. Unequal costs would set the log odds to $\log(C_{\rm FP}/C_{\rm FN})$ instead of zero.

2. There are two observations in each class, so $\hat\pi=1/2$, $\hat\mu_0=0$, and $\hat\mu_1=2$. The four squared residuals are all one, giving $\hat\sigma^2=4/4=1$. Therefore $a(x)=2x-2$. At $x=1$, the posterior is $1/2$; at $x=2$, it is $\sigma(2)\approx0.8808$. Using the unbiased variance 2 would give $\sigma(1)\approx0.7311$ at the second point, demonstrating that the variance convention affects probabilities even when the boundary stays fixed.

3. Each $X_j$ has variance 2, their covariance is $\operatorname{Var}(Z)=1$, and each has covariance 1 with $Z$. These facts produce the displayed $\Sigma$. Direct matrix multiplication verifies $\Sigma\Omega=I$. The nonzero $\Sigma_{12}$ establishes that $X_1$ and $X_2$ are not marginally independent in this Gaussian model. The zero $\Omega_{12}$ establishes conditional independence given $Z$, agreeing with the independent residual noises. It does not assert that their unconditional correlation vanishes.

4. The MLE is $0/3=0$. Under the uniform Beta prior, the predictive probability is $(0+1)/(3+2)=1/5$. The Bernoulli random variable has two outcomes, presence and absence, regardless of vocabulary size. A multinomial model instead distributes token mass across vocabulary categories and conditions on a different observation structure. Reusing its denominator would mix incompatible likelihoods.

5. Collect labeled examples from the new building population and compare class frequencies, class-conditional feature distributions, and calibration with the old population. A pure prior-shift hypothesis predicts stable class-conditional distributions and an intercept-only log-odds correction. Inspect feature construction and run an ablation that removes duplicate or strongly redundant indicators. Evaluate log loss and calibration as well as classification error, since confidence can fail before rankings do. Finally, compute decisions under explicit false-positive and false-negative costs; a changed threshold can be appropriate even when the probability model is unchanged. Hold out an evaluation set from all these model and threshold choices.

</details>

## Primary references and evidence boundaries

The [recording](https://www.youtube.com/watch?v=zRdE8A4UZes) supplies the chapter's topic order and model sequence. The [official CS229 notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) provide an earlier written treatment of GDA and naive Bayes. [Ng and Jordan (2001)](https://proceedings.neurips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) support the researched comparison of generative and discriminative estimation. [Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf) support the graphical-model extension. The inspection examples, explicit independence correction, calculations, code, and exercises are original exposition; the chapter does not claim that they appeared on the recording's slides.

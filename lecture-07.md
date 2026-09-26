# Lecture 7 — Neural networks as compositions of learned representations

Research extensions are identified through links to primary papers; numerical examples and executable checks are original teaching constructions.

The property service has reached the limit of choosing a fixed list of useful features by hand. Area, age, and inspection measurements interact, and a single affine prediction cannot express every relevant interaction. A neural network makes intermediate features trainable. The statistical task remains familiar: choose an output distribution or loss, fit parameters on training data, and evaluate on fresh data.

This chapter follows the recording's progression from general nonlinear supervised learning to dense layers, activation functions, residual connections, normalization, and a short introduction to convolution. Its numerical network and implementation are original constructions. The next chapter differentiates these computations; here the priority is to specify exactly what is computed and why its structure matters.

## 1. Changing the model does not change the learning problem [02:00](https://www.youtube.com/watch?v=fRM41w9jzQo&t=120s)

Let an input be $x\in\mathbb R^d$, a target be $y$, and all trainable parameters be collected in $\theta$. A model $f_\theta$ maps inputs to predictions. For independent training pairs $(x_i,y_i)$, empirical risk is

$$J(\theta)=\frac1n\sum_{i=1}^n\ell(f_\theta(x_i),y_i).$$

The loss $\ell$ describes what counts as an error; the architecture describes which functions the model can represent. These are separate decisions. A network producing a scalar can predict a conditional mean with squared loss. A network producing class logits can define a categorical distribution with softmax and cross entropy.

For fixed-variance Gaussian regression,

$$Y\mid x\sim\mathcal N(f_\theta(x),\sigma^2),$$

the negative log likelihood differs from squared error only by a positive scaling factor and a parameter-independent constant. The derivation from Lecture 3 still applies even if $f_\theta$ is a complicated network. However, a scaling factor can change a gradient step at a fixed learning rate; equivalent minimizers do not imply identical optimization trajectories.

For $K$ mutually exclusive classes, let the network output logits $z=f_\theta(x)\in\mathbb R^K$. Define

$$p_k(x)=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}},\qquad
\ell(z,y)=-z_y+\log\sum_{j=1}^Ke^{z_j}.$$

The class label $y$ is an integer index in this notation. Equivalently, a one-hot target vector $q$ has $q_y=1$ and all other entries zero, and the loss is $-\sum_kq_k\log p_k$. For a conditional true distribution $p_*(\cdot\mid x)$, averaging this loss over labels gives its cross entropy with $p_\theta$, namely $H(p_*,p_\theta)=H(p_*)+\operatorname{KL}(p_*\|p_\theta)$. Thus fitting probabilities is not merely choosing the largest score.

### Nonlinearity in inputs and nonlinearity in parameters

The model $\theta_0+\theta_1x+\theta_2x^2$ is nonlinear in the input but linear in its trainable coefficients. With fixed features $\phi(x)=(1,x,x^2)$, it is ordinary linear regression in feature space. Its squared-loss objective is convex in $\theta$.

A learned representation has the form

$$f_\theta(x)=v^\top\phi_{W,b}(x)+c,$$

where the feature map itself depends on trainable parameters. For example, $\phi_{W,b}(x)=\operatorname{ReLU}(Wx+b)$. Moving $W$ or $b$ changes which feature is being measured, not just how much weight it receives at the output.

Nonlinear parameterization does not by itself prove that every resulting optimization problem is difficult, and some special parameterizations admit useful reparameterizations. The substantive change in a general multilayer network is joint optimization over interacting layers. Convexity of the loss in output logits does not imply convexity in all network weights.

For the property service, a fixed quadratic basis assumes in advance which interactions matter. A network can learn intermediate combinations, but that flexibility creates an estimation problem rather than a guarantee of meaningful features. A hidden unit is not automatically “school quality” or “walkability.” Such interpretations require independent evidence.

## 2. Gradient-based optimization remains the training mechanism [21:00](https://www.youtube.com/watch?v=fRM41w9jzQo&t=1260s)

Gradient descent uses $\theta_{t+1}=\theta_t-\alpha\nabla J(\theta_t)$. If $I$ is sampled uniformly from the training indices, a single-example gradient satisfies

$$\mathbb E_I[\nabla_\theta\ell(f_\theta(x_I),y_I)]
=\nabla J(\theta).$$

An average over independently sampled minibatch indices has the same expectation and typically lower sampling variance. The identity is conditional on the current parameters. It does not say that the expected stochastic trajectory equals the deterministic gradient-descent trajectory after many steps: nonlinear gradients generally satisfy $\mathbb E[\nabla J(\theta_t)]\ne\nabla J(\mathbb E[\theta_t])$.

Minibatches balance gradient noise, memory, and hardware utilization. The relevant efficiency measure is progress toward the chosen validation target per unit of compute or elapsed time, not simply steps per second. A batch too small may underuse matrix hardware; a batch too large may spend computation obtaining redundant information.

Nonconvexity means that generic global optimality guarantees disappear. It does not make Newton's method mathematically inapplicable. An unmodified Newton direction can fail to be a descent direction when the Hessian is indefinite; damped, trust-region, and other curvature methods address this in different ways. Large networks often favor first-order methods because of their computational and memory costs, not because second derivatives cease to exist.

Likewise, random gradient noise is not a universal escape mechanism. At a point where every per-example gradient is zero, ordinary sampling introduces no gradient noise at all. Claims about escaping saddles or local minima require assumptions about the objective, noise, and algorithm.

A stationary point has zero gradient. A strict saddle has at least one negative-curvature direction. At a differentiable local minimum, the Hessian must be positive semidefinite, but positive semidefiniteness alone is not sufficient to establish a minimum: $f(t)=t^3$ has both zero first and second derivatives at zero. The recording's discussion of high-dimensional landscapes is motivation and research intuition, not a theorem that all neural networks lack bad local minima.

Our practical contract is therefore modest and testable: define the objective, compute its gradient correctly, choose a stable update, and measure both optimization and generalization. Backpropagation solves the gradient-computation problem. It does not solve model selection or certify global optimality.

## 3. A ReLU unit learns a threshold and a slope [36:00](https://www.youtube.com/watch?v=fRM41w9jzQo&t=2160s)

The rectified linear unit is the scalar function

$$\operatorname{ReLU}(t)=\max(0,t).$$

A single-input model $f(x)=a\operatorname{ReLU}(wx+b)+c$ is piecewise affine. If $w>0$, the hinge occurs at $x=-b/w$. To its left the output is $c$; to its right the slope is $aw$. Changing the bias moves the threshold, while changing the weights changes the active region's slope.

For an original repair-cost example, take

$$f(x)=1+2\operatorname{ReLU}(x-3).$$

For measurements $x=2,3,5$, predictions are $1,1,5$. The model represents a baseline cost followed by a linear increase beyond a threshold. It is not differentiable at $x=3$, but is differentiable elsewhere. An automatic-differentiation implementation must choose a convention at the kink; the next chapter makes that choice explicit.

A ReLU hidden unit does not make the final prediction nonnegative. A negative output weight or bias can produce a negative scalar after later layers. If positivity is a genuine output constraint, enforce it in the output parameterization or choose a suitable probabilistic model. Do not infer it merely from the presence of ReLUs somewhere in the network.

For a vector input, a unit computes $\operatorname{ReLU}(w^\top x+b)$. Its active region is a half-space, separated from the inactive region by the hyperplane $w^\top x+b=0$. Multiple units create multiple learned boundaries, and the output combines their responses.

### Learned hinges form a useful constructive picture

Any continuous one-dimensional piecewise-affine function with breakpoints $t_1,\ldots,t_m$ can be written as

$$f(x)=a+bx+\sum_{j=1}^mc_j\operatorname{ReLU}(x-t_j),$$

where $c_j$ is the change in slope at breakpoint $t_j$. To verify this, differentiate away from breakpoints: the derivative is $b+\sum_{j:x>t_j}c_j$. Each activated hinge adds precisely the desired slope change, and $a$ sets the vertical offset.

For example, slopes 0, 2, and 1 across intervals separated by 3 and 5 give $f(x)=1+2\operatorname{ReLU}(x-3)-\operatorname{ReLU}(x-5)$. The second hinge reduces the slope. This construction explains expressiveness without implying that training will recover the intended hinges from noisy data.

ReLU is positively homogeneous: $\operatorname{ReLU}(ct)=c\operatorname{ReLU}(t)$ for $c>0$. Consequently, multiplying a hidden unit's incoming weights and bias by $c$ and dividing its outgoing weight by $c$ leaves the function unchanged. Parameters are therefore not uniquely identified by the represented function. Weight penalties and optimization can distinguish parameterizations that make identical predictions.

## 4. Dense layers compose features with explicit dimensions [42:00](https://www.youtube.com/watch?v=fRM41w9jzQo&t=2520s)

We use column vectors for individual examples. A dense layer with input width $d$ and output width $m$ computes

$$u=Wx+b,\qquad a=\phi(u),$$

where $W\in\mathbb R^{m\times d}$, $b\in\mathbb R^m$, and $\phi$ is applied coordinatewise. There are $md+m$ trainable parameters. The row $W_{j,:}$ contains the weights feeding output unit $j$.

For layers of widths $m_0=d,m_1,\ldots,m_L$, write

$$a^{(0)}=x,\quad
u^{(\ell)}=W^{(\ell)}a^{(\ell-1)}+b^{(\ell)},\quad
a^{(\ell)}=\phi_\ell(u^{(\ell)}).$$

Then $W^{(\ell)}$ has shape $m_\ell\times m_{\ell-1}$. The output activation depends on the task; a regression head may simply use the final affine values, while a categorical head interprets them as logits.

Two affine layers without an intervening nonlinearity collapse into one:

$$W_2(W_1x+b_1)+b_2
=(W_2W_1)x+(W_2b_1+b_2).$$

Depth alone therefore does not create nonlinear input–output behavior. Nonlinear activations, attention, normalization, or other nonlinear operations prevent this collapse. Even in a linear network, factorization can alter optimization and implicit bias, but it does not expand the represented class beyond affine maps.

### A complete forward pass

Take two standardized property features $x=(2,1)^\top$ and a three-unit hidden layer:

$$W_1=\begin{pmatrix}1&-1\\-1&2\\1/2&1/2\end{pmatrix},
\qquad b_1=\begin{pmatrix}0\\0\\-1\end{pmatrix}.$$

The preactivations are $u=(1,0,1/2)^\top$, and ReLU leaves them unchanged. With output weights $v=(2,-1,3)^\top$ and intercept $c=1/2$, the prediction is

$$f(x)=v^\top a+c=2(1)-1(0)+3(1/2)+1/2=4.$$

This $2\to3\to1$ architecture has $6+3+3+1=13$ parameters. A three-class output head instead has a $3\times3$ weight matrix and three biases, making $6+3+9+3=21$ parameters. Hidden width and output dimension play different roles.

The following forward computation accepts fixed finite arrays with exactly these compatible dimensions and returns a scalar. It is an inference function: it performs no parameter update and has no external effects. Shape validation makes a row/column convention error visible rather than relying on accidental broadcasting.

```python
import numpy as np


def dense_regression(x, w1, b1, w2, b2):
    x, w1, b1, w2 = [
        np.asarray(value, dtype=float) for value in (x, w1, b1, w2)
    ]
    if x.ndim != 1 or w1.ndim != 2 or w1.shape[1] != x.size:
        raise ValueError("First-layer dimensions do not match the input")
    if b1.shape != (w1.shape[0],) or w2.shape != b1.shape:
        raise ValueError("Hidden and output dimensions do not match")
    if not all(np.isfinite(v).all() for v in (x, w1, b1, w2)):
        raise ValueError("Arrays must be finite")
    if not np.isscalar(b2) or not np.isfinite(b2):
        raise ValueError("Output bias must be a finite scalar")
    hidden = np.maximum(w1 @ x + b1, 0)
    return float(w2 @ hidden + b2)


w1 = np.array([[1, -1], [-1, 2], [0.5, 0.5]])
b1 = np.array([0, 0, -1])
w2 = np.array([2, -1, 3])
assert np.isclose(dense_regression([2, 1], w1, b1, w2, 0.5), 4)
assert w1.size + b1.size + w2.size + 1 == 13
```

For a minibatch stored as rows in $X\in\mathbb R^{B\times d}$, the same first layer becomes $U=XW^\top+\mathbf1b^\top$. The transpose changes because examples are now rows; the mathematical map for each example is unchanged. Recording the convention prevents many gradient and broadcasting errors.

### Alternative activation functions change local behavior

The sigmoid $\sigma(t)=1/(1+e^{-t})$ has derivative $\sigma(t)(1-\sigma(t))$, at most $1/4$. Hyperbolic tangent has derivative $1-\tanh^2(t)$. Both saturate at large absolute inputs, where derivatives become small. A chain of many small derivatives can weaken gradient propagation.

Leaky ReLU uses $t$ for positive inputs and $\alpha t$ for negative inputs, with a chosen $\alpha>0$, retaining a nonzero slope on the negative side. GELU uses $t\Phi(t)$, where $\Phi$ is the standard normal cumulative distribution function. Its derivative is $\Phi(t)+t\varphi(t)$, with $\varphi$ the normal density; it is smooth and can be slightly nonmonotone. [Hendrycks and Gimpel](https://arxiv.org/abs/1606.08415) introduced GELU and evaluated it empirically.

There is no theorem that a smoother activation always produces a better trained model. Activation choice interacts with initialization, architecture, normalization, and the task. Its formula is a mathematical fact; superiority in a particular setting is an experimental claim.

## 5. Residual connections make the identity map easy to represent [65:00](https://www.youtube.com/watch?v=fRM41w9jzQo&t=3900s)

A residual block computes

$$h_{\rm out}=h+F_\theta(h).$$

The input and residual output must have the same dimension for this sum. A common residual branch is $F_\theta(h)=W_2\phi(W_1h+b_1)+b_2$, where an intermediate width can differ from the block width as long as the second affine layer maps back.

If the useful transformation is already close to identity, the branch can learn a small correction. Setting the residual branch to zero realizes identity immediately. This provides a parameterization in which preserving a representation is easy. It does not require intermediate features to have the same semantic meaning as the final labels.

The local derivative is

$$\frac{\partial h_{\rm out}}{\partial h}=I+J_F(h),$$

where $J_F$ is the residual branch's Jacobian. The identity term supplies a direct path for changes in the input. If $\|J_F\|_2\le\delta<1$, then for any perturbation $v$,

$$(1-\delta)\|v\|\le\|(I+J_F)v\|\le(1+\delta)\|v\|.$$

The lower bound follows from the reverse triangle inequality and the upper bound from the ordinary triangle inequality. Under this explicit local condition, the block neither completely kills nor arbitrarily amplifies a small perturbation.

The condition is essential. If $J_F=-I$, the total derivative is zero; if many blocks each expand a direction slightly, their product can still grow substantially. Residual connections improve representational and optimization options, but do not guarantee stable gradients at every parameter setting.

For a numerical case, let $F(h)=0.1h$ in a two-dimensional block. The output and every small perturbation are multiplied by 1.1. Ten such blocks multiply by $1.1^{10}\approx2.594$, illustrating both the mildness of one block and accumulation across depth.

[He et al.'s residual-network paper](https://arxiv.org/abs/1512.03385) provides empirical evidence that residual parameterization enabled substantially deeper image-recognition networks to be optimized effectively. The Jacobian bound above is an original explanatory calculation, not a complete theoretical explanation of those experimental results.

When a block changes width, an optional learned projection $Ph$ can replace the identity shortcut, giving $Ph+F(h)$. It must have compatible output dimensions. Because $P$ need not be identity or norm-preserving, the simple $I+J_F$ argument changes to $P+J_F$.

## 6. Normalization changes which variations a layer can see [72:00](https://www.youtube.com/watch?v=fRM41w9jzQo&t=4320s)

For one feature vector $h\in\mathbb R^m$, layer normalization computes

$$\mu=\frac1m\sum_jh_j,\qquad
v=\frac1m\sum_j(h_j-\mu)^2,\qquad
\operatorname{LN}(h)_j=
\gamma_j\frac{h_j-\mu}{\sqrt{v+\varepsilon}}+\beta_j.$$

Here $\varepsilon>0$ prevents division by zero; gain $\gamma$ and shift $\beta$ are trainable vectors. Statistics are computed across features of this one example, not across the minibatch. [Ba, Kiros, and Hinton](https://arxiv.org/abs/1607.06450) introduced this construction and its use in recurrent networks.

Before the learned affine transform, the normalized vector has mean zero. Its average squared coordinate is $v/(v+\varepsilon)$, approximately one when $v$ is much larger than $\varepsilon$. Its Euclidean norm is approximately $\sqrt m$, not one. With coordinate-specific gains and shifts, the final vector need not have any single prescribed mean or variance.

RMS normalization instead uses the root mean square

$$r=\sqrt{\frac1m\sum_jh_j^2+\varepsilon},\qquad
\operatorname{RMSNorm}(h)_j=\gamma_j\frac{h_j}{r}.$$

It does not subtract the mean. The standard form shown here has no additive shift. [Zhang and Sennrich](https://arxiv.org/abs/1910.07467) study this simplification and report performance and efficiency results in their evaluated settings.

For $h=(1,3)$ and $\varepsilon=0$, layer normalization with unit gains and zero shifts gives $(-1,1)$. RMS normalization gives $(1/\sqrt5,3/\sqrt5)$. The first removes a common offset; the second retains it while normalizing total squared magnitude. Both outputs have Euclidean norm $\sqrt2$ in this example.

For positive $c$ and zero stabilizer, both normalized cores are invariant to multiplying the input by $c$. Layer normalization is also invariant to adding the same scalar to all coordinates. RMS normalization generally is not. A negative scale flips the normalized core's sign, so unrestricted “scale invariance” is too broad. With a fixed positive stabilizer, positive scale invariance is only approximate.

```python
def normalize_features(values, epsilon=1e-8, center=True):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("Expected a nonempty feature vector")
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Stabilizer must be finite and positive")
    residual = values - values.mean() if center else values
    return residual / np.sqrt(np.mean(residual ** 2) + epsilon)


assert np.allclose(normalize_features([1, 3]), [-1, 1])
assert np.allclose(normalize_features([1, 3], center=False),
                   np.array([1, 3]) / np.sqrt(5))
assert np.allclose(normalize_features([7, 7]), [0, 0])
assert np.allclose(normalize_features([11, 13]),
                   normalize_features([1, 3]))
```

This code implements only the normalized core, with no learned gain or shift, for a single vector. It accepts finite inputs in ordinary numerical ranges; squaring extremely large floating-point values can overflow, so it is not a production kernel for arbitrary magnitudes. That limitation also explains why numerical precision and implementation details matter beyond the formal equations.

Normalization can remove useful information. If overall feature magnitude carries an important signal, a normalized branch alone may lose it. Residual paths or additional features can preserve information that a normalized sublayer discards. Also, invariant forward predictions do not imply invariant parameter gradients: changing parameter scale changes the relationship between parameter steps and function changes.

## 7. Convolution is structured parameter sharing [79:00](https://www.youtube.com/watch?v=fRM41w9jzQo&t=4740s)

The recording closes with a brief convolution sketch. In one dimension, a shared filter $(k_0,k_1)$ can compute $y_i=k_0x_i+k_1x_{i+1}$. For four inputs and valid positions, this is

$$y=\begin{pmatrix}k_0&k_1&0&0\\
0&k_0&k_1&0\\0&0&k_0&k_1\end{pmatrix}x.$$

A generic $3\times4$ dense map has 12 weights; this structured map has two shared filter weights. The operation written here is cross-correlation, conventionally called convolution in neural-network libraries. Mathematical convolution reverses the filter indexing.

With $k=(1,-1)$ and $x=(1,2,4,7)$, the output is $(-1,-2,-3)$, detecting local differences. Sharing the same filter expresses the belief that the same local pattern matters at different positions. Away from boundary effects, shifting the input shifts the output correspondingly: this is translation equivariance, not translation invariance.

Padding, stride, boundaries, channels, and two-dimensional images add details but preserve the central idea of local connectivity and shared parameters. The recording does not develop a full convolutional-network treatment, so this section remains a precise account of that sketch rather than an invented additional lecture.

## Exercises

1. **Compute.** For widths $4\to5\to3\to2$ with a bias in every affine layer, count trainable parameters and list all weight and bias shapes.
2. **Construct.** Write a ReLU representation of a continuous function with $f(0)=2$, slope 1 below $x=1$, slope 3 between 1 and 4, and slope $-1$ above 4.
3. **Counterexample.** Explain why residual connections do not guarantee nonvanishing derivatives, and why stacking affine layers without nonlinearities does not create a nonlinear function of inputs.
4. **Compute and distinguish.** For $h=(2,4)$, calculate the zero-stabilizer cores of layer normalization and RMS normalization. Repeat after adding 10 to both coordinates. Which operation removes that shift?
5. **Design.** A network predicts negative repair costs and relies on hidden units described as “building quality.” Propose mathematical and empirical checks before interpreting or deploying it.

<details><summary>Solutions</summary>

1. The first weight matrix is $5\times4$ with a 5-vector bias, contributing 25 parameters. The second is $3\times5$ with a 3-vector bias, contributing 18. The output is $2\times3$ with a 2-vector bias, contributing 8. Total: $25+18+8=51$. Coordinatewise activations add no trainable parameters in the simple ReLU case. Normalization layers or learned activation parameters would require additional accounting.

2. Start with $2+x$, which has the required initial value and slope. At 1 the slope must increase by 2; at 4 it must decrease by 4. Hence $f(x)=2+x+2\operatorname{ReLU}(x-1)-4\operatorname{ReLU}(x-4)$. The function is continuous because each hinge is continuous, and differentiating on the three open intervals verifies slopes 1, 3, and $-1$.

3. A residual branch $F(h)=-h$ makes the full block identically zero and gives Jacobian $I-I=0$. The identity path can be canceled. For affine composition, direct expansion gives $W_2W_1x+W_2b_1+b_2$, again affine in $x$. More parameters or more layers do not by themselves establish nonlinear input–output behavior.

4. Layer normalization gives mean 3, standard deviation 1, and output $(-1,1)$. RMS normalization divides by $\sqrt{(4+16)/2}=\sqrt{10}$, giving $(2/\sqrt{10},4/\sqrt{10})$. After the common shift, layer normalization still gives $(-1,1)$, whereas RMS normalization gives $(12/\sqrt{170},14/\sqrt{170})$. Layer normalization removes the common offset; RMS normalization retains a dependence on it.

5. Inspect the output map: hidden ReLUs do not constrain the final affine output's sign. If the task requires nonnegative conditional means, choose and validate a positive output parameterization or an appropriate response distribution, rather than clipping silently without measuring its effect. Test calibration and error on held-out properties, including unusual feature ranges. For interpretation, compare hidden activations with independently measured quality variables and examine whether alternative parameterizations or unit permutations preserve predictions while changing the proposed labels. A suggestive activation pattern is evidence to investigate, not proof that a latent unit has a unique semantic meaning.

</details>

## Primary references and evidence boundaries

The [recording](https://www.youtube.com/watch?v=fRM41w9jzQo) provides the architectural sequence. Primary research anchors are [ResNet](https://arxiv.org/abs/1512.03385), [Layer Normalization](https://arxiv.org/abs/1607.06450), [RMSNorm](https://arxiv.org/abs/1910.07467), and [GELU](https://arxiv.org/abs/1606.08415). Shape calculations, the property network, the local residual bound, normalization examples, and convolution matrix are original derivations. Corrections concerning Newton methods, normalization norms, and scale invariance make the spoken discussion mathematically explicit without claiming additional visual evidence.

# Lecture 8 — Backpropagation, computational graphs, and higher-order derivatives

Research extensions are identified through links to primary papers; numerical examples and executable checks are original teaching constructions.

A neural network is useful for learning only if we can determine how its loss changes when its parameters change. Computing one numerical difference per parameter would make large models prohibitively expensive. Backpropagation exploits the structure of the computation: it reuses intermediate values and propagates the influence of a scalar loss backward through local operations.

We will differentiate the small property-cost network from Lecture 7, first by hand and then with a complete finite-difference check. The recording begins with general differentiable circuits and higher-order applications, then develops the chain rule, parameter branches, affine maps, and activations. The chapter retains that progression while revisiting the same numerical model throughout. All numerical examples and code are original.

## 1. The computational graph is the object being differentiated [00:00](https://www.youtube.com/watch?v=ne2ngVAoMG8&t=0s)

Let $J(\theta)$ be a scalar loss and $\theta\in\mathbb R^p$ its parameter vector. A forward evaluation computes $J$ through a directed acyclic graph of elementary operations. Nodes hold intermediate values; edges show which values are needed to compute later ones.

For example, define $u=a+b$, $v=uc$, and $J=v+u^2$. The intermediate $u$ is used twice but need only be computed once. A computational graph preserves that sharing. Expanding a symbolic formula without care can duplicate work and obscure where derivative contributions must be added.

Automatic differentiation, abbreviated AD, applies derivative rules to the executed operations. It is not finite differences: there is no chosen perturbation size in the derivative computation. It is also not necessarily symbolic differentiation into a giant closed-form expression. It evaluates derivatives using the same numerical values and structural dependencies as the program.

Subject to floating-point arithmetic and the specified primitive derivative rules, AD computes the derivative of the implemented computation. This last phrase matters. If the program implements the wrong loss, detaches part of the graph, or takes a nondifferentiable discrete decision, differentiation does not repair the modeling mistake.

### Why one scalar loss is especially favorable

Suppose the forward graph requires $N$ elementary operations and each operation has a backward rule costing at most a constant multiple of its forward cost. Reverse-mode AD can compute the gradient of one scalar output in $O(N)$ work, aside from representing requested outputs. It does not need a separate full forward pass for each of the $p$ parameters.

This is a complexity statement about operations, not an unconditional statement that work always equals parameter count. A shared weight used at a thousand sequence positions contributes to a thousand computations. Attention can have substantial sequence-length-dependent cost without a matching increase in parameter count. For a single dense matrix–vector operation, operation count and weight count do have the same order.

Forward-mode AD propagates a directional perturbation of the inputs and returns a Jacobian–vector product. Reverse mode propagates sensitivity from an output and returns a transposed-Jacobian–vector product. For a scalar loss and many parameters, reverse mode obtains all parameter sensitivities in one backward traversal. For few inputs and many outputs, forward mode may be more attractive.

Computational efficiency does not imply zero storage cost. A backward rule often needs values computed in the forward pass. Saving those values creates an activation-memory requirement; recomputing them creates extra work. The graph determines both the mathematical derivatives and the systems tradeoff.

[Baydin and colleagues' AD survey](https://arxiv.org/abs/1502.05767) gives a broader account of these modes and their use in machine learning. The central rule needed here is simpler: propagate a scalar loss's differential backward, adding contributions whenever several paths meet.

## 2. Differentiating derivatives gives useful products without a full Hessian [13:00](https://www.youtube.com/watch?v=ne2ngVAoMG8&t=780s)

The Hessian $H(\theta)=\nabla^2J(\theta)$ contains all second partial derivatives. A dense $p\times p$ Hessian requires quadratic storage, which is often impractical. Many algorithms instead need its action on a vector $v\in\mathbb R^p$.

Assume $J$ is twice continuously differentiable near the evaluation point, and hold $v$ fixed independently of $\theta$. Define

$$g(\theta)=\nabla J(\theta)^\top v.$$

Then

$$\nabla g(\theta)=H(\theta)^\top v=H(\theta)v,$$

where symmetry follows from the smoothness assumption. Computing this product by differentiating the gradient calculation avoids materializing the Hessian. If $v$ depends on $\theta$, another derivative term appears; treating it as fixed is part of the specification.

An exact two-dimensional example is

$$J(\theta_1,\theta_2)=\frac12(\theta_1^2+3\theta_2^2)
+\theta_1\theta_2.$$

Its gradient is $(\theta_1+\theta_2,\theta_1+3\theta_2)^\top$, and its Hessian is $\begin{pmatrix}1&1\\1&3\end{pmatrix}$. For $v=(2,-1)^\top$, the product is $(1,-1)^\top$. One can obtain that result by differentiating $2(\theta_1+\theta_2)-(\theta_1+3\theta_2)=\theta_1-\theta_2$.

[Pearlmutter's original Hessian-product paper](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf) develops efficient exact multiplication by the Hessian. The practical gain is access to curvature through an operator rather than a stored matrix. It does not make assembling every Hessian entry linear-time.

### Backpropagation through an optimization step

Let one training update be

$$\theta'=\theta-\alpha\nabla L_{\rm train}(\theta),$$

and evaluate an outer objective $F(\theta)=L_{\rm valid}(\theta')$. Here $\alpha$ is a fixed step size. Applying the chain rule gives

$$\nabla_\theta F=
\left(I-\alpha\nabla^2L_{\rm train}(\theta)\right)^\top
\nabla_{\theta'}L_{\rm valid}(\theta').$$

The Hessian appears only through its product with the outer gradient. For a scalar example, let $L_{\rm train}(\theta)=\tfrac12(\theta-2)^2$, $L_{\rm valid}(\theta')=\tfrac12(\theta'-4)^2$, and $\alpha=1/2$. Then $\theta'=\theta/2+1$, and at $\theta=0$ the outer derivative is $(1-4)/2=-1.5$. Ignoring the derivative of the update would instead return $-3$.

If optimizing the learning rate while holding the initial $\theta$ fixed, the derivative is

$$\frac{\partial F}{\partial\alpha}
=-\nabla L_{\rm valid}(\theta')^\top
\nabla L_{\rm train}(\theta).$$

These are mathematical foundations for differentiating through learning procedures. [Finn, Abbeel, and Levine's MAML paper](https://arxiv.org/abs/1703.03400) trains an initialization for rapid adaptation across tasks. A single train–validation calculation is not the complete MAML algorithm, but it exposes the derivative on which that construction relies.

Unrolling many optimizer steps can be expensive in memory, and discrete choices such as selecting the best checkpoint require separate treatment. Higher-order differentiation also requires backward computations to remain differentiable. A library returning first derivatives does not automatically preserve the graph needed for second derivatives.

## 3. The chain rule becomes a local backward contract [18:00](https://www.youtube.com/watch?v=ne2ngVAoMG8&t=1080s)

Let $u=g(z)$ with $z\in\mathbb R^m$ and $u\in\mathbb R^n$, followed by a scalar $J=f(u)$. Define the Jacobian by

$$(J_g)_{ij}=\frac{\partial u_i}{\partial z_j},
\qquad J_g\in\mathbb R^{n\times m}.$$

Use column gradients and denote an upstream sensitivity by $\bar u=\nabla_uJ$. The chain rule is

$$\boxed{\bar z=J_g(z)^\top\bar u.}$$

The shapes certify the formula: an $m\times n$ transposed Jacobian multiplies an $n$-vector to produce an $m$-vector. In differential form, $dJ=\bar u^\top du=\bar u^\top J_g\,dz=(J_g^\top\bar u)^\top dz$.

A local backward operation therefore needs the incoming sensitivity and whatever saved forward values its own derivative requires. It does not need to reconstruct the entire downstream function $f$. All of $f$'s relevant first-order effect is summarized by $\bar u$.

This abstraction allows independently implemented modules to compose. An activation module, a matrix multiplication, and a normalization module can each define a backward rule. The reverse traversal combines those rules in the opposite order from the forward computation.

### Shared paths require addition, not overwriting

Return to $u=a+b$, $v=uc$, $J=v+u^2$, with $a=1$, $b=2$, and $c=4$. The forward values are $u=3$, $v=12$, and $J=21$.

Initialize $\bar J=1$. The addition gives sensitivity 1 to $v$ and to the square term. The path through $v=uc$ contributes $c=4$ to $\bar u$ and $u=3$ to $\bar c$. The square contributes $2u=6$ to $\bar u$. Therefore

$$\bar u=4+6=10,\qquad
\bar a=\bar b=10,\qquad \bar c=3.$$

Overwriting one contribution to $\bar u$ with the other would give an incorrect derivative. At a general node with several consumers, sum the backward contributions from every outgoing path before propagating further.

The same issue occurs with tied weights, recurrent computations, and residual branches. A parameter used at several sites is a single variable with several paths to the loss. Its gradient is the sum of the contributions, not the contribution from whichever use happens to be visited last.

For a residual block $u=z+F(z)$, the backward rule is

$$\bar z=\bar u+J_F(z)^\top\bar u.$$

For a coordinatewise product $u=a\odot b$, it is $\bar a=\bar u\odot b$ and $\bar b=\bar u\odot a$. Here $\odot$ denotes elementwise multiplication, not a dot product. Such rules make gated networks differentiable without requiring a special global algorithm for each architecture.

### Broadcasting has a corresponding reduction

If a bias vector is added to every row of a minibatch matrix, the forward operation reuses that bias across examples. The backward operation sums the gradient contributions across those rows. A broadcast over several axes must be reversed by summing over the corresponding axes.

This is a general dependency rule rather than a library quirk. One shared scalar added to every element of a matrix receives the sum of all element sensitivities. Missing that reduction can produce an array with plausible numbers but the wrong gradient shape.

## 4. Affine layers expose both activation and parameter gradients [33:00](https://www.youtube.com/watch?v=ne2ngVAoMG8&t=1980s)

Consider

$$u=Wz+b,\qquad
z\in\mathbb R^m,\quad W\in\mathbb R^{n\times m},
\quad b,u\in\mathbb R^n.$$

Given $\bar u=\nabla_uJ$, vary all three inputs:

$$du=(dW)z+W\,dz+db.$$

Substitute into $dJ=\bar u^\top du$ and identify the coefficients of the independent variations. The resulting backward rules are

$$\boxed{\bar z=W^\top\bar u,\qquad
\nabla_WJ=\bar u z^\top,\qquad
\nabla_bJ=\bar u.}$$

The matrix gradient uses the Frobenius inner product convention $dJ=\operatorname{tr}((\nabla_WJ)^\top dW)+\cdots$. Its $(i,j)$ entry is $\bar u_i z_j$. This also follows by noting that changing $W_{ij}$ directly changes only $u_i$, at rate $z_j$.

An original numerical example uses

$$W=\begin{pmatrix}1&2\\-1&3\end{pmatrix},\quad
z=\begin{pmatrix}2\\-1\end{pmatrix},\quad
\bar u=\begin{pmatrix}4\\5\end{pmatrix}.$$

Then $\bar z=(-1,23)^\top$ and

$$\nabla_WJ=\begin{pmatrix}8&-4\\10&-5\end{pmatrix},
\qquad \nabla_bJ=(4,5)^\top.$$

The forward value of the affine map is not itself the gradient. The upstream sensitivity describes how the eventual loss responds to each output coordinate; multiplying by the input tells how a particular weight affects that output.

For a single use of a dense layer on one example, the weight gradient is an outer product and has rank at most one. It is rank zero if either factor vanishes. With a minibatch, gradients are summed or averaged and can have higher rank. A tied weight matrix used at multiple sequence positions also receives a sum of outer products even for one sequence, so “one example always gives a rank-one weight gradient” is too broad.

For row-batched input $Z\in\mathbb R^{B\times m}$, forward output is $U=ZW^\top+\mathbf1b^\top$. If $G=\nabla_UJ\in\mathbb R^{B\times n}$ already contains any averaging factor from the loss,

$$\nabla_ZJ=GW,\qquad
\nabla_WJ=G^\top Z,\qquad
\nabla_bJ=G^\top\mathbf1.$$

Do not divide by $B$ again if $G$ already includes that factor. Whether the loss sums or averages examples is part of its definition and must remain consistent through the derivative.

### Backward work and memory follow the same dependencies

For one dense example, the forward matrix–vector product costs $O(nm)$. The backward matrix–vector product and outer product also cost $O(nm)$. Elementwise operations cost linear time in the number of elements. This concretely realizes the circuit complexity argument.

To compute the weight gradient, the layer needs its forward input $z$. To compute the input gradient, it needs $W$ and $\bar u$. Once a saved value has no remaining backward consumers, it can be released. Mutating a needed parameter or activation before its backward use can invalidate the derivative.

[Chen and colleagues' checkpointing paper](https://arxiv.org/abs/1604.06174) trades recomputation for reduced activation storage. For an idealized chain of $L$ equally sized layers, checkpointing at intervals of $k$ layers requires roughly $O(L/k+k)$ activation storage while recomputing segments during backward traversal. Choosing $k$ near $\sqrt L$ gives the familiar square-root memory scaling in that simplified setting. Real graphs have unequal tensor sizes, branches, and additional parameter and optimizer memory.

## 5. Activation derivatives complete a trainable network [56:00](https://www.youtube.com/watch?v=ne2ngVAoMG8&t=3360s)

For an elementwise activation $a=\phi(u)$, the Jacobian is diagonal because $a_i$ depends only on $u_i$. Thus

$$\bar u=\bar a\odot\phi'(u).$$

For ReLU, use derivative 1 at positive inputs and 0 at negative inputs. At zero there is no ordinary derivative; the implementation below chooses 0. That convention should not be described as a theorem that ReLU is differentiable everywhere.

Sigmoid has derivative $\sigma(u)(1-\sigma(u))$, and hyperbolic tangent has derivative $1-\tanh^2(u)$. These can use saved activation values. A softmax, by contrast, is not elementwise: its Jacobian is $\operatorname{diag}(p)-pp^\top$. Combining it with cross entropy simplifies the derivative with respect to logits to $p-q$ for a target distribution $q$.

### A full backward pass through the property network

Use the parameters from Lecture 7, but choose $x=(2,0.5)^\top$ to avoid a ReLU kink:

$$W_1=\begin{pmatrix}1&-1\\-1&2\\1/2&1/2\end{pmatrix},
\quad b_1=(0,0,-1)^\top,\quad
w_2=(2,-1,3)^\top,\quad b_2=1/2.$$

The forward pass gives $u=(1.5,-1,0.25)^\top$, $a=(1.5,0,0.25)^\top$, and prediction $\hat y=4.25$. For target $y=3$, use $J=\tfrac12(\hat y-y)^2=0.78125$.

First, $\bar{\hat y}=\hat y-y=1.25$. The output affine layer gives

$$\nabla_{w_2}J=1.25a=(1.875,0,0.3125)^\top,\qquad
\nabla_{b_2}J=1.25,$$

$$\bar a=1.25w_2=(2.5,-1.25,3.75)^\top.$$

The ReLU mask is $(1,0,1)^\top$, so $\bar u=(2.5,0,3.75)^\top$. Finally,

$$\nabla_{W_1}J=\bar u x^\top
=\begin{pmatrix}5&1.25\\0&0\\7.5&1.875\end{pmatrix},
\qquad \nabla_{b_1}J=(2.5,0,3.75)^\top.$$

The optional input gradient is $W_1^\top\bar u=(4.375,-0.625)^\top$. Input gradients can support sensitivity analysis, but they are local derivatives at the chosen point, not causal effects of changing a building's physical characteristics.

The inactive unit receives zero gradient through its ReLU at this example. Other examples may activate it, so one zero gradient does not imply a permanently dead unit. Conversely, a unit inactive on every training input can receive no learning signal through this path.

### A runnable all-parameter check

The following function packs the network's 13 parameters into one vector: six first-layer weights in row-major order, three first-layer biases, three output weights, and the output bias. It accepts one two-dimensional input and one scalar target, and returns the loss and a gradient in the same parameter order. It does not update parameters or perform external actions.

```python
import numpy as np


def loss_and_gradient(parameters, x, target):
    parameters = np.asarray(parameters, dtype=float)
    x = np.asarray(x, dtype=float)
    if parameters.shape != (13,) or x.shape != (2,):
        raise ValueError("Expected 13 parameters and two input features")
    if not np.isfinite(parameters).all() or not np.isfinite(x).all():
        raise ValueError("Parameters and input must be finite")
    if not np.isscalar(target) or not np.isfinite(target):
        raise ValueError("Target must be a finite scalar")
    w1 = parameters[:6].reshape(3, 2)
    b1, w2, b2 = parameters[6:9], parameters[9:12], parameters[12]
    preactivation = w1 @ x + b1
    activation = np.maximum(preactivation, 0)
    error = float(w2 @ activation + b2 - target)
    hidden_gradient = error * w2 * (preactivation > 0)
    gradient = np.concatenate([
        np.outer(hidden_gradient, x).ravel(),
        hidden_gradient,
        error * activation,
        [error],
    ])
    return 0.5 * error ** 2, gradient


parameters = np.array([
    1, -1, -1, 2, 0.5, 0.5, 0, 0, -1, 2, -1, 3, 0.5
], dtype=float)
x = np.array([2, 0.5])
loss, gradient = loss_and_gradient(parameters, x, 3)
expected = np.array([
    5, 1.25, 0, 0, 7.5, 1.875, 2.5, 0, 3.75,
    1.875, 0, 0.3125, 1.25,
])
assert np.isclose(loss, 0.78125)
assert np.allclose(gradient, expected)

step = 1e-6
finite_difference = np.empty_like(parameters)
for index in range(parameters.size):
    direction = np.zeros_like(parameters)
    direction[index] = step
    plus = loss_and_gradient(parameters + direction, x, 3)[0]
    minus = loss_and_gradient(parameters - direction, x, 3)[0]
    finite_difference[index] = (plus - minus) / (2 * step)
assert np.allclose(gradient, finite_difference, atol=1e-7, rtol=1e-6)
assert loss_and_gradient(parameters - 0.01 * gradient, x, 3)[0] < loss
```

The finite-difference calculation is an independent check of the analytical backward rule, not the intended training method. Checking all 13 parameters is inexpensive here. For a large network, a directional check compares $v^\top\nabla J$ with $[J(\theta+hv)-J(\theta-hv)]/(2h)$ and tests the whole gradient through a small number of random directions.

Finite differences have truncation error when $h$ is too large and cancellation error when it is too small. They can also disagree with a chosen ReLU derivative convention if the perturbation crosses a kink. Use double precision, avoid nondifferentiable points for smooth checks, and inspect discrepancies rather than adjusting tolerances until a faulty derivative passes.

### A normalization backward rule as a research-level extension

The preceding rules are sufficient for the recorded dense-network derivation. To connect to Lecture 7's normalization, consider the unscaled layer-normalization core. Let $c=u-\mu\mathbf1$, $s=\sqrt{m^{-1}c^\top c+\varepsilon}$, and $a=c/s$. For upstream gradient $g=\nabla_aJ$,

$$\nabla_uJ=\frac1s\left[g-\operatorname{mean}(g)\mathbf1
-a\,\operatorname{mean}(g\odot a)\right].$$

To derive it, centering gives $dc=(I-\mathbf1\mathbf1^\top/m)du$. Differentiating the scale gives $ds=c^\top du/(ms)$ because $c^\top\mathbf1=0$. Substitute into $da=dc/s-c\,ds/s^2$, then transpose the resulting linear map against $g$. The formula follows without constructing a dense Jacobian.

If the layer includes learned gain $\gamma$ and shift $\beta$, first replace $g$ by $g\odot\gamma$ for the core's backward rule; the gain and shift gradients are the appropriate accumulated products with $a$ and ones. This extension illustrates the same local-contract principle for a more coupled operation.

## Exercises

1. **Derive.** For $u=a+b$, $v=uc$, and $J=v+u^2$, derive the three input gradients symbolically. Explain why the contribution to $u$ must be accumulated.
2. **Compute.** In the worked network, compute the gradient with respect to the input and identify which hidden unit blocks its upstream sensitivity. Explain why zero training loss would make every displayed parameter gradient zero.
3. **Counterexample.** A weight matrix is used twice in one sequence. Why can its gradient have rank two even when each use contributes an outer product? Give a $2\times2$ example.
4. **Derive.** For one inner update $\theta'=\theta-\alpha\nabla L_{\rm train}(\theta)$, derive the outer gradient and the learning-rate derivative. State what is being held fixed in each derivative.
5. **Design and check.** A custom normalization layer has correct-looking outputs but unstable training. Design a minimal derivative check and list graph or reduction mistakes it could expose.

<details><summary>Solutions</summary>

1. Since $J=(a+b)c+(a+b)^2$, the derivatives with respect to $a$ and $b$ are both $c+2(a+b)$, while the derivative with respect to $c$ is $a+b$. The intermediate $u$ has two consumers, multiplication by $c$ and squaring. Their sensitivities add to $c+2u$. Overwriting either branch would drop a real path by which a perturbation changes the loss.

2. The masked hidden sensitivity is $(2.5,0,3.75)^\top$. Multiplying by $W_1^\top$ gives $(2.5+1.875,-2.5+1.875)^\top=(4.375,-0.625)^\top$. The second unit has preactivation $-1$ and blocks its sensitivity under ReLU. If the scalar squared-error residual were zero, the initial upstream derivative would be zero, and every downstream product in this particular backward graph would be zero. That says nothing about loss on other examples.

3. Let one use contribute $e_1e_1^\top$ and another contribute $e_2e_2^\top$, where $e_1=(1,0)^\top$ and $e_2=(0,1)^\top$. Each contribution has rank one, but their sum is the identity matrix, which has rank two. The rank-one statement applies to a single affine use with one input vector, not to every possible reuse of a shared parameter.

4. Holding $\alpha$ fixed, $d\theta'/d\theta=I-\alpha H_{\rm train}(\theta)$, so $\nabla_\theta F=(I-\alpha H_{\rm train})^\top\nabla L_{\rm valid}(\theta')$. Holding the initial $\theta$ fixed instead, $\partial\theta'/\partial\alpha=-\nabla L_{\rm train}(\theta)$, giving $\partial F/\partial\alpha=-\nabla L_{\rm valid}(\theta')^\top\nabla L_{\rm train}(\theta)$. If the step size or direction depends on additional trainable quantities, their dependencies must also be differentiated.

5. Choose a small, nonconstant vector in double precision and define a scalar test loss as a fixed weighted sum of the layer outputs. Compare the analytical input derivative with central differences, then separately test gain and shift parameters. Use nonuniform upstream weights so an incorrect derivative cannot hide behind a symmetric sum. Check whether feature means were taken over the intended axis, whether broadcast parameters receive summed gradients, whether the stabilizer appears in the derivative consistently, and whether saved forward values were mutated. Add a constant-input case to verify finite behavior with a positive stabilizer, while recognizing that it tests numerical stability as well as differentiation.

</details>

## Primary references and evidence boundaries

The [recording](https://www.youtube.com/watch?v=ne2ngVAoMG8) supplies the circuit, chain-rule, affine-gradient, and activation sequence. Extensions are grounded in [Baydin et al. on AD](https://arxiv.org/abs/1502.05767), [Pearlmutter on Hessian products](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf), [Finn et al. on MAML](https://arxiv.org/abs/1703.03400), and [Chen et al. on checkpointing](https://arxiv.org/abs/1604.06174). The all-parameter gradient check, normalization derivative, numerical outer-gradient example, and exercises are original developments. Operation-count and rank qualifications clarify the limits of informal statements in the recording.

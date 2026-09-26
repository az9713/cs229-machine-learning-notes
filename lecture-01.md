# Lecture 1 — What is being learned?

A property-research assistant is asked to estimate a house's price, identify comparable sales, summarize the neighborhood, and decide whether another document would improve its answer. These requests sound like one application, yet they require different learning problems. Predicting price needs a target and a loss. Finding comparable properties needs a representation and a similarity rule. Choosing another search needs a policy that trades information against cost. This chapter uses that assistant to make the course's organizing distinctions precise.

## 1. Experience, tasks, and measurable improvement [05:00](https://www.youtube.com/watch?v=DATnpGoGhM8&t=300s)

### Separate the learning procedure from the learned function

Let $\mathcal X$ be the set of possible property descriptions and $\mathcal Y$ the set of outputs. A predictor $f:\mathcal X\to\mathcal Y$ maps a description to an output. A dataset $D_n=\{(x_i,y_i)\}_{i=1}^n$ contains $n$ observed input-output pairs. A learning algorithm $A$ maps the dataset to a predictor, written $f_D=A(D_n)$. This separates the procedure that learns from the function it produces. Two algorithms can produce the same predictor; the same algorithm can produce different predictors from different data.

For a price task, $x_i$ might contain area, location, and condition, while $y_i$ is the eventual sale price. For a document task, the input might be a passage and the output a topic label. The format of the data does not alone define the goal. We must also specify what performance means and on which population it is measured.

A **loss function** $\ell(\hat y,y)$ assigns a numerical cost to a prediction $\hat y$ when the target is $y$. Let $P$ be a probability distribution over input-target pairs, and let $\mathbb E$ denote expectation. The **population risk** is

$$R_P(f)=\mathbb E_{(X,Y)\sim P}[\ell(f(X),Y)].$$

The notation $(X,Y)\sim P$ means that the random pair is drawn according to $P$. Population risk is the average loss on the specified population, including outcomes absent from the training sample. Its observed counterpart is the **empirical risk**,

$$\widehat R_D(f)=\frac1n\sum_{i=1}^n\ell(f(x_i),y_i).$$

For a fixed predictor chosen independently of an independent, identically distributed evaluation sample, the empirical average is an unbiased estimate of its population risk, assuming the expectation exists. The same statement cannot be applied without qualification to a predictor selected to minimize that very sample's error. Selection makes the predictor depend on favorable features of the sample.

### Generalization is an evaluation claim

The **generalization gap** $R_P(f_D)-\widehat R_D(f_D)$ compares population and training loss for the fitted predictor. It is random because the training sample, and possibly the algorithm, are random. A small training error establishes fit to observed examples. A small gap says that this particular population average is close to that fit. Neither quantity identifies whether the chosen population is the right one for the user.

Suppose the assistant memorizes the exact sales prices of every training property. Its training loss is zero. For an unseen property it always outputs zero, so its deployment performance is disastrous. This counterexample requires no sophisticated mathematics: success on observations used for fitting and success on new observations are distinct propositions.

Conversely, more experience need not improve every measured score monotonically. Added data can come from another market, labels can become noisier, and a fixed computation budget can allow fewer passes over a larger sample. Even under a stable population, finite test scores fluctuate. A claim that a procedure learns should specify the source of experience, the procedure, and the evaluation protocol, rather than asserting that each extra example must help.

The application also separates **training** from **inference**. Training changes fitted parameters using data; inference applies the fitted system to a current input. A prompt can change an answer without changing model parameters. In [Brown and colleagues' GPT-3 study](https://arxiv.org/abs/2005.14165), task demonstrations were placed in the prompt for few-shot evaluation without task-specific gradient updates. This is a concrete example of changed behavior through conditioning, distinct from weight updates.

## 2. Supervised learning begins with the target [09:00](https://www.youtube.com/watch?v=DATnpGoGhM8&t=540s)

### Regression, classification, and structured outputs

In **supervised learning**, examples supply both inputs and desired outputs. **Regression** predicts a numerical quantity, such as a sale price. **Classification** predicts a member of a discrete set, such as detached house or townhouse. An input may be a high-dimensional image in either case. The distinction concerns the target and decision problem, not whether the model uses neural networks.

Outputs can also be structured. A bounding box is often represented by four real coordinates; a translation is a variable-length sequence of discrete tokens. Such problems combine multiple predictions with consistency requirements. For a box, the left coordinate should not exceed the right coordinate. For a generated sentence, later token predictions depend on earlier ones. Treating every output component independently can discard these relationships.

The [AlexNet paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) is a primary example of large-scale supervised image classification with a deep convolutional network. Its historical benchmark result concerns a specific classification protocol. It should not be summarized as a general percentage of “vision understood,” nor should top-one and top-five metrics be silently interchanged. The relevant teaching point is the alignment among labeled data, architecture, training procedure, and measured task.

### A loss determines which numerical prediction is best

Fix a property description $x$ and write $m(x)=\mathbb E[Y\mid X=x]$ for the conditional mean sale price. Assume the conditional second moment exists. For a proposed prediction $a$, add and subtract $m(x)$ inside squared loss. The cross term has conditional expectation zero, yielding

$$\mathbb E[(Y-a)^2\mid X=x]
=\operatorname{Var}(Y\mid X=x)+(m(x)-a)^2.$$

Here $\operatorname{Var}$ denotes variance, the expected squared deviation from the mean. Only the second term depends on the choice $a$, so the conditional mean minimizes squared error. This conclusion does not assume Gaussian prices. A restricted hypothesis class may nevertheless fail to represent the conditional mean, which motivates the modeling choices of Lecture 2.

Consider three equally likely prices, 300, 300, and 900, in thousands of dollars, for houses indistinguishable in the recorded features. The mean is 500. Its mean squared error is $(40000+40000+160000)/3=80000$, in squared-thousand-dollar units. Predicting 300 has mean squared error 120000, but mean absolute error 200 rather than $800/3$. Absolute loss favors a median, here 300. The average price and the most useful decision need not coincide.

If the assistant will support an asymmetric decision, such as whether to commission an inspection, squared error may not be the final performance measure. Probability estimation can be an intermediate step, followed by a decision rule reflecting the actual costs. This separation becomes essential when class frequencies or consequences are unequal.

### Derive the binary decision threshold

Let $Y\in\{0,1\}$ indicate whether a property has a costly defect, and let $p(x)=P(Y=1\mid X=x)$. Suppose a false positive costs $c_{\rm FP}\gt0$, a false negative costs $c_{\rm FN}\gt0$, and correct decisions have zero loss. Predicting 1 has conditional expected cost $c_{\rm FP}(1-p)$; predicting 0 has cost $c_{\rm FN}p$. Choosing the cheaper action gives

$$\text{predict 1 when}\quad
p(x)\gt\frac{c_{\rm FP}}{c_{\rm FP}+c_{\rm FN}}.$$

With costs 1 and 9, the threshold is $0.1$. At a defect probability of $0.2$, predicting 1 costs $0.8$ on average, while predicting 0 costs $1.8$. The threshold $0.5$ would optimize the wrong decision problem. This calculation assumes the probability estimate is appropriate for the deployment population and that the stated cost table captures the consequences. If an inspection has variable cost or imperfect sensitivity, those outcomes belong in a richer decision model.

## 3. Unlabeled data still require a chosen objective [19:30](https://www.youtube.com/watch?v=DATnpGoGhM8&t=1170s)

### Similarity is not supplied by the data alone

In **unsupervised learning**, a dataset does not provide a desired target for each observation. It may still contain structure useful for a task. Our assistant could group property descriptions without sale-price labels, cluster documents by topic, or compress a large set of measurements into a smaller representation. Each operation requires an objective that defines which structure matters.

For a simple clustering model, let $z_i\in\mathbb R^d$ be a property's numerical representation, let $K$ be a chosen number of clusters, let $c_i\in\{1,\ldots,K\}$ assign property $i$ to a cluster, and let $\mu_k\in\mathbb R^d$ be cluster $k$'s center. A possible objective is

$$\sum_{i=1}^n\|z_i-\mu_{c_i}\|_2^2.$$

The Euclidean norm $\|v\|_2$ measures the square root of the sum of squared coordinates. This objective favors compact groups under that geometry. It does not discover a unique natural taxonomy. If one coordinate is area in square feet and another is number of bedrooms, the area coordinate can dominate numerically. Changing units without appropriate scaling can change the clustering, even though the houses did not change.

For the one-dimensional points $0,1,10,11$ with two clusters, assigning the first pair together and the second pair together gives centers $0.5$ and $10.5$, and total squared distance $1$. The crossed partition $\{0,10\},\{1,11\}$ gives centers 5 and 6 and total squared distance 100. The objective distinguishes these partitions decisively. It does not establish that either cluster is a causal category or the right grouping for an insurance decision.

The gene-expression and document examples in the recording illustrate this distinction. Reordering a table may reveal co-occurrence or correlated expression. Correlation alone does not show that one gene causes a disease, or that every document in a word-frequency cluster has the same meaning. The objective and the downstream validation determine what the pattern can support.

### Representations change the geometry

An **embedding** is a map $\phi:\mathcal X\to\mathbb R^d$ from objects to numerical vectors. A document embedding can make semantically related descriptions close even when they use different words. But closeness depends on what the representation was trained to preserve. Shared topic, substitutability, contradiction, and causal relation are different relationships; a single distance cannot be assumed to encode them all faithfully.

[Mikolov and colleagues](https://arxiv.org/abs/1301.3781) studied efficient word representations and evaluated syntactic and semantic relationships. Their work supplies a primary reference for the recording's vector-representation discussion. An analogy represented approximately by a vector offset is an empirical regularity of a learned representation, not an algebraic law of language. A property assistant must validate retrieval quality on its actual document questions rather than assuming a generic embedding's geometry is universally suitable.

### Self-supervision constructs targets from observations

In **self-supervised learning**, targets are obtained from the data itself. Let $(w_1,\ldots,w_T)$ be a token sequence of length $T$, with tokens drawn from a finite vocabulary. Let $p_\theta(w_t\mid w_{<t})$ be a model's probability for token $w_t$ given the preceding tokens, where $\theta$ denotes fitted parameters and $w_{<t}$ denotes the prefix. A next-token objective is

$$L(\theta)=-\sum_{t=1}^T\log p_\theta(w_t\mid w_{<t}).$$

The observed next token supplies the target, so the individual prediction is supervised even though no human separately labels each prefix. The negative logarithm penalizes assigning low probability to the observed continuation. By the probability chain rule, the sum is the negative log probability of the full sequence under the model. Lecture 14 develops this factorization and its computational consequences.

For observed-token probabilities $0.5$, $0.25$, and $0.1$, the sequence probability is $0.0125$, and the loss is $-\log(0.0125)\approx4.3820$ using natural logarithms. This number measures prediction of the observed sequence. It does not directly measure truthfulness, usefulness, or the success of a property analysis. Those require additional data, objectives, or evaluation procedures.

## 4. Reinforcement learning makes actions part of the data process [28:00](https://www.youtube.com/watch?v=DATnpGoGhM8&t=1680s)

### A prediction does not describe an intervention

Our assistant can stop with its current evidence or retrieve another inspection report. The latter action changes the information available for the next decision and incurs a cost. This is a sequential decision problem: actions affect what happens later, including the observations from which the system can learn.

Let $s_t$ be the environment state at time step $t$, $a_t$ an action, and $r_t$ a numerical reward. A **policy** $\pi(a\mid s)$ gives the probability of selecting action $a$ in state $s$. The next state is drawn from an environment transition distribution conditional on the current state and action. For a finite horizon $T$ and discount factor $\gamma\in[0,1]$, a policy objective is

$$J(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].$$

The expectation averages over initial conditions, sampled actions, and environment randomness. The discount makes later rewards count less when $\gamma\lt1$; in a finite-horizon problem, $\gamma=1$ is also well defined. The return is the entire discounted sum, whereas a reward is one term. Maximizing immediate reward can therefore choose the wrong action.

Suppose stopping yields reward 1. Searching costs 1 immediately and, with probability $q$, enables reward 4 on the next step; otherwise the later reward is zero. With $\gamma=0.9$, the search action has expected return $-1+3.6q$. It beats stopping exactly when $q\gt5/9$. At $q=0.8$, searching gives 1.88. This original example turns “more information is useful” into a quantitative claim with a cost and a success probability.

### Why sampling does not eliminate gradient methods

A generated token is discrete, so ordinary differentiation through the sampled token's identity is generally unavailable. But the probability of selecting that token can be differentiable in the model parameters. This distinction underlies policy gradients, developed later in the course. It does not mean reinforcement learning is the only conceivable technique for a nondifferentiable computation; it means a useful derivative can be obtained by differentiating the sampling distribution.

For a minimal demonstration, consider two actions with fixed rewards $r_0$ and $r_1$. Let the probability of action 1 be $p_\theta=1/(1+e^{-\theta})$, the sigmoid of a scalar parameter $\theta$. Then $J(\theta)=(1-p_\theta)r_0+p_\theta r_1$, so

$$\frac{dJ}{d\theta}=p_\theta(1-p_\theta)(r_1-r_0).$$

Alternatively, if $A\in\{0,1\}$ is sampled from this distribution, the derivative of its log probability is $A-p_\theta$. Averaging $r_A(A-p_\theta)$ over the two actions gives the same expression. The reward is not differentiated; the probabilities are. With rewards 0 and 2 and probability $1/4$, both calculations give $3/8$.

The following standard-library code checks that identity by exact enumeration. It accepts only the fixed finite two-action construction, has no file or network effects, and prints no sampled estimate. The assertion compares two independent calculations, so it tests the mechanism rather than relying on a noisy simulation.

```python
from math import isclose, log

p = 0.25
rewards = [0.0, 2.0]
probabilities = [1.0 - p, p]
score_gradient = sum(
    probabilities[action] * rewards[action] * (action - p)
    for action in [0, 1]
)
direct_gradient = p * (1.0 - p) * (rewards[1] - rewards[0])
assert isclose(score_gradient, direct_gradient)
assert isclose(score_gradient, 0.375)

token_probabilities = [0.5, 0.25, 0.1]
sequence_loss = -sum(log(value) for value in token_probabilities)
assert isclose(sequence_loss, -log(0.0125))
```

The token-loss check implements the previous section's chain-rule calculation. The reward-gradient check illustrates a different objective on discrete outputs. Sharing a language-model architecture does not make those objectives equivalent.

## 5. Research extension: a feedback loop needs a trustworthy signal [32:00](https://www.youtube.com/watch?v=DATnpGoGhM8&t=1920s)

### Generation, verification, and another training round

The recording uses theorem proving to illustrate self-generated training data: propose proofs, verify candidates, and train on successful results. Our assistant could analogously generate document-grounded answers and retain those whose cited evidence actually supports the conclusion. The quality of the loop depends on the verifier. If incorrect answers receive positive labels, repeated training can amplify the error rather than improve the system.

Let $C$ mean that a generated candidate is correct and $V$ that the verifier accepts it. Write $p=P(C)$ for the candidate success rate, $t=P(V\mid C)$ for verifier sensitivity, and $f=P(V\mid\neg C)$ for its false-positive rate. Bayes' rule gives the correctness of accepted candidates:

$$P(C\mid V)=\frac{pt}{pt+(1-p)f}.$$

For $p=0.2$, $t=0.9$, and $f=0.1$, accepted precision is $0.18/(0.18+0.08)\approx0.6923$. Even a verifier that rejects 90% of incorrect candidates leaves roughly 31% incorrect examples among the accepted set in this scenario. Reducing $f$ to $0.01$ raises precision to about $0.9574$. The base rate matters as much as the verifier's isolated error rate.

[STaR, by Zelikman and colleagues](https://arxiv.org/abs/2203.14465), provides a primary example of bootstrapping reasoning through generated rationales and training on selected outcomes. Its particular procedure and empirical results do not establish that arbitrary self-training loops improve indefinitely. The equation above is an independent selection model; it highlights one condition to test before treating accepted model outputs as reliable training labels.

### Average performance can conceal concentrated failure

Let $G$ denote a subgroup, such as a property type, and let $R_g$ be the predictor's expected loss within group $g$. The law of total expectation gives $R=\sum_gP(G=g)R_g$. A subgroup with probability $0.001$ can suffer a loss increase of 100 while increasing the overall average by only $0.1$. Good average error can coexist with serious concentrated failure.

This is not an argument against averages. It identifies the information they discard. For the property assistant, evaluate ordinary houses, unusual construction types, missing-data cases, and changed markets separately when those distinctions matter to use. Keep answer quality, retrieval cost, latency, and unsupported claims separate before applying a chosen utility rule. A single score hides which trade-off produced an improvement.

The taxonomy now becomes operational. Supervised learning fits specified targets, unsupervised or self-supervised learning extracts information under constructed objectives, and reinforcement learning optimizes consequences under an action-dependent process. A modern assistant can use all three. The categories describe parts of its learning procedure; they are not mutually exclusive species of application.

## Limitations and exercises

This introductory chapter defines problems and demonstrates their simplest mathematical consequences. It does not establish sample-complexity guarantees, solve representation learning, or certify a feedback verifier. Its toy calculations use known probabilities and costs that a real application would have to estimate. The following chapters develop the models, optimization methods, and evaluation assumptions needed to make those estimates useful.

1. **Derive.** False-positive cost is 2 and false-negative cost is 8. Derive the positive-decision threshold and compare both expected costs at probability $0.3$.
2. **Compute.** For prices 300, 300, and 900, verify that the mean minimizes squared error. Give the mean absolute error of predicting 500.
3. **Find the hidden assumption.** A document clustering changes when floor area is converted from square feet to square meters. Does this show the optimizer is broken?
4. **Compute and interpret.** A verifier has sensitivity $0.95$, false-positive rate $0.02$, and candidate correctness rate $0.1$. What fraction of accepted candidates are correct? What happens when candidate correctness rises to $0.5$?
5. **Design.** Specify a test that distinguishes improvement from prompt examples from improvement through parameter training in the property assistant. State what must be held fixed.

<details><summary>Solutions</summary>

1. Predicting positive costs $2(1-p)$, and predicting negative costs $8p$. The inequality $2-2p\lt8p$ reduces to $p\gt0.2$. At $p=0.3$, positive cost is $1.4$ and negative cost is $2.4$, so the positive decision is preferable. This assumes zero cost for correct decisions and that the two error costs do not vary across properties. At exactly $p=0.2$, either action has equal expected cost.

2. The mean is $(300+300+900)/3=500$. Writing the average squared error as the variance around 500 plus $(a-500)^2$ shows that every other prediction adds a nonnegative term. At 500 the squared error is 80000. Its absolute error is $(200+200+400)/3=800/3$, approximately $266.67$ thousand dollars. The median 300 instead gives absolute error 200, demonstrating that changing the loss changes the preferred prediction.

3. No. Euclidean distance on raw coordinates depends on scale. Converting one feature changes its weight relative to others, which changes the objective geometry. First decide whether raw physical distance, standardized variation, or a task-specific metric is intended. Then transform the data consistently and rerun the algorithm. Even standardization is a modeling choice: equalizing coordinate variance does not prove equal relevance to the downstream task.

4. At correctness rate $0.1$, accepted precision is $0.095/(0.095+0.9\cdot0.02)=0.095/0.113\approx0.8407$. At rate $0.5$, it is $0.475/(0.475+0.5\cdot0.02)=0.475/0.485\approx0.9794$. The verifier has not changed, but incorrect candidates form a much smaller share of what it sees. Its precision therefore improves. Reporting sensitivity and false-positive rate without the candidate distribution cannot determine the quality of selected training data.

5. Fix an evaluation set of unseen property questions, a base model checkpoint, decoding settings, retrieval access, and a scoring procedure. First compare the frozen model with and without prompt demonstrations; any improvement occurs through changed conditioning. Separately train a copy of the checkpoint on a stated dataset and evaluate it with the same prompt conditions as the base model. Keep evaluation examples out of both demonstrations and training. Repeat stochastic generation or use paired seeds where meaningful. These interventions identify different sources of improvement; changing prompts and weights simultaneously would confound them.

</details>

## Primary sources and continuation

- [Original recording](https://www.youtube.com/watch?v=DATnpGoGhM8): learning taxonomy, supervised examples, representations, and interactive learning.
- [Krizhevsky, Sutskever, and Hinton, ImageNet Classification](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html): supervised deep image classification.
- [Mikolov et al., Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781): learned vector representations.
- [Brown et al., Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): prompt conditioning without task-specific weight updates.
- [Zelikman et al., STaR](https://arxiv.org/abs/2203.14465): a concrete bootstrapping procedure for reasoning.
- Continue with [Lecture 2](lecture-02.html), which turns the prediction setup into a fully derived and executable least-squares model.

# Lecture 14 — Autoregressive language models and the mechanics of attention

The inspection assistant has retrieved the right report. It must now produce an answer token by token, preserve a negation such as “no structural movement,” and relate a measurement to the correct building. What probability model generates this sequence, and what computation allows a token to use the relevant earlier passage?

We build one model from the outside inward: text becomes token IDs; token IDs select embedding vectors; a causal transformer produces logits; softmax defines conditional probabilities; their product defines a sequence distribution. This is the transcript's substantive order. The worked archive examples, formal derivations, and code are original teaching constructions. Research extensions clarify tokenization and exact memory-efficient attention using primary sources. Visual slides were not reviewed.

## 1. Tokenization chooses the units of the model [01:48](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=108s)

Let $\mathcal V$ be a finite vocabulary of size $V$. A tokenizer maps a text string into a sequence $x_1,\ldots,x_T$, where each $x_t$ is an integer identifying a vocabulary item. A token can be a word, a word fragment, punctuation, whitespace-bearing text, or another unit defined by the tokenizer. Its numeric ID is a category label: token 400 is not twice the meaning of token 200.

Character-level units keep the vocabulary small but make sequences long. Whole-word units make common words compact but need a strategy for rare or unseen words. Subword methods compromise by reusing frequent pieces. The exact segmentation depends on the tokenizer and its version; a pedagogical split of “internationalization” is not a claim about every deployed model.

[Sennrich, Haddow, and Birch](https://aclanthology.org/P16-1162/) introduced a widely used application of byte-pair encoding to subword translation. A simplified training procedure starts with small symbols and repeatedly merges the most frequent adjacent pair in a corpus. The learned merge order becomes part of tokenization. Byte-based and character-based variants differ in their base alphabet and handling of text.

For a miniature corpus with “low” appearing twice and “lower” once, the adjacent pairs “l o” and “o w” each occur three times if words begin as character sequences. A tie rule can first merge “l o” into “lo”; the pair “lo w” then occurs three times and can become “low.” The rare word can share “low” with the common one. Real tokenizers add boundary conventions and other details, so this is an algorithmic illustration rather than a compatible tokenizer implementation.

### Tokenization changes both statistics and cost

If a report uses twice as many tokens under another tokenizer, it consumes twice as many sequence positions. Dense attention's pairwise computation may then grow by roughly four, with width fixed. Conversely, expanding a vocabulary increases input and output table sizes. A vocabulary of $250{,}000$ items needs at least $\lceil\log_2 250{,}000\rceil=18$ bits to distinguish IDs in an ideal fixed-length code; implementation storage can use wider integer types.

Per-token likelihood and perplexity depend on these units. Comparing perplexities across unrelated tokenizers without accounting for segmentation is misleading. For the archive, preserve tokenizer identity with the checkpoint and compute context budgets on actual tokenized text rather than word counts.

## 2. The chain rule turns a sequence into conditional decisions [10:54](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=654s)

For fixed length $T$, a categorical table over every sequence would have $V^T$ possible entries. The probability chain rule states

$$p_\theta(x_{1:T})=\prod_{t=1}^T p_\theta(x_t\mid x_{<t}),$$

where $x_{1:T}$ denotes the whole sequence and $x_{<t}$ its prefix before position $t$. This identity is exact and requires no Markov approximation. Efficiency comes from parameterizing all conditional distributions with shared model parameters $\theta$, rather than storing a table for every possible prefix.

Variable-length generation also needs a stopping convention, commonly an end-of-sequence token or a maximum length. A model that never terminates is not automatically a normalized distribution over finite text strings merely because every local softmax sums to one.

Let $E\in\mathbb R^{V\times d}$ be the learned embedding table, with model width $d$. Token $x_t$ selects row $E_{x_t,:}$. This lookup initially depends only on the token ID. Contextual hidden states later depend on the prefix. Equal token IDs share an initial vector, while the same word in two contexts can acquire different hidden states.

Introduce a beginning token $x_0$. A causal network maps the prefix $x_0,\ldots,x_{t-1}$ to logits $z_t\in\mathbb R^V$. Define

$$p_\theta(x_t=j\mid x_{<t})
=\frac{\exp(z_{t,j})}{\sum_{k=1}^V\exp(z_{t,k})}.$$

The logit vector is not itself a probability distribution. Softmax converts it into positive entries summing to one, and selecting entry $x_t$ gives the probability assigned to the observed token.

Suppose a three-token report fragment receives conditional probabilities $0.5,0.25,0.1$. Its probability is $0.0125$, its negative log-likelihood is $-\log0.0125\approx4.3820$ nats, and its average loss is approximately $1.4607$ nats per token. A nat uses the natural logarithm. The corresponding perplexity is $\exp(1.4607)\approx4.3089$, the inverse geometric mean of the assigned probabilities.

No individual conditional probability establishes factual truth. A plausible continuation can still attach a correct measurement to the wrong building. The language objective models observed text, while the assistant's operational objective concerns supported answers.

## 3. Sampling and likelihood use the same probabilities differently [26:24](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=1584s)

Generation starts with a prefix, samples the next token, appends it, and repeats. Likelihood evaluation starts with a complete observed sequence and asks how probable each actual token was given its actual prefix. These are different computations over the same conditional model.

At positive temperature $\tau$, decoding uses

$$p_\tau(j)=\frac{\exp(z_j/\tau)}{\sum_k\exp(z_k/\tau)}.$$

For two tokens, the probability ratio is $\exp[(z_i-z_j)/\tau]$. Smaller $\tau$ magnifies logit differences without changing their order. For logits $(2,1,-1)$ and $\tau=1$, probabilities are approximately $(0.7054,0.2595,0.0351)$. At $\tau=0.01$, the first token receives probability extremely close to one.

The limit $\tau\to0^+$ concentrates on maximizing logits; if several are tied, the mathematical limit splits mass among them. “Temperature zero” in an implementation usually requests a greedy selection rule rather than literal division by zero. Reproducibility can still depend on ties and numerical execution.

Top-$k$ sampling keeps the $k$ largest probabilities and renormalizes them. It changes the distribution by assigning zero probability outside the retained set. This can be useful for generation, but the resulting text is not sampled from the original temperature-one model.

### Greedy token choices need not maximize sequence probability

Suppose the first token has probabilities $P(A)=0.6$ and $P(B)=0.4$. After $A$, the most probable second token has conditional probability $0.5$; after $B$, a second token has conditional probability $0.99$. Greedy decoding chooses the $A$ branch and obtains a sequence of probability $0.3$. The best $B$ sequence has probability $0.396$. Local maximization does not generally solve the global sequence optimization problem.

Training minimizes

$$L(\theta)=-\sum_{t=1}^T\log p_\theta(x_t\mid x_{<t}).$$

This is a sum over every selected target position, not merely the final token. In teacher forcing, each target sees the observed prefix during training. For input positions containing $(x_0,x_1,\ldots,x_{T-1})$, targets are $(x_1,x_2,\ldots,x_T)$. The one-position shift is essential: allowing a position to read its target turns prediction into copying.

The following numerical check computes stable token losses from finite logits and integer target IDs. It returns a loss per position, rejects incompatible inputs, and has no external effects. It does not implement the transformer producing the logits.

```python
import numpy as np


def token_losses(logits, targets):
    logits, targets = np.asarray(logits, float), np.asarray(targets)
    if logits.ndim != 2 or not all(logits.shape):
        raise ValueError("Use a nonempty positions-by-vocabulary matrix")
    if targets.shape != (len(logits),) or targets.dtype.kind not in "iu":
        raise ValueError("One integer target is required per position")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("Target outside vocabulary")
    shifted = logits - logits.max(axis=1, keepdims=True)
    return np.log(np.exp(shifted).sum(axis=1)) - shifted[
        np.arange(len(logits)), targets
    ]


probabilities = np.array([[0.5, 0.5], [0.75, 0.25], [0.9, 0.1]])
losses = token_losses(np.log(probabilities), np.array([0, 1, 1]))
assert np.isclose(losses.sum(), -np.log(0.0125))
assert np.isclose(np.exp(losses.mean()), 80 ** (1 / 3))
assert np.isclose(token_losses([[1000, -1000]], np.array([1]))[0], 2000)
```

## 4. Attention makes information flow depend on content [39:27](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=2367s)

A transformer alternates attention with a positionwise feedforward network, also called a multilayer perceptron or MLP. The MLP shares weights across positions but acts on each position separately. Attention moves information between positions. Removing all cross-position operations leaves a model unable to relate a measurement to an earlier building name.

For one attention head, collect hidden row vectors in $H\in\mathbb R^{T\times d}$. Learned projections produce

$$Q=HW_Q,\quad K=HW_K,\quad V_{\rm val}=HW_V,$$

where $W_Q,W_K\in\mathbb R^{d\times d_k}$ and $W_V\in\mathbb R^{d\times d_v}$. The subscript on $V_{\rm val}$ distinguishes the value matrix from vocabulary size $V$. Queries and keys have dimension $d_k$ so their inner products are defined; values can have another dimension $d_v$.

The unmasked attention operation is

$$A=\operatorname{softmax}_{\rm row}(QK^\top/\sqrt{d_k}),
\qquad O=AV_{\rm val}.$$

Rowwise softmax means that each row is normalized separately. At position $i$,
$O_i=\sum_j A_{ij}v_j$, a convex combination of value vectors. Queries ask which information to use, keys determine compatibility, and values carry the combined content. These are computational roles, not externally supplied annotations.

### Why scale the dot product?

Under the illustrative assumption that query and key coordinates are independent, zero-mean, unit-variance random variables, $q^\top k$ has variance $d_k$. Dividing by $\sqrt{d_k}$ makes the variance one. Without this scaling, large dimensions can produce large logit differences and saturated softmax probabilities. The assumption motivates the factor; learned coordinates need not remain independent.

For a scalar query $q=1$, keys $(0,\log3)$, and scalar values $(2,8)$, the attention probabilities are $(1/4,3/4)$ and output is $6.5$. Changing values changes the content of the output; changing keys changes which content is selected. This separation explains why replacing all three projections with one fixed average restricts the model.

[Vaswani et al.](https://arxiv.org/abs/1706.03762) supplies the primary transformer formulation. The original architecture is an encoder-decoder model, not identical to every modern decoder-only language model. Positional information is also necessary to represent order robustly: unmasked self-attention without positional signals is permutation-equivariant. A causal mask introduces ordering constraints, but positional encodings or related mechanisms give an explicit representation of locations and distances.

## 5. A causal mask makes the probability model implementable [60:18](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=3618s)

Unmasked attention lets each position read later inputs. That violates the prefix-only conditional distribution. Define an additive mask $M\in\mathbb R^{T\times T}$ by $M_{ij}=0$ for $j\le i$ and $M_{ij}=-\infty$ for $j>i$. Then

$$A=\operatorname{softmax}_{\rm row}(QK^\top/\sqrt{d_k}+M).$$

Masked entries become zero after exponentiation, and permitted entries renormalize to sum to one. The diagonal is allowed because the hidden state at input position $i$ predicts the next token, rather than the token already occupying position $i$.

Why does this remain causal through many layers? At the first layer, input state $i$ contains only its input token and positional information. Assume every state at layer $\ell$ and position $j$ depends only on input positions up to $j$. Masked attention at position $i$ reads only $j\le i$, so its output depends only on positions up to $i$. Positionwise MLPs, normalization over features, and residual addition preserve that property. Induction proves causality for the stack.

This proof also identifies a failure condition: an operation that normalizes using statistics across future sequence positions could introduce leakage despite the attention mask. Layer normalization across a position's feature coordinates does not do so.

The following single-head implementation accepts finite compatible matrices and returns outputs and attention weights. It constructs dense scores for clarity, with quadratic storage; it is not a production kernel. Every causal row has at least its diagonal available, preventing an all-masked softmax.

```python
def causal_attention(queries, keys, values):
    queries, keys, values = map(
        lambda x: np.asarray(x, float), (queries, keys, values)
    )
    if queries.ndim != 2 or keys.shape != queries.shape:
        raise ValueError("Queries and keys must have equal matrix shapes")
    if not all(queries.shape) or values.ndim != 2:
        raise ValueError("Use nonempty matrices")
    if values.shape[0] != len(queries) or not values.shape[1]:
        raise ValueError("One nonempty value vector per position is required")
    if not all(np.isfinite(a).all() for a in (queries, keys, values)):
        raise ValueError("Inputs must be finite")
    scores = queries @ keys.T / np.sqrt(queries.shape[1])
    future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    scores[future] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ values, weights


q = np.ones((3, 1))
k = np.zeros((3, 1))
v = np.array([[2.0], [4.0], [9.0]])
output, attention = causal_attention(q, k, v)
assert np.allclose(output[:, 0], [2, 3, 5])
assert np.allclose(attention.sum(axis=1), 1)
assert np.allclose(np.triu(attention, 1), 0)
changed, _ = causal_attention(q, [[0], [0], [20]], [[2], [4], [-100]])
assert np.allclose(changed[:2], output[:2])
```

The final assertion is a behavioral test of causality: changing a future key and value cannot alter earlier outputs. Checking only matrix shapes would not detect that bug.

## 6. Multiple heads and residual blocks compose the computation [66:13](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=3973s)

With $h$ heads, compute $O^{(1)},\ldots,O^{(h)}$ from the same input using different projection matrices, concatenate their feature columns, and apply an output projection:

$$\operatorname{MHA}(H)=
\operatorname{Concat}(O^{(1)},\ldots,O^{(h)})W_O.$$

If each value dimension is $d_v$, then $W_O\in\mathbb R^{hd_v\times d}$. The heads operate in parallel, not as successive layers. Their distinct parameters permit distinct routing patterns, but a particular head is not guaranteed to have a clean human-interpretable specialty.

For the common choice $d_k=d_v=d/h$, the three input projections and one output projection contain approximately $4d^2$ weights, ignoring biases. Increasing the number of heads at fixed model width does not multiply this count by $h$; it changes how the width is partitioned. At $d=4,h=2$, each head has width two and the attention projection count is $64$.

A representative pre-normalized block is

$$\tilde H=H+\operatorname{MHA}(\operatorname{Norm}(H)),\qquad
H'=\tilde H+\operatorname{MLP}(\operatorname{Norm}(\tilde H)).$$

Normalization acts independently on each row's features. Other architectures place normalization differently. Residual addition retains a direct path from the input while learned branches modify it, connecting this architecture to Lecture 7's Jacobian analysis. Attention outputs before projection are convex combinations, but output projections, residuals, and nonlinear MLPs mean the complete block is not restricted to a convex hull of input vectors.

## 7. Exact attention can use less memory without fewer pairs [73:40](https://www.youtube.com/watch?v=pwQ0l4hFCVI&t=4420s)

For dense attention, forming all query-key products costs $O(T^2d_k)$ per head; combining values costs $O(T^2d_v)$. Materializing the attention matrix uses $O(T^2)$ numbers. At $T=32{,}768$, one matrix has $1{,}073{,}741{,}824$ entries, occupying 2 GiB at two bytes per entry, before other tensors and heads.

[Dao et al.'s FlashAttention](https://arxiv.org/abs/2205.14135) changes the execution schedule to reduce transfers between GPU memory levels and avoid storing the full score matrix. Its dense version computes exact attention up to floating-point effects; it does not turn dense attention's arithmetic into linear complexity.

The central algebra can be understood for one query. Given scores $s_j$ and values $v_j$, maintain a running maximum $m$, denominator $l=\sum_j e^{s_j-m}$, and numerator $a=\sum_j e^{s_j-m}v_j$. If a new block has corresponding statistics $(m_b,l_b,a_b)$, set $m'=\max(m,m_b)$ and update

$$l'=e^{m-m'}l+e^{m_b-m'}l_b,\qquad
a'=e^{m-m'}a+e^{m_b-m'}a_b.$$

The output is $a'/l'$. Multiplying old and new sums by their rescaling factors expresses both in the same exponential reference frame. Thus no full row of probabilities need remain in memory.

For scores $(0,\log3)$ and values $(2,8)$, the first singleton block has $(m,l,a)=(0,1,2)$ and the second $(\log3,1,8)$. The combined denominator is $1/3+1=4/3$, numerator is $2/3+8=26/3$, and output is $6.5$, matching ordinary attention. GPU tiling, backward recomputation, and memory layout turn this identity into an efficient kernel.

## Limitations and practical interpretation

Autoregression specifies a probability factorization; attention specifies one architecture for its conditionals. Tokenization, context length, hidden width, number of heads, and implementation schedule affect different costs. A long context does not guarantee that relevant evidence will influence an answer. The inspection assistant still needs tests for negation, entity binding, and citation support.

Training can process many observed positions in parallel because all prefix tokens are supplied and future dependencies are masked. Generation must ordinarily wait for each newly sampled token. Lecture 16 develops caching and architectural choices that address this asymmetry.

## Exercises

1. Compute the negative log-likelihood and perplexity for conditional target probabilities $1/2,1/4,1/10$. Explain why they are tokenizer-dependent.
2. Give a two-step example where greedy decoding misses the most probable sequence. What happens to equal maximal logits as temperature tends to zero?
3. Derive the variance of $q^\top k$ under independent zero-mean, unit-variance coordinates. State an assumption that learned representations may violate.
4. In the causal attention example, change the last value from $9$ to $90$ while keeping all scores zero. Find all outputs and explain the unchanged prefix.
5. Combine blocks with $(m,l,a)=(0,2,6)$ and $(\log2,1,10)$ using the online softmax equations. Explain why this reduces storage without eliminating pairwise score computation.

<details>
<summary>Fully worked solutions</summary>

1. Sequence probability is $1/80$, negative log-likelihood is $\log80\approx4.382027$, and perplexity is $80^{1/3}\approx4.308869$. Average loss divides by three tokens. Another tokenizer changes both the sequence of prediction events and their count, so raw per-token perplexities need not be comparable.
2. Use $P(A)=0.6$, $P(B)=0.4$, with best second-token probabilities $0.5$ after $A$ and $0.99$ after $B$. Greedy yields $0.3$ while the $B$ sequence reaches $0.396$. If $r$ logits tie for the maximum, their exponentials remain equal and dominate all others, giving limiting probability $1/r$ each. An implementation may apply a separate deterministic tie rule.
3. Write $q^\top k=\sum_{j=1}^{d_k}q_jk_j$. Each product has mean zero and variance $\mathbb E[q_j^2]\mathbb E[k_j^2]=1$, and distinct products have zero covariance under the stated independence. The total variance is $d_k$, reduced to one after division by $\sqrt{d_k}$. Learned keys and queries can be correlated and anisotropic, so this is a scaling motivation rather than a maintained distributional theorem.
4. Zero scores give uniform weights over each permitted prefix. Outputs are $2$, $(2+4)/2=3$, and $(2+4+90)/3=32$. The first two rows cannot attend to position three. Causality restricts dependencies; it does not make the final output insensitive to its own value.
5. The new maximum is $\log2$. Rescale the old block by $1/2$: denominator becomes $(1/2)2+1=2$, numerator becomes $(1/2)6+10=13$, and output is $6.5$. Only accumulated statistics and a current block need storage. Every required score still has to be computed in dense exact attention, so the pair count remains quadratic across all queries.
</details>

## Primary sources

- Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).

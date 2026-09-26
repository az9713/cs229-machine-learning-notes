# Lecture 16 — Efficient transformers and the difference between prompting and training

The inspection assistant works for one user, but stalls when many users submit long reports. Adding more expert parameters may improve its language model while making deployment harder. Meanwhile, a few examples in a prompt can alter its behavior without changing a single weight. These observations concern different resources and different meanings of adaptation.

We retain Lecture 14's causal transformer and ask what changes when keys and values are shared, attention is local, or the feedforward network is sparsely activated. We then distinguish conditioning on examples from supervised parameter updates. The recording supplies this topic sequence. Equations, resource calculations, and small implementations are original teaching constructions; research extension claims cite primary papers. Caption wording is corrected where dimensions, historical attribution, or execution details require precision.

## 1. Cached keys and values are reusable state [00:06](https://www.youtube.com/watch?v=hHC-SF3utxg&t=6s)

At a given layer, each input position produces a query, key, and value vector. When a causal model generates a new token, earlier hidden states at that layer cannot depend on the new token. Their keys and values can therefore be reused. The new query compares against those stored keys and combines their values.

This is the key-value cache, usually abbreviated KV cache. It stores intermediate states for a particular prefix, not updated model parameters. Earlier queries ordinarily need not remain in the decoding cache because their outputs have already been computed. Training and prompt processing have different activation requirements; a statement about decoding memory should not be generalized to all phases.

Let $L$ be layer count, $B$ simultaneous sequences, $T$ cached positions per sequence, $h_{\rm kv}$ key-value heads, $d_h$ coordinates per key and value head, and $b$ bytes per coordinate. If every layer uses these dimensions, the cache size is

$$M_{\rm KV}=2LBT h_{\rm kv}d_hb.$$

The factor two counts keys and values. This formula excludes model weights, temporary workspaces, allocator overhead, and unequal sequence lengths.

For $L=32$, $B=1$, $T=8192$, $h_{\rm kv}=32$, $d_h=128$, and $b=2$, the cache occupies $4{,}294{,}967{,}296$ bytes, or 4 GiB. Eight simultaneous sequences require 32 GiB under the same assumptions. Cache memory can therefore constrain batching even if the model weights fit.

### Capacity and bandwidth are different constraints

Memory capacity determines whether state fits. Memory bandwidth determines how quickly stored state can be read. Arithmetic intensity is the ratio of arithmetic operations to bytes transferred. A roofline-style upper bound on achievable throughput is the smaller of peak arithmetic throughput and bandwidth times arithmetic intensity.

This is an explanatory model, not a device benchmark. In decoding, repeatedly loading weights or a growing cache can dominate time. Batching can reuse weights across more work, but requires additional per-sequence cache. The same design change can ease one constraint and leave another largely unchanged.

Prompt processing, often called prefill, computes many positions together. Decoding computes one new position per sequence at a time. Dense prefill includes quadratic attention over the prompt; a cached decoding step attends to a length-$T$ history and costs linear attention work in $T$. Generating many additional tokens still accumulates these growing per-step costs.

## 2. Grouped-query attention reduces the number of stored heads [07:17](https://www.youtube.com/watch?v=hHC-SF3utxg&t=437s)

Ordinary multi-head attention gives each of $h_q$ query heads its own key and value head. Grouped-query attention, or GQA, instead uses $h_{\rm kv}\le h_q$ key-value heads shared across groups of queries. Assume $h_q$ is divisible by $h_{\rm kv}$, and let $g=h_q/h_{\rm kv}$ be the number of query heads per group.

Using zero-based head indices, query head $j$ uses group
$c(j)=\lfloor j/g\rfloor$. Its output is

$$O^{(j)}=
\operatorname{softmax}_{\rm row}
\left(\frac{Q^{(j)}(K^{(c(j))})^\top}{\sqrt{d_h}}+M\right)
V^{(c(j))},$$

where $M$ is the causal additive mask. Each query head still gets its own attention weights. Shared keys do not force identical outputs, because the queries differ.

When $h_{\rm kv}=h_q$, this recovers ordinary multi-head attention. When $h_{\rm kv}=1$, it is multi-query attention. [Shazeer](https://arxiv.org/abs/1911.02150) presents multi-query attention, while [Ainslie et al.](https://arxiv.org/abs/2305.13245) introduces GQA and a method for converting existing checkpoints. These primary sources establish the method's origin; later model adoption should not be mistaken for its invention.

In the earlier cache example, reducing key-value heads from 32 to 8 reduces cache storage from 4 GiB to 1 GiB per sequence. The number of query heads can remain 32. It does not follow that every operation becomes four times cheaper: there are still 32 query-specific score computations and their value combinations.

### A small shared-head calculation

Take two query heads sharing one scalar key sequence $(0,\log3)$ and values $(2,8)$. Query $q_1=1$ produces weights $(1/4,3/4)$ and output $6.5$. Query $q_2=-1$ produces weights $(3/4,1/4)$ and output $3.5$. The cache is shared; the decisions are not.

The following single-position function implements grouped queries over an already available prefix. Inputs are finite arrays shaped as heads-by-dimension for queries and positions-by-groups-by-dimension for keys and values. It returns one output vector per query head. It does not allocate a growing cache or perform projections. Invalid dimensions raise an error; there are no persistent effects.

```python
import numpy as np


def grouped_query_step(queries, keys, values):
    queries, keys, values = map(
        lambda x: np.asarray(x, float), (queries, keys, values)
    )
    if queries.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("Use Q [heads,d] and K/V [time,groups,d]")
    if not all(queries.shape) or not all(keys.shape):
        raise ValueError("Dimensions must be nonzero")
    heads, width = queries.shape
    groups = keys.shape[1]
    if width != keys.shape[2] or heads % groups:
        raise ValueError("Head widths must match and groups must divide heads")
    if not all(np.isfinite(a).all() for a in (queries, keys, values)):
        raise ValueError("Inputs must be finite")
    outputs = []
    for head in range(heads):
        group = head // (heads // groups)
        scores = keys[:, group] @ queries[head] / np.sqrt(width)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        outputs.append(weights @ values[:, group])
    return np.asarray(outputs)


queries = np.array([[1.0], [-1.0]])
keys = np.array([[[0.0]], [[np.log(3)]]])
values = np.array([[[2.0]], [[8.0]]])
assert np.allclose(grouped_query_step(queries, keys, values), [[6.5], [3.5]])
assert 2 * 32 * 8192 * 32 * 128 * 2 == 4 * 1024 ** 3
assert 2 * 32 * 8192 * 8 * 128 * 2 == 1024 ** 3
```

The implementation deliberately keeps keys and values physically shared. Expanding them into repeated arrays would preserve outputs but waste the memory the architecture aims to save. A real kernel must exploit the shared layout and hardware memory hierarchy.

## 3. A sliding window changes which dependencies are possible [25:58](https://www.youtube.com/watch?v=hHC-SF3utxg&t=1558s)

Let a causal window of width $w$ include the current position and the previous $w-1$ positions. Position $i$ can attend only to
$\max(0,i-w+1),\ldots,i$. The score graph now has at most $Tw$ edges instead of order $T^2$. With fixed $w$, pairwise attention arithmetic is linear in sequence length.

This is an architectural restriction, unlike FlashAttention's exact reorganization of dense attention. It removes direct access to distant states. However, a state in a deeper layer may already summarize earlier positions.

### Deriving the receptive field

The receptive field is the set of input positions that can influence an output. One layer can move information at most $w-1$ positions forward. Assume a depth-$\ell$ state at position $i$ depends only on inputs from $i-\ell(w-1)$ through $i$. At the next layer, it reads states as early as $i-(w-1)$; those states can themselves depend on inputs another $\ell(w-1)$ positions earlier. Thus depth $\ell+1$ reaches back $(\ell+1)(w-1)$.

After $L$ layers, the maximum receptive-field width is therefore

$$1+L(w-1),$$

clipped at the beginning of the sequence. With $L=3$ and $w=4$, at most ten input positions can influence the output. A convention defining the window as $w$ previous positions plus the current one instead gives $1+Lw$; the convention must be stated.

This is a possible dependency bound, not evidence that every earlier fact is faithfully retained. If the building identifier falls outside all available paths, a purely local stack cannot use it. If it lies inside, the learned computation can still fail to preserve it.

[Mistral 7B](https://arxiv.org/abs/2310.06825) provides a primary example combining GQA and sliding-window attention. Real architectures may interleave global layers or add other mechanisms, changing the receptive-field calculation. For a purely local layer, decoding can discard keys and values that no future window will access; retained states may carry summaries of older inputs, but the discarded vectors themselves are unavailable.

The recording also considers normalizing queries and keys before their dot product. Root-mean-square normalization divides a vector by the square root of its mean squared coordinate, with numerical stabilization and optional learned scaling. It controls scale differently from unit-length normalization. If coordinates are all one, RMS-normalized coordinates remain one, giving norm $\sqrt{d_h}$, not one. A scaling justification must therefore track the normalization actually used.

## 4. Mixture-of-experts separates stored parameters from active computation [32:44](https://www.youtube.com/watch?v=hHC-SF3utxg&t=1964s)

The transformer feedforward block usually applies the same MLP to every token position. A sparse mixture-of-experts, or MoE, replaces it with several candidate MLPs and a router that selects a small subset for each token at each layer.

Let the hidden vector be $h\in\mathbb R^d$, and let there be $m$ expert functions $E_s:\mathbb R^d\to\mathbb R^d$. A router produces scores $r(h)\in\mathbb R^m$. Select the $k$ largest scores, with index set $S(h)$, and define one possible normalized gate:

$$a_s(h)=\frac{\exp(r_s(h))}
{\sum_{j\in S(h)}\exp(r_j(h))},\quad s\in S(h),\qquad
F(h)=\sum_{s\in S(h)}a_s(h)E_s(h).$$

Only selected experts are evaluated. Other gate functions and normalization conventions exist. A shared expert can add a branch evaluated for every token; it must be counted in active cost.

If each expert contains $P$ parameters, expert storage is approximately $mP$ parameters while expert arithmetic per token is associated with $kP$, ignoring router, attention, communication, and other overhead. With 128 experts and eight selected, the active fraction of expert parameters is $1/16$. It is not $1/16$ of every resource in the model.

A two-byte representation of 30 billion stored parameters alone occupies 60 GB in decimal units, even if only three billion parameters are activated per token. Sparse activation does not mean the remaining weights cease to exist. They may reside on other devices or be moved as needed, which introduces further costs.

### Routing is a learning and scheduling problem

Suppose routing logits are $(\log4,\log2,0)$ and the top two experts return scalar outputs $10$ and $4$. Selected gates renormalize to $(2/3,1/3)$, giving output $8$. The third expert need not execute. If every token chooses the same pair, the unused experts contribute little learned capacity and devices hosting them may sit idle.

[Fedus, Zoph, and Shazeer's Switch Transformer](https://arxiv.org/abs/2101.03961) studies sparse expert routing, including its training and systems difficulties. Load balancing is motivated both by learning and by execution. A batch assigned unevenly across devices can be limited by the busiest device rather than average load.

Top-$k$ selection is discrete. Away from a routing boundary, ordinary differentiation follows the currently selected branch; it does not differentiate a smooth change of the selected set. Gate-weight design matters. If $k=1$ and one renormalizes the selected weight to exactly one, the main output loss has no local gradient through that constant gate into router logits. Some methods retain the selected softmax probability or add auxiliary objectives. The generic normalized formula is therefore not a complete training recipe for every MoE.

The following deterministic check selects experts before evaluating them and records which branches execute. It assumes supplied callables return compatible numeric outputs. It validates logits and selection count; the callables determine any additional effects, so the example uses pure arithmetic functions.

```python
def sparse_experts(hidden, logits, experts, k):
    logits = np.asarray(logits, float)
    if logits.shape != (len(experts),) or not np.isfinite(logits).all():
        raise ValueError("One finite logit per expert is required")
    if not isinstance(k, int) or not 1 <= k <= len(experts):
        raise ValueError("Choose an integer number of active experts")
    selected = np.argsort(-logits, kind="stable")[:k]
    weights = np.exp(logits[selected] - logits[selected].max())
    weights /= weights.sum()
    outputs = np.asarray([experts[i](hidden) for i in selected], float)
    return np.tensordot(weights, outputs, axes=1), selected


experts = [lambda x: 10 * x, lambda x: 4 * x, lambda x: -100 * x]
result, active = sparse_experts(1, np.log([4, 2, 1]), experts, 2)
assert np.array_equal(active, [0, 1])
assert np.isclose(result, 8)
```

Calling experts “mathematics,” “physics,” or “language” is an interpretation that requires measurement. A single mathematical answer can route different tokens through different experts at different layers. Sparse routing does not establish a clean human taxonomy of expertise.

## 5. In-context learning changes the conditional input [51:24](https://www.youtube.com/watch?v=hHC-SF3utxg&t=3084s)

Let $D=((x_1,y_1),\ldots,(x_n,y_n))$ be task demonstrations and $x_*$ a new input. In-context learning formats them into a prompt and obtains

$$p_\theta(y_*\mid D,x_*).$$

The parameters $\theta$ stay fixed. Activations and cached states change because the input changes. In Lecture 12's fine-tuning, by contrast, the data changed the parameters themselves.

For the inspection assistant, demonstrations might map report excerpts to labels “urgent review,” “routine repair,” and “insufficient evidence.” Examples can clarify a boundary that a short instruction leaves ambiguous. A zero-shot prompt provides the task description without demonstrations; zero-shot does not mean the task or its ingredients never appeared during pretraining.

[Brown et al.'s GPT-3 paper](https://arxiv.org/abs/2005.14165) evaluates this no-gradient adaptation protocol. Its findings motivate the method, not a guarantee that arbitrary tasks can be learned from a few examples. Formatting, example order, label balance, and similarity to the query can all matter operationally and should be tested.

### Two examples do not uniquely identify a rule

The demonstrations $2\star3=5$ and $6\star7=13$ suggest addition, yielding $15\star2=17$. Yet the alternative

$$f(a,b)=a+b+(a-2)(a-6)$$

fits both demonstrations and gives $134$ on $(15,2)$. Finite examples underdetermine the rule without an inductive preference. A pretrained model brings such preferences, which can help or conflict with a user's intended mapping.

A useful research-style interpretation is Bayesian conditioning over possible tasks: demonstrations alter a posterior over task rules while the general inference mechanism remains fixed. This is an explanatory model of the behavior, not a claim that every transformer implements exact Bayes. Fixed weights can still implement substantial input-dependent computation; “no weight update” does not mean “no reasoning.”

For a controlled evaluation, reserve test cases by building and time period, keep prompt construction separate from held-out answers, and compare instructions alone with selected demonstrations. Prompting removes a training step, not the need to measure whether the system works.

## 6. Instruction tuning changes the parameters and the output objective [64:18](https://www.youtube.com/watch?v=hHC-SF3utxg&t=3858s)

Supervised fine-tuning, or SFT, continues training a pretrained model on labeled examples. Instruction tuning is SFT whose inputs describe tasks and whose targets demonstrate desired answers. Let example $i$ contain prompt $x_i$ and response tokens $y_{i,1:T_i}$. An example-averaged sequence loss is

$$L_{\rm seq}(\theta)=
-\frac1n\sum_{i=1}^n\sum_{t=1}^{T_i}
\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}).$$

The prompt is conditioning information. In response-only training, prompt positions have zero loss weight, but remain visible in the causal computation. The response tokens are shifted against their preceding context exactly as in Lecture 14.

[Wei et al.'s FLAN paper](https://arxiv.org/abs/2109.01652) supplies primary evidence that instruction tuning can improve zero-shot performance on held-out task types. This is a statement about a training and evaluation setup, not an assertion that every instruction-tuned model generalizes to every new domain.

### Loss normalization is a weighting choice

Token-averaged loss divides the total sum by $\sum_i T_i$. On a fixed dataset this differs from $L_{\rm seq}$ by a global constant, although minibatch sampling and changing batch token counts can affect optimization. In contrast, averaging each example's mean token loss,

$$L_{\rm equal}=
-\frac1n\sum_i\frac1{T_i}\sum_t
\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}),$$

changes the relative weights of long and short responses.

For two examples, one with a single token of loss $2$ and another with three tokens each of loss $1$, the total is $5$. Sequence-averaged summed loss is $2.5$; token-average is $1.25$; equal-example mean loss is $1.5$. The first two have the same relative token weights on this fixed dataset; the third weights each short-response token more heavily.

The difference between masking and detaching also matters. Removing prompt tokens from the loss does not necessarily block all gradients through their hidden states: response predictions can attend to them, so response loss can still train the computations that produced those states.

SFT rewards the demonstrated continuation. If a task admits many correct answers but the dataset supplies one, ordinary token-level likelihood treats the others as alternatives rather than giving them equal outcome credit. Demonstrations can still teach valuable strategies and formats. Outcome-based reinforcement learning supplies a different objective, developed in the remaining chapters; it does not make demonstration quality irrelevant.

## Limitations and a resource-aware comparison

GQA reduces distinct cached key-value heads; sliding windows restrict attention connections; MoE reduces the fraction of expert computation per token; prompting changes context; SFT changes weights. These interventions are not substitutes. For the inspection service, measure peak memory, time to first token, time per generated token, throughput under realistic concurrency, and task accuracy separately.

A concrete design exercise is to hold answer quality approximately fixed while varying one resource constraint at a time. Otherwise a faster configuration may merely be solving an easier problem with less context, fewer active parameters, or shorter answers.

## Exercises

1. Calculate KV cache bytes for $L=24,B=2,T=4096,h_{\rm kv}=8,d_h=128,b=2$. What changes if only query-head count doubles?
2. Four query heads share two key-value groups equally. List their zero-based group indices. Why can outputs within a group differ?
3. Derive the receptive field of five local layers whose width-three window includes the current position. What changes if every second layer is global?
4. Explain why top-one routing with renormalization to one can deprive the router of a main-loss gradient. Distinguish this from gradients into the selected expert.
5. Compute the three loss normalizations for a two-token response with losses $(1,3)$ and a one-token response with loss $6$. Explain why a response-only mask still permits gradients through prompt states.

<details>
<summary>Fully worked solutions</summary>

1. The cache is $2(24)(2)(4096)(8)(128)(2)=805{,}306{,}368$ bytes, or $0.75$ GiB. Doubling only query heads leaves this formula unchanged. It can increase query-related computation, so unchanged cache size is not unchanged execution cost.
2. The mapping is $(0,0,1,1)$ because each group serves two query heads. Shared keys and values still interact with different query vectors. Different dot products yield different softmax weights and therefore different weighted values, as the $6.5$ versus $3.5$ example demonstrates.
3. Each layer adds at most two earlier positions, giving width $1+5(2)=11$. The derivation uses induction over layer depth. A global causal layer can read every preceding position, so the pure-local bound no longer applies above it; all available earlier positions may have a dependency path.
4. On a region with unchanged selected index, the normalized gate is exactly one, whose derivative is zero. The output is simply the selected expert's output, with no differentiable dependence on router scores inside that region. Expert parameters still affect that output and receive gradients. Router training requires another differentiable path or objective, or a gate convention that retains score dependence.
5. Total loss is $1+3+6=10$. Sequence-averaged summed loss is $10/2=5$; token-average is $10/3$; equal-example average is $[(1+3)/2+6]/2=4$. Response loss depends on prompt hidden states through attention, so the chain rule sends gradients through those states even when prompt positions themselves contribute no direct prediction loss.
</details>

## Primary sources

- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150).
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245).
- Jiang et al., [Mistral 7B](https://arxiv.org/abs/2310.06825).
- Fedus, Zoph, and Shazeer, [Switch Transformers](https://arxiv.org/abs/2101.03961).
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- Wei et al., [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652).

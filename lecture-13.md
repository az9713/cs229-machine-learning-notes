# Lecture 13 — Contrastive representations, semantic search, and grounded generation

An inspector asks, “Which earlier reports describe a horizontal foundation crack?” The archive contains photographs, repair invoices, inspection reports, and documents about unrelated cosmetic cracks. A keyword match may miss a paraphrase; a visually similar photograph may depict the wrong defect. We need a learned notion of relevance, an efficient way to search it, and an answer that remains accountable to the retrieved evidence.

The recurring model in this chapter is an encoder $\phi_\theta:\mathcal X\to\mathbb R^m$, where $\mathcal X$ is an input space, $m$ an embedding dimension, and $\theta$ learned parameters. Contrastive training shapes its geometry; retrieval searches that geometry; a language model then conditions on selected documents. The transcript supplies this sequence and its questions about collapse, negative pairs, and search. Derivations, numerical examples, and code are original constructions. Research extension sections distinguish the precise published methods from the simplified classroom model.

## 1. What should an embedding preserve? [00:05](https://www.youtube.com/watch?v=lNTajqxxOn4&t=5s)

An embedding is a vector representation of an object. “Semantic” describes the intended similarity, not a mathematical guarantee. Two photographs of the same defect under different lighting should be near each other for one application; two different defects requiring the same repair may need to be near each other for another. These are different equivalence relations.

For nonzero vectors $u,v\in\mathbb R^m$, cosine similarity is

$$\operatorname{cos}(u,v)=\frac{u^\top v}{\|u\|_2\|v\|_2}.$$

The Euclidean norm $\|u\|_2$ is the square root of the sum of squared coordinates. If both vectors are normalized to length one, cosine similarity is their inner product and

$$\|u-v\|_2^2=\|u\|_2^2+\|v\|_2^2-2u^\top v=2-2u^\top v.$$

Thus maximizing cosine and minimizing Euclidean distance give the same ranking for normalized embeddings. Without normalization, the equivalence fails. For query $(1,0)$, documents $(2,2)$ and $(1,0)$ have inner products $2$ and $1$, yet cosine similarities $1/\sqrt2$ and $1$. The larger norm reverses the preferred result.

Raw pixels also form a vector, but their distance is not automatically useful. A one-pixel shift can move every edge while preserving the depicted defect. A representation must learn or be designed to tolerate the transformations relevant to the task.

### Supervised pretraining gives one possible representation

Train a classifier with logits $W\phi_\theta(x)$, where $W$ maps features to class scores. Its penultimate layer can become a reusable representation after the classification head is removed. Good classification does not identify one unique geometry. For any invertible matrix $M$, replace features by $M\phi_\theta(x)$ and the head by $WM^{-1}$; logits remain unchanged while distances can change substantially.

Consequently, accurate prediction of pretraining labels does not prove that arbitrary nearest-neighbor queries will work. Labels also constrain what must be represented. A network trained only to distinguish color from grayscale need not preserve structural damage. For the inspection archive, this motivates learning from paired views and query-document relationships rather than assuming a generic classification head has already solved retrieval.

## 2. Positive pairs define invariance; collapse defines the failure [12:25](https://www.youtube.com/watch?v=lNTajqxxOn4&t=745s)

Let $a$ and $a'$ be random augmentation functions, such as crops or color transformations. Given image $x$, the two views $a(x)$ and $a'(x)$ form a positive pair. The intended signal is that these views should represent the same underlying content. A simple attraction loss is

$$L_{\rm attract}=\mathbb E\left[
\|\phi_\theta(a(x))-\phi_\theta(a'(x))\|^2
\right].$$

The constant encoder $\phi_\theta(x)=c$ makes this loss zero for every input. This is representation collapse: distinct inputs become indistinguishable. Even fixing all output norms to one does not prevent collapse; the same unit vector can represent everything.

Contrastive learning adds competing candidate objects. A positive should score above the alternatives, commonly called negatives. A negative is a training role, not a certified semantic fact. Two separate reports about the same crack may be sampled as negatives and pushed apart.

### Augmentation is an assumption about the answer

If a crop removes the only visible crack, treating cropped and full images as equivalent encourages ignoring the crack. This can be useful for object identity and harmful for defect localization. Likewise, deleting “not” from an inspection sentence reverses its meaning. Augmentations are therefore part of the statistical model, rather than interchangeable cosmetic perturbations. [Chen et al.'s SimCLR study](https://arxiv.org/abs/2002.05709) explicitly investigates their role.

A useful original formalization views images as vertices of a graph, with an edge whenever the training procedure declares a pair equivalent. If exact invariance holds on every edge, it also holds along every path: equality is transitive. Accidentally connecting two semantic classes through an augmentation bridge can force them into the same representation. This explains why reviewing positive-pair construction may matter more than tuning an optimizer.

Suppose semantic classes have probabilities $\pi_c$. Two independent images have the same class with probability $\sum_c\pi_c^2$. With $C$ equally likely classes, this is $1/C$. In an archive dominated by two equally common defect types, it is $1/2$, not a negligible accident. Conditional on an anchor's class probability $\pi_c$, a batch of $B$ independently sampled images supplies on average $(B-1)\pi_c$ same-class negatives. Diversity and sampling structure determine whether the negative-pair approximation is tolerable.

## 3. A contrastive loss is a classification problem [26:06](https://www.youtube.com/watch?v=lNTajqxxOn4&t=1566s)

Sample $B\ge2$ source objects and create two views of each. Let $u_i$ and $v_i$ be their normalized embeddings. For clarity, start with the cross-view similarity matrix

$$S_{ij}=\frac{u_i^\top v_j}{\tau},\qquad \tau>0,$$

where $\tau$ is a temperature. For anchor $u_i$, define a categorical distribution over candidate partners:

$$p_{ij}=\frac{\exp(S_{ij})}{\sum_{k=1}^B\exp(S_{ik})},
\qquad
\ell_i=-\log p_{ii}.$$

The correct candidate is the other view of the same source. This is ordinary softmax cross-entropy with a pairing label generated from data construction. Average $\ell_i$ over anchors. A symmetric version also classifies each $v_i$ against the $u_j$ candidates.

### The gradient reveals the competing forces

Differentiating log-sum-exp gives

$$\frac{\partial\ell_i}{\partial S_{ij}}=p_{ij}-\mathbf1[j=i].$$

The positive score has derivative $p_{ii}-1\le0$; each negative has derivative $p_{ij}\ge0$. Gradient descent therefore increases the positive score and decreases competing scores, subject to the shared encoder's parameterization. In terms of the unscaled inner product, gradients receive an additional factor $1/\tau$.

Ignoring the normalization Jacobian momentarily, the derivative with respect to an anchor is

$$\frac{\partial\ell_i}{\partial u_i}
=\frac1\tau\left(\sum_jp_{ij}v_j-v_i\right).$$

It points from the desired partner toward a probability-weighted average of all candidates. Hard negatives receive greater weight because they currently score highly. For a raw vector $h$ normalized as $u=h/\|h\|$, the actual derivative multiplies by $(I-uu^\top)/\|h\|$, which removes the radial component. Normalization changes optimization, not merely notation.

If all scores in a row are equal, the loss is $\log B$. With one positive similarity $1$, two negative similarities $0$, and $\tau=1$, the positive probability is $e/(e+2)\approx0.5761$ and loss approximately $0.5514$. Decreasing temperature sharpens this distribution when the positive already has the largest score. It also magnifies confidently wrong rankings and changes gradient concentration.

### Classroom cross-view loss versus SimCLR

The published [SimCLR method](https://arxiv.org/abs/2002.05709) uses a nonlinear projection head after the representation encoder and a contrastive objective over $2B$ views. For each anchor, its paired view is positive, the anchor itself is excluded, and the denominator contains $2B-1$ candidates. The cross-view $B\times B$ construction above explains the classification mechanism but is not the complete SimCLR denominator. Keeping the distinction prevents an implementation from quietly changing methods.

The projection head allows the space optimized for the contrastive task to differ from the representation used downstream. It is an empirical architectural choice; a successful contrastive loss still needs evaluation on retrieval or classification.

As a research extension, [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) connects a contrastive classification objective to information shared between paired variables. Such information-theoretic interpretations depend on the positive joint distribution and negative sampling scheme. They should not be applied unchanged after arbitrary hard-negative mining.

### A stable executable loss

The function below implements the symmetric cross-view loss, not full SimCLR. Inputs are two finite, nonzero, equally shaped matrices with at least two rows. It normalizes each row, returns one scalar loss, and raises an error for invalid inputs. It has no persistent effects. Forming the complete score matrix costs $O(B^2m)$ arithmetic and $O(B^2)$ storage.

```python
import numpy as np


def contrastive_loss(left, right, temperature=1.0):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 2 or left.shape != right.shape or left.shape[0] < 2:
        raise ValueError("Use equal matrices containing at least two rows")
    if not left.shape[1] or not np.isfinite(left).all():
        raise ValueError("Left embeddings must be finite and nonempty")
    if not np.isfinite(right).all():
        raise ValueError("Right embeddings must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    ln, rn = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
    if np.any(ln == 0) or np.any(rn == 0):
        raise ValueError("Zero vectors cannot be cosine-normalized")
    scores = (left / ln[:, None]) @ (right / rn[:, None]).T / temperature

    def row_loss(matrix):
        shifted = matrix - matrix.max(axis=1, keepdims=True)
        return np.mean(
            np.log(np.exp(shifted).sum(axis=1)) - np.diag(shifted)
        )

    return 0.5 * (row_loss(scores) + row_loss(scores.T))


orthogonal = np.eye(3)
assert np.isclose(contrastive_loss(orthogonal, orthogonal),
                  np.log(np.e + 2) - 1)
collapsed = np.ones((3, 2))
assert np.isclose(contrastive_loss(collapsed, collapsed), np.log(3))
assert contrastive_loss(orthogonal, orthogonal, 0.1) < 0.001
assert contrastive_loss(orthogonal, orthogonal[::-1]) > 1
```

Subtracting a row maximum preserves softmax probabilities while preventing exponential overflow for ordinary finite scores. The assertions check correct pairing, collapse, temperature, and a deliberately incorrect pairing. They do not prove that an encoder trained on real images avoids collapse or learns the desired semantics.

## 4. Cross-modal pairs and difficult negatives [39:50](https://www.youtube.com/watch?v=lNTajqxxOn4&t=2390s)

The two sides of a positive pair need not share a modality. Let $u_i$ encode photograph $i$ and $v_i$ encode its text description. Both encoders output vectors in the same dimension, but can have different architectures and parameters. The same similarity matrix now trains image-to-text and text-to-image matching.

[Radford et al.'s CLIP paper](https://arxiv.org/abs/2103.00020) is the primary example of this approach. At inference, textual class descriptions can serve as candidate prototypes. For an image embedding $u$, select the class with largest $u^\top v_c$, where $v_c$ is the normalized embedding of its description. “Zero-shot” refers to the downstream evaluation protocol; it does not mean the pretraining corpus lacked related images or language.

For an inspection image, compare descriptions “structural foundation crack,” “surface paint crack,” and “undamaged wall.” A correct coarse category does not establish crack width, severity, or repair cost. Those require separate evidence. Likewise, image-text correspondence can be noisy: captions may describe context while omitting the defect.

Hard-negative mining chooses alternatives that resemble positives but should be distinguished. A report about repairing plaster is more demanding than a report about swimming pools. The gradient formula explains why: an easy negative already has tiny $p_{ij}$ and contributes little. But mining by high similarity also selects more false negatives, such as two valid reports about the same repair. Better difficulty is useful only when the pair labels remain meaningful.

An operational check should therefore record negative source, annotation rule, and false-negative rate. Sample within the same building category or document type to prevent trivial shortcuts, then audit whether the resulting pairs really require separation. For text, title-body pairs create supervision without explicit relevance labels, but repeated titles and boilerplate can teach shortcuts rather than answer relevance.

## 5. Search turns geometry into a ranked evidence set [49:25](https://www.youtube.com/watch?v=lNTajqxxOn4&t=2965s)

Let the archive contain documents $d_1,\ldots,d_N$. Precompute normalized document embeddings as rows of $D\in\mathbb R^{N\times m}$. Encode query $q$ as normalized vector $u$. Exact retrieval ranks scores $Du$ and returns the largest $k$. Document encoding is an offline cost; scoring one query costs $O(Nm)$ arithmetic, while storing the embeddings costs $Nm$ numbers.

One million documents at dimension $768$ occupy $3.072$ GB in four-byte floating point, before metadata and index overhead. This calculation is decimal gigabytes, not gibibytes. Compression and approximate indexes alter the cost and retrieval quality.

Consider query $u=(1,0)$. Three document embeddings are $(0.8,0.6)$, $(0.6,0.8)$, and $(-1,0)$. The scores are $0.8,0.6,-1$, and squared distances are $0.4,0.8,4$. Both criteria select the first document. The following exact reference implementation uses finite normalized inputs and stable tie order. It returns row indices and scores, without modifying the archive or making a network call.

```python
def retrieve(query, documents, k):
    query, documents = np.asarray(query, float), np.asarray(documents, float)
    if documents.ndim != 2 or query.shape != (documents.shape[1],):
        raise ValueError("Query dimension must match document columns")
    if not 1 <= k <= len(documents) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer in the archive range")
    if not np.isfinite(query).all() or not np.isfinite(documents).all():
        raise ValueError("Embeddings must be finite")
    if not np.isclose(np.linalg.norm(query), 1):
        raise ValueError("Normalize the query first")
    if not np.allclose(np.linalg.norm(documents, axis=1), 1):
        raise ValueError("Normalize document rows first")
    scores = documents @ query
    indices = np.argsort(-scores, kind="stable")[:k]
    return indices, scores[indices]


docs = np.array([[0.8, 0.6], [0.6, 0.8], [-1.0, 0.0]])
indices, scores = retrieve([1, 0], docs, 2)
assert np.array_equal(indices, [0, 1])
assert np.allclose(scores, [0.8, 0.6])
assert np.allclose(np.sum((docs - [1, 0]) ** 2, axis=1), [0.4, 0.8, 4])
```

At larger scale, approximate nearest-neighbor search trades exactness for speed. [Malkov and Yashunin's HNSW paper](https://arxiv.org/abs/1603.09320) describes a graph-based method. The relevant check is recall against exact neighbors on representative queries, together with latency and memory. Approximate-index recall is distinct from semantic relevance recall: an exact algorithm can faithfully return the wrong documents if the embedding model is unsuitable.

Changing an encoder or its normalization changes the coordinate system. Query and document vectors must use compatible versions; rebuilding only the query encoder can invalidate rankings. Chunking documents also changes the retrieval unit. A chunk must retain enough surrounding context to make its content interpretable.

## 6. Retrieval-augmented generation is an evidence pipeline [53:04](https://www.youtube.com/watch?v=lNTajqxxOn4&t=3184s)

Retrieval-augmented generation, or RAG, supplies selected external material to a generator. A simple deployed pipeline retrieves documents, places them with the query in the model context, and generates an answer without updating model weights. This allows the inspection assistant to consult a newly uploaded report.

The original [Lewis et al. RAG paper](https://arxiv.org/abs/2005.11401) is more specific than generic prompt concatenation. One formulation treats the retrieved document $z$ as latent and models

$$p(y\mid q)\approx\sum_{z\in\mathcal R(q)}
p_\eta(z\mid q)\,p_\theta(y\mid q,z),$$

where $y$ is an answer sequence, $\eta$ parameterizes retrieval, $\theta$ parameterizes generation, and $\mathcal R(q)$ is a truncated candidate set. Normalization and truncation must be specified. A system that concatenates several chunks into one prompt is a related engineering architecture, not necessarily this latent-document model.

### Retrieval success and answer success are separate

Let $E$ be the event that sufficient evidence reaches the generator and $C$ that its answer is correct. The law of total probability gives

$$P(C)=P(C\mid E)P(E)+P(C\mid E^c)P(E^c).$$

If evidence recall is $0.8$, correctness with evidence is $0.9$, and correctness without it is $0.1$, total correctness is $0.74$. Improving the generator alone cannot repair every retrieval omission. Conversely, retrieving the right report does not ensure that its date, negation, or units are respected.

For the inspection question, preserve document IDs, passage boundaries, dates, and answer-to-passage citations. Test whether each claimed fact is supported, not merely whether a citation exists. Compare dense retrieval with lexical matching for precise identifiers such as building IDs and report numbers; a hybrid can be evaluated against both. No proprietary vendor's hidden retrieval design is assumed here.

Access filtering must occur before unauthorized text is supplied to the generator. A local index does not make a remote model call private: the selected text still travels to that endpoint. Removing a document from active retrieval also does not automatically remove prior prompts, caches, logs, or information already learned by a model. These are data-flow consequences of the architecture, rather than properties conferred by the acronym RAG.

## Limitations and an evaluation plan

The end-to-end system has at least four distinct failure boundaries: pair construction, embedding geometry, search approximation, and answer generation. Evaluate them separately with held-out buildings and realistic questions. Include paraphrases, exact IDs, missing evidence, contradictory reports, and questions whose correct answer is “the archive does not establish this.”

A useful extension project is to maintain an exact-search reference and compare each optimization against it. This isolates index errors from model errors, much as a numerical solver is checked against a small analytically solvable problem.

## Exercises

1. Show why unit-normalized cosine and squared Euclidean distance rank candidates identically. Give a counterexample without normalization.
2. In four equally likely semantic classes, how many same-class negatives should an anchor expect among 31 independent negative candidates? Explain why this does not prove contrastive learning fails.
3. Derive the contrastive score gradient. For a positive probability of $0.6$ and two negative probabilities of $0.3$ and $0.1$, give all three derivatives.
4. A $B=4$ cross-view loss and full $2B$-view SimCLR both collapse to identical embeddings. What is each anchor loss, and why are they different?
5. Evidence recall rises from $0.8$ to $0.95$ while the two conditional correctness rates remain $0.9$ and $0.1$. Calculate the improvement. Design a check that distinguishes evidence retrieval from citation faithfulness.

<details>
<summary>Fully worked solutions</summary>

1. For unit vectors, squared distance is $2-2u^\top v$, a strictly decreasing affine function of the cosine score, so ties and ordering coincide. Without normalization, query $(1,0)$ and candidates $(2,2)$ and $(1,0)$ have inner-product ranking $2>1$ but cosine ranking $1/\sqrt2<1$. The changing norm breaks the identity used for the ranking argument.
2. Each negative shares the anchor class with probability $1/4$. Linearity of expectation gives $31/4=7.75$; independence is not needed for that expectation once the marginal probabilities are specified. The training objective can still learn useful distinctions, but its negative labels are imperfect. Success depends on competing signals, sampling, architecture, and the downstream definition of relevance.
3. Write $\ell_i=-S_{ii}+\log\sum_j e^{S_{ij}}$. Differentiation gives $p_{ij}-\mathbf1[j=i]$, hence derivatives $-0.4,0.3,0.1$. They sum to zero because adding a constant to all logits leaves the probabilities unchanged. For similarities before temperature scaling, divide each derivative by $\tau$.
4. Cross-view classification has four equal candidates, giving loss $\log4$. SimCLR excludes the anchor itself from eight total views and retains seven equal candidates, giving $\log7$. The positive is included in both denominators. The different losses reflect different candidate sets, not different evidence that collapse is desirable.
5. New correctness is $0.95(0.9)+0.05(0.1)=0.86$, an improvement of $0.12$. First label whether retrieved passages contain sufficient evidence without looking at the generated answer. Separately inspect each answer claim and its cited passage for entailment, including dates and negation. This permits a correct-retrieval/wrong-answer case to be distinguished from a retrieval omission.
</details>

## Primary sources

- Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).
- van den Oord, Li, and Vinyals, [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- Malkov and Yashunin, [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320).
- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

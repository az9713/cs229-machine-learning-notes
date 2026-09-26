# Stanford CS229 — Machine Learning, Spring 2026

An independent, transcript-grounded study edition of **Stanford CS229**, based on the **Stanford Online YouTube recordings**. The course and recordings are Stanford's work; the expanded exposition described below is our contribution.

**[Read the live textbook](https://az9713.github.io/cs229-machine-learning-notes/)** · **[Study guide](https://az9713.github.io/cs229-machine-learning-notes/study-guide.html)** · **[Recording and timestamp map](https://az9713.github.io/cs229-machine-learning-notes/source-map.html)**

[Official Stanford course page](https://cs229.stanford.edu/index.html-spr26) · [Source YouTube playlist](https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw) · [First source video: Introduction](https://www.youtube.com/watch?v=DATnpGoGhM8)

The edition contains **17 chapters**, approximately **64,169 words**, **85 exercises with worked solutions**, and **32 executable Python examples**. Each major lecture section links to the relevant moment in its recording. The links below open rendered web pages hosted on GitHub Pages.

## Read the lectures online

| Lecture | Live chapter | Stanford source video |
|---|---|---|
| 1 | [What is being learned?](https://az9713.github.io/cs229-machine-learning-notes/lecture-01.html) | [YouTube recording](https://www.youtube.com/watch?v=DATnpGoGhM8) |
| 2 | [Linear regression as prediction, geometry, and computation](https://az9713.github.io/cs229-machine-learning-notes/lecture-02.html) | [YouTube recording](https://www.youtube.com/watch?v=cmNIMjPYdgM) |
| 3 | [From noise models to logistic regression and Newton steps](https://az9713.github.io/cs229-machine-learning-notes/lecture-03.html) | [YouTube recording](https://www.youtube.com/watch?v=uJF_gL3jhxI) |
| 4 | [Exponential families, generalized linear models, and softmax](https://az9713.github.io/cs229-machine-learning-notes/lecture-04.html) | [YouTube recording](https://www.youtube.com/watch?v=8gVi4Rk21Eg) |
| 5 | [Generative classification, Gaussian geometry, and naive Bayes](https://az9713.github.io/cs229-machine-learning-notes/lecture-05.html) | [YouTube recording](https://www.youtube.com/watch?v=zRdE8A4UZes) |
| 6 | [Bias, variance, regularization, and model selection](https://az9713.github.io/cs229-machine-learning-notes/lecture-06.html) | [YouTube recording](https://www.youtube.com/watch?v=llnEgyyuYkQ) |
| 7 | [Neural networks as compositions of learned representations](https://az9713.github.io/cs229-machine-learning-notes/lecture-07.html) | [YouTube recording](https://www.youtube.com/watch?v=fRM41w9jzQo) |
| 8 | [Backpropagation, computational graphs, and higher-order derivatives](https://az9713.github.io/cs229-machine-learning-notes/lecture-08.html) | [YouTube recording](https://www.youtube.com/watch?v=ne2ngVAoMG8) |
| 9 | [Clustering, Gaussian mixtures, and the geometry of lower bounds](https://az9713.github.io/cs229-machine-learning-notes/lecture-09.html) | [YouTube recording](https://www.youtube.com/watch?v=bSmIGBCoffA) |
| 10 | [Expectation–maximization and principal component analysis](https://az9713.github.io/cs229-machine-learning-notes/lecture-10.html) | [YouTube recording](https://www.youtube.com/watch?v=sUS-eTa0l6s) |
| 11 | [Diffusion models: forward noise, reverse dynamics, and a trajectory ELBO](https://az9713.github.io/cs229-machine-learning-notes/lecture-11.html) | [YouTube recording](https://www.youtube.com/watch?v=dqUMCzWjZSI) |
| 12 | [Learning to denoise, then learning to adapt](https://az9713.github.io/cs229-machine-learning-notes/lecture-12.html) | [YouTube recording](https://www.youtube.com/watch?v=_kREM2UAiJ8) |
| 13 | [Contrastive representations, semantic search, and grounded generation](https://az9713.github.io/cs229-machine-learning-notes/lecture-13.html) | [YouTube recording](https://www.youtube.com/watch?v=lNTajqxxOn4) |
| 14 | [Autoregressive language models and the mechanics of attention](https://az9713.github.io/cs229-machine-learning-notes/lecture-14.html) | [YouTube recording](https://www.youtube.com/watch?v=pwQ0l4hFCVI) |
| 16 | [Efficient transformers and the difference between prompting and training](https://az9713.github.io/cs229-machine-learning-notes/lecture-16.html) | [YouTube recording](https://www.youtube.com/watch?v=hHC-SF3utxg) |
| 18 | [Markov decisions, Bellman equations, and policy gradients](https://az9713.github.io/cs229-machine-learning-notes/lecture-18.html) | [YouTube recording](https://www.youtube.com/watch?v=xveNBYVTrqw) |
| 20 | [Credit assignment, PPO, and reinforcement learning for language models](https://az9713.github.io/cs229-machine-learning-notes/lecture-20.html) | [YouTube recording](https://www.youtube.com/watch?v=J7CossjMvEg) |

The supplied upload sequence omits lecture numbers **15, 17, and 19**. We retain the original numbering and cover all 17 available recordings; we do not reconstruct missing lectures. Some upload titles differ from their spoken content, so chapter titles describe the content while the source map preserves the original upload titles.

## Our contribution beyond the transcripts

The transcripts determine the topic sequence and provide recording anchors. We add:

- **Explicit mathematical models and derivations:** definitions, domains, assumptions, intermediate steps, interpretations, and limiting cases.
- **Original worked examples:** a recurring property and inspection-assistant case, exact numerical calculations, and counterexamples that expose a method's boundaries.
- **Executable teaching code:** small NumPy implementations with assertions, finite-difference checks, or exact enumeration, connected to the mathematics they implement.
- **Researched extensions:** primary papers integrated beside the claims they support, including optimization, diffusion, LoRA, contrastive learning, efficient attention, and policy optimization.
- **Exercises and full solutions:** derivation, computation, counterexamples, and design judgment, with reasoning available in expandable solution panels.
- **Clarifications and corrections:** distinctions such as likelihood versus evidence bounds, constant baselines versus reward variance, and PPO clipping versus a bound on policy divergence.
- **A usable reading edition:** offline mathematical rendering, chapter search, previous/next navigation, code-copy controls, phone layouts, and print styles that expand solutions.

These additions are authored teaching material. A timestamp identifies the related lecture discussion; it does **not** imply that Stanford presented every derivation, example, correction, or research extension in that section. Research findings remain attributed to their primary authors.

## Attribution and source boundary

This is an independent educational project, not an official Stanford publication or an endorsement by Stanford. Course names, recordings, and cited research belong to their respective creators. Recordings are linked, not redistributed.

The complete available English caption text was read sequentially for each chapter. YouTube marked these tracks as non-automatic; that metadata does not guarantee perfect transcription. Slides and whiteboard images were **not comprehensively reviewed**. The notes reconstruct mathematics rather than claiming visual transcription or a verbatim account.

Raw captions, downloaded research documents, private working files, and acquisition logs are excluded from this repository. [Source metadata](source-manifest.json) and the [section ledger](analysis-ir.json) retain recording identities and timestamp provenance. The [reference audit](sources/reference-audit.json) reports retrieval outcomes, including any automated-access failures.

## Use locally

Download the repository ZIP through GitHub's **Code → Download ZIP**, extract it, and open `index.html`. The HTML, CSS, JavaScript, and mathematics work offline; recordings and external papers require an internet connection. Each chapter also has an editable Markdown version. Use **Print / save PDF** for a printable chapter.

Python examples use NumPy and the Python standard library. Run a chapter's code blocks in order in a fresh interpreter or notebook; later blocks may reuse earlier definitions. These are small educational implementations with stated simplifications, not production training systems.

The [validation report](validation/study-edition-checks.json) records source hashes, local links, timestamp bounds, mathematical rendering, responsive layouts, and solution controls. The [code report](validation/embedded-code-checks.json) records execution of all 32 Python blocks. These checks support the examples and presentation; they are not a claim of formal verification of every statement.

GitHub Pages serves the files from the `main` branch root. `.nojekyll` preserves the static bundle. MathJax is bundled under its [license](assets/mathjax-LICENSE).

Prepared 25 September 2026. Historical research comparisons are reported in their original context, not as current product rankings.

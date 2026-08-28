# Sparse compression of MLP weight actions: relevant literature and bilin18 map

Date: 2026-08-28

## The important object distinction

The historical “weight SAE” was not trained on the entries of a static weight matrix.
For native gate vectors (g), it approximated the action

\[
a(g)=Wg
\qquad\text{by}\qquad
\hat a(g)=D\,\operatorname{TopK}(Eg)+c.
\]

Thus it is an **input-conditioned sparse operator approximation**. It uses real gate
vectors to choose a sparse code, but reconstructs the output of a fixed weight matrix.
This is closer to paired/coupled dictionary learning and learned sparse inference than
to an activation SAE. It is also not a global factorization of (W): because TopK is
nonlinear, equality on the language-data manifold does not imply equality for arbitrary
inputs.

The native bilin18 interface is

\[
g=\operatorname{Left}(x)\odot\operatorname{Right}(x),\qquad
\operatorname{MLP}(x)=Wg+\operatorname{Down\_bias}.
\]

`Down` itself has no bias. `Down_bias` is added by the containing MLP after a `Down`
hook. A learned (c) is therefore a compressor intercept, while native `Down_bias`
remains a separate exact constant.

## Most relevant families, in priority order

### 1. Alternating synthesis dictionary learning plus an executable encoder

Classical sparse coding first solves

\[
z^*(g)=\arg\min_{\|z\|_0\le k}\|Wg-Dz-c\|_2^2
\]

with OMP, IHT, or LASSO, and alternates this with a dictionary update such as MOD or
K-SVD. This cleanly separates two questions that the old one-pass encoder confounded:

1. can a (k)-sparse dictionary represent the observed weight actions?;
2. can a cheap encoder infer those codes from (g)?

Mairal et al.'s online dictionary learning supplies a scalable convergent optimization
framework. Gregor and LeCun's LISTA supplies the second stage: train a fixed-depth
network to imitate optimized sparse codes. This is the highest-priority correction to
our historical top-k fit.

Sources:

- Julien Mairal et al., [Online Learning for Matrix Factorization and Sparse Coding](https://www.jmlr.org/papers/v11/mairal10a.html), JMLR 2010.
- Karol Gregor and Yann LeCun, [Learning Fast Approximations of Sparse Coding](https://icml.cc/2010/papers/449.pdf), ICML 2010.

### 2. Coupled or task-driven dictionary learning

Coupled dictionary learning uses one sparse code to link paired spaces. Here the pair
is (g\leftrightarrow Wg), and the more ambitious pair is an MLP0 code linked to the
responses of MLP1 and attention readers. Task-driven dictionary learning adds a
downstream loss to the dictionary objective. For bilin18 that loss should be held-out
CE/KL or finite-response error, with a reconstruction anchor; otherwise CE-only
finetuning can destroy local faithfulness.

The executable joint version must actually use the sparse edge (C):

\[
a_0=\operatorname{TopK}(p_0E_0+c_0),\quad w_0=a_0D_0,
\]

\[
a_1=\operatorname{TopK}(p_1E_1+c_1+a_0C),\quad w_1=a_1D_1.
\]

A penalty on (C) without executing (C) is not a compressed causal program.

Sources:

- Jianchao Yang et al., [Coupled Dictionary Training for Image Super-Resolution](https://doi.org/10.1109/TIP.2012.2192127), IEEE TIP 2012.
- Julien Mairal, Francis Bach, and Jean Ponce, [Task-Driven Dictionary Learning](https://doi.org/10.1109/TPAMI.2011.156), IEEE TPAMI 2012.

### 3. Analysis/transform learning

Synthesis coding asks whether outputs are sparse combinations of decoder atoms.
Analysis or transform learning instead learns an operator whose direct responses are
sparse. This is closer to our desired cheap encoder (Eg), and formulations explicitly
regularize conditioning to avoid scale-degenerate transforms. It is a good matched
alternative to a free encoder-decoder SAE, but it still needs an output decoder or a
coupled objective to preserve (Wg).

Source:

- Saiprasad Ravishankar and Yoram Bresler, [Learning Sparsifying Transforms](https://doi.org/10.1109/TSP.2012.2226449), IEEE TSP 2013.

### 4. Structured, hierarchical, and multi-layer sparse coding

Overlapping group penalties and tree-structured sparse coding can represent multiple
memberships: e.g. a token can activate both “capitalized” and “city” parents. A
multi-layer sparse model can represent shared parents and specialized children.
However, a hierarchy is earned only if it reduces total producer-plus-reader bytes or
operations at matched causal fidelity. Co-activation alone does not identify a causal
DAG.

Sources:

- Rodolphe Jenatton et al., [Proximal Methods for Hierarchical Sparse Coding](https://arxiv.org/abs/1009.2139), 2010.
- Aviad Aberdam, Jeremias Sulam, and Michael Elad, [Multi-Layer Sparse Coding: The Holistic Way](https://arxiv.org/abs/1804.09788), 2018.

### 5. Double sparsity for compressing the dictionary itself

An overcomplete decoder (D) can cost more parameters than it saves. Double sparsity
constrains (D=\Phi A), where (Phi) is a fixed or cheap base and (A) is sparse.
For bilin18, plausible bases are the rank-64 MLP0 output subspace, a gauge-fixed tensor
basis, or shared reader directions. This directly attacks executable cost rather than
only code sparsity.

Source:

- Ron Rubinstein, Michael Zibulevsky, and Michael Elad, [Double Sparsity: Learning Sparse Dictionaries for Sparse Signal Approximation](https://doi.org/10.1109/TSP.2009.2036477), IEEE TSP 2010.

## What applies if we truly use weights only

If no input distribution is used, then sparse coding of (Wg) is unavailable. The
relevant objects become direct structured factorizations of (W): truncated SVD,
sparse-plus-low-rank, Kronecker sums, tensor train, or a factorization of the folded
third-order bilinear tensor. These can certify storage and operation count globally,
but cannot by themselves discover which distinctions downstream computation uses.

Tensor Train and Kronecker decompositions are useful matched compression grammars, not
semantic guarantees:

- Alexander Novikov et al., [Tensorizing Neural Networks](https://papers.nips.cc/paper/2015/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html), NeurIPS 2015.
- Shuchang Zhou and Jia-Nan Wu, [Compression of Fully-Connected Layer in Neural Network by Kronecker Product](https://arxiv.org/abs/1507.05775), 2015.

For MLP0 specifically, the direct rank-64 output bond is already the strongest
weight-only starting point. Sparse coding should therefore be attempted *inside that
64-dimensional bond*, or with a double-sparse decoder based on it, rather than paying
for a new dense (1152\times P) decoder.

## Noise is a metric choice, not a generic cure

For linear approximation error (A=W-\hat W) and perturbation
(epsilon\sim(0,\Sigma)),

\[
\mathbb E\|A(g+\epsilon)\|_2^2
=\mathbb E\|Ag\|_2^2+\operatorname{tr}(A\Sigma A^\top)
\]

when the cross term vanishes. Isotropic noise therefore adds a Frobenius-like global
operator penalty. It can improve local support stability, but it deliberately changes
the objective away from the empirical language manifold. Use small covariance-matched
noise and retain it only if it improves noisy/OOD response without materially harming
clean CE and finite-response fidelity.

## Limits of compressed-sensing guarantees here

Exact dictionary-recovery theorems generally assume genuinely sparse independent
coefficients, sufficiently incoherent atoms, enough samples, and often a generative
model (y=Dz). Bilin18 gates are correlated bilinear products and we have not shown
that they were generated by a sparse dictionary. Therefore those theorems justify
optimizer controls and measurable recovery conditions, not semantic claims about the
learned atoms.

Representative recovery result:

- Alekh Agarwal et al., [Learning Sparsely Used Overcomplete Dictionaries](https://proceedings.mlr.press/v35/agarwal14a.html), COLT 2014.

## Cheapest decisive experiment

At fixed dictionary size and sparsity, compare on untouched documents:

1. the historical one-pass positive TopK encoder;
2. signed, scale-normalized TopK;
3. oracle OMP/IHT codes with a learned decoder;
4. a LISTA-style executable encoder trained to imitate the oracle;
5. a rank-64-bond double-sparse version at matched serialized bytes;
6. shuffled-pair and equal-byte dense low-rank controls.

Report convergence curves, three or more seeds, reconstruction (R^2), CE/KL, finite
response prediction, selective removal collateral, OOD transport, bytes, and executed
operations. A large oracle-versus-encoder gap licenses better inference; no oracle gap
rejects “optimizer failure” and sends us to joint readers or hierarchy.

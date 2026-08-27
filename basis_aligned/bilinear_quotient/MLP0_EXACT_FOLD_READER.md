# MLP0 exact folding and downstream-use quotient

## Result

The earlier MLP0 work contained a weight-only length-one token map and an approximate
downstream-defined k-means experiment. It did **not** compute the exact coarsest
partition induced by literal downstream readers. This module supplies the missing
mathematical primitive without claiming that a checkpoint audit has already run.

## Input folding

For any already-assembled MLP0 input row $z$, the native map is evaluated exactly as

$$
m_0(z)=D_0[(L_0z)\odot(R_0z)]+b_0.
$$

For the context-free token map, $z_t$ must itself be constructed from the token
embedding, exact block-0 length-one attention, residual assembly, and RMSNorm. Folding
the matrices does not make RMSNorm or attention linear. For real contextual behavior,
the actual contextual $z_{t,c}$ must be supplied.

## Hard computational equivalence

Let $W_i$ be every declared immediate block-1 reader of the MLP0 write: both MLP1
factors and attention-1 Q/K/Q2/K2/V. Two writes are indistinguishable to these linear
preactivations exactly when

$$
v\sim w
\quad\Longleftrightarrow\quad
W_i(v-w)=0\ \text{for every }i.
$$

Equivalently, $v-w$ lies in the joint kernel of the vertically stacked reader matrix.
The observable quotient is represented canonically by the projector onto that
stacked matrix's rowspace. No random projection and no k-means step is needed to
define this equivalence.

This is the linear-algebraic analogue of observational equivalence: distinctions in
the common kernel cannot be transmitted through the declared reader channel. It is
not yet full-network equivalence because the residual stream also carries the write
forward and later nonlinear readers may distinguish it.

## RMSNorm-aware local equivalence

Immediate readers act after residual addition and RMSNorm. For

$$
n(x)=\frac{x}{\sqrt{d^{-1}x^Tx+\epsilon}},
$$

the exact Jacobian is

$$
J_x=\sigma^{-1}I-\frac{xx^T}{d\sigma^3},
\qquad \sigma^2=d^{-1}x^Tx+\epsilon.
$$

At a particular residual state, the first-order observable maps are therefore
$W_iJ_x$. Stacking them across a declared set of states yields the exact local
first-order quotient on those states. The implementation is checked against PyTorch
autograd. This applies the normalization geometry described by
[Zhang and Sennrich (2019)](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
rather than treating the residual coordinates as Euclidean by default.

## Literal output-weight composition

Let $d_k=D_0[:,k]$ be hidden unit $k$'s residual write. In native reader coordinates,
the exact pairwise sensitivity Gram is

$$
G=\sum_i W_i^TW_i,
\qquad
G_{\mathrm{unit}}=D_0^TGD_0.
$$

For equivalence independent of invertible coordinate changes within the readers,
replace $G$ by the joint-rowspace projector $P$. Then $D_0^TPD_0$ compares units by
which declared downstream directions can observe their writes. This is the literal
weight composition requested by the user.

## Why this is not automatically clustering

Exact equivalence supplies quotient classes only when differences land in a true
joint kernel. If the stacked readers have full column rank, every distinct write is
linearly distinguishable and all hard classes are singletons. A useful *approximate*
clustering may still exist, but then one must declare:

- native sensitivity Gram versus rowspace-projector metric;
- state-free versus RMSNorm-tangent metric;
- token writes versus hidden-unit output columns;
- clustering algorithm, number of clusters, and tolerance;
- causal interchange test used to validate the resulting groups.

The existing downstream clustering used selected block-1 readers, fixed random
projections, and k-means. It was a useful approximation, not the exact quotient.

## What remains to execute

The CPU algebra and eight synthetic/autograd tests are complete. A checkpoint audit
must still measure:

1. the exact rank and spectrum of the block-1 reader stack;
2. whether any token writes or $D_0$ columns are exactly equivalent;
3. the no-random-projection reader-metric clustering frontier;
4. whether proposed approximate groups survive disjoint-row causal interchange.

Until those measurements exist, “MLP0 performs soft overlapping token clustering”
is a reasonable coarse description, while “the exact downstream computational
clusters are known” is false.

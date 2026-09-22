# Full quadratic target: a product-count bound beyond dense Tucker

2026-09-22 04:25 UTC. Derived from existing mode spectra; no new model fit or spectrum computation.

The full joint third-order tensor and the recent 16-output quartic experiments are different targets. This note concerns the **full vocabulary-readout quadratic tensor of the last MLP**, with the input geometries used in the existing full-tensor audit. It does not bound the quartic candidate restricted to 16 selected output coordinates.

Let the quadratic program use $K$ scalar products of learned linear forms, with arbitrary linear sharing and a linear output readout:

$$
f(x)=\sum_{k=1}^K c_k(a_k^\top x)(b_k^\top x).
$$

Its symmetric coefficient tensor is

$$
T=\sum_{k=1}^K c_k\otimes\frac{a_kb_k^\top+b_ka_k^\top}{2}.
$$

Flattening the two input indices into one matrix column index, each term has output-mode matrix rank at most one. Thus the output-mode rank is at most $K$. Both input modes lie in the span of the $2K$ vectors $a_k,b_k$, so each input-mode rank is at most $2K$. If the measured mode spectra require output rank $r_o$ and input rank $r_i$ at error tolerance $\epsilon$, then

$$
K\geq\max\left(r_o,\left\lceil r_i/2\right\rceil\right).
$$

Linear recombinations or caching shared products do not evade this bound: count each distinct computed product once. A quadratic feature assembled from many products counts all of its constituent products. This argument directly applies to the stated quadratic program class. A later [degree-two propagation derivation](FULL_QUADRATIC_DAG_BOUND_EXTENSION_V1.md) extends the same product bound to division-free polynomial DAGs, including higher-degree intermediate terms and cancellation. Variable division and normalization remain outside this bound; functional approximation on data is also a different metric.

| Input coefficient geometry | Necessary products at 20% error | At 10% | At 5% |
| --- | ---: | ---: | ---: |
| folded_euclidean | 962 | 1088 | 1133 |
| residual_euclidean | 967 | 1090 | 1134 |
| activation_covariance | 403 | 818 | 1030 |

The original native last-MLP factorization uses 4,608 products. In folded Euclidean coefficient geometry, 10% error needs at least 1,088 products even before charging any dense linear readers or readout coefficients. The bound leaves room for improvement over the original factorization; it does not prove that a program with 1,088 products exists. Likewise, a sparse Tucker core can retain high ranks and still be economical, but it cannot produce broad output span from too few scalar products.

The covariance geometry still needs at least 818 products at 10% error. This covariance-weighted coefficient norm is not empirical text error or the fourth-moment quadratic function loss. Input means and their lower-degree terms are outside the audited homogeneous tensor. No text fidelity, semantic identification, extraction, or intervention guarantee follows from these counts.

This refines the practical direction: allow a broad output span with sparse or shared computation when targeting the full tensor. Do not mistake a successful 16-output program for full-output recovery, and do not revisit output widths already excluded by the corresponding metric. The current quartic residual learner remains relevant to its narrower selected path and is not rejected by this bound.

[Existing spectra and interpretation](FULL_TENSOR_MODE_INTERPRETATION_V1.md) · [Derived counts](FULL_QUADRATIC_PRODUCT_BOUNDS_V1.json) · [Original geometry results](FULL_TENSOR_GEOMETRY_BOUNDS_V1.json).

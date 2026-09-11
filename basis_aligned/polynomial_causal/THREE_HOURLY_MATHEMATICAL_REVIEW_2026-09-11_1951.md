# Mathematical review — 11 September 19:51 UTC

The goal remains a simpler executable decomposition with OOD prediction, extraction, selective removal and composition/reuse. Follow the bilinear handoff and Logan's weights-first clarification, not the stale goal's `better_math_ideas` wording. The immediate mathematical question is how an arithmetic rewrite can expose a cheaper organization **without silently changing the intervention interface**.

The latest behavioral distinction matters: complete-product donor swaps show useful contextual alignment, but a common-reader swap with closely matched private partners does not pass its finite positive-cost screen. This closes that independent-reader screen on the existing panels. It does not refute the complete product or establish a semantic circuit. Repeatedly changing its validation controls is now lower-information than advancing the arithmetic representation.

## The actual object

The final folded quadratic tensor is

$$
T_{vij}=\sum_{k=1}^{4608}(UD)_{vk}\operatorname{sym}(L_kR_k^\top)_{ij},
\qquad T\in\mathbb R^{50304\times1152\times1152}.
$$

Both input modes act on the same RMS-normalized vector. Antisymmetric coefficients are invisible. A factor of $U^\top U$ gives an exact 1152-dimensional output metric without materializing the vocabulary tensor. Bias, residual background, native RMS epsilon, and the $30\tanh$ logit cap remain outside this homogeneous quadratic. Earlier attention uses normalized QK products and rounded rotary operations; the entire network is not a polynomial in its original token vectors.

The frozen spectral surrogate has 64 rank-16 quadratic output groups, 12 shared readers and 17 globally deduplicated shared/shared products. Its coefficient fit is still an unconverged approximation to the native tensor, with about 11.87% coefficient capture. The present rewrite changes neither that fit nor the tensor objective. The existing objective uses the full folded coefficient norm plus 0.01 times summed group energy; no corpus is used to fit the rewrite.

For group $g$, the existing graph reads common values $z_g$ and private values $y_g$, then computes

$$
q_g=z_g^\top H_gz_g+z_g^\top C_gy_g+y_g^\top\Lambda_gy_g,
\qquad F=\sum_gc_gq_g.
$$

$\Lambda_g$ is diagonal and signed; $C_g$ is the code's `cross` array, including the factor of two. With $k$ parents, there are $l=16-k$ private coordinates. Input-basis changes, output/core rescaling and permutations create gauges. All native background and output adapters remain charged. The shared variable is an explicit graph port: an intervention on $z_i$ must propagate to every consumer.

## Literature and exact mappings

**Arithmetic circuits and tensor rank.** The linear/bilinear gate argument in Shpilka–Yehudayoff's [arithmetic-circuit survey, Theorem 3.23 proof](https://www.cse.iitk.ac.in/users/nitin/courses/CS748-2015-16-II/pdfs/survey-SY10.pdf) relates product gates to tensor decompositions. For our restricted degree-respecting grammar, the mapping is direct: collapse every linear input subgraph to a linear form, and every output linear subgraph to coefficients on the product nodes. A program with $M$ linear-by-linear products then has

$$
F(x)=\sum_{m=1}^{M}c_m(a_m^\top x)(b_m^\top x).
$$

Thus an unrestricted real-product factorization already represents this class at fixed product count. A DAG additionally describes cheap **linear arithmetic and reuse** inside the input/output maps; flattening those maps into dense arrays can lose those savings. This derivation assumes no higher-degree cancellation, divisions by variable quantities, or nonlinear normalization inside the object. It is not a general minimum-circuit algorithm or uniqueness theorem. The tied-input symmetric case is not identical to ordinary CP rank on two independent inputs.

**Block boundaries are not an optimality guarantee.** [Shitov's direct-sum counterexamples](https://archive.intlpress.com/site/pub/files/_fulltext/journals/acta/2019/0222/0002/ACTA-2019-0222-0002-a003.pdf) show that independent bilinear systems do not always have additive tensor rank. Our shared-input, partially symmetric, approximate LL1 groups are not the paper's constructed tensors, so it provides no numerical native saving here. It does rule out using a blanket “optimize every block separately and sum the minima” argument as a general proof of arithmetic optimality. Cross-group sharing remains a distinct search question.

**Schur complements give an exact local rewrite.** The block elimination described in [Higham's Schur-complement exposition](https://nhigham.com/2023/06/01/what-is-the-schur-complement-of-a-matrix/) applies whenever the eliminated block is invertible; positivity is not required for the algebra. Set

$$
M_g=\frac12\Lambda_g^{-1}C_g^\top,
\qquad S_g=H_g-\frac14 C_g\Lambda_g^{-1}C_g^\top.
$$

Expanding the square proves

$$
q_g=(y_g+M_gz_g)^\top\Lambda_g(y_g+M_gz_g)+z_g^\top S_gz_g.
$$

This is unique for the fixed partition and invertible diagonal $\Lambda_g$, not a canonical semantic decomposition. Construction costs $O(kl+k^2l)$ arithmetic and $O(kl+k^2)$ temporary storage per group. Tiny pivots may create large constants and cancellation; no positive-definite stability theorem is assumed. The implementation refuses relative diagonal pivots at or below $10^{-12}$ and reports all skipped groups and coefficient growth.

The input readers, output directions, shared-product bank and number of stored coefficients remain unchanged. `mix` replaces `cross` and Schur coefficients replace the old shared coefficients. For an affected group, both forms use $l+t+kl$ scalar coefficient multiplications and $l+t+kl-1$ additions, where $t=k(k+1)/2$ shared-product references. The old form also uses $k$ explicit shared/private variable products. Completing the squares removes those $k$ products. Offline divisions used to construct constants are not recurring execution operations.

**Alternatives and assumption checks.** CP/signed-square pairing, shared-square merging and global shared-product deduplication already exist in this repository; they are not new work. Sparse Tucker could propose another coordinate organization, but sparsity is not this operation count. Hierarchical Tucker and TT organize mode subspaces and do not supply an exact arithmetic or semantic hierarchy for the normalized network. Weighted-automaton minimal realization would require a verified finite linear state/Hankel representation that we do not have. [egg](https://arxiv.org/abs/2004.03082) could organize exact rewrite alternatives, but does not infer approximate weight equalities or automatically preserve our port interventions. No new framework is needed to test this one identity.

## Executed consequence: a cheaper graph with the same ports

[The compiler and control](schur_graph_rewrite_v1.py) transformed the frozen spectral graph with no skipped pivots. Its smallest relative pivot was 0.1045, maximum mixing coefficient 3.641, and maximum Schur-to-whole-core norm ratio 2.309. All A/B/C bars pass in [the primary receipt](SCHUR_GRAPH_REWRITE_V1.json):

- Variable multiplications decrease **1,041 → 1,016**, a saving of 25.
- Local scalar additions remain **1,329** and scalar coefficient multiplications **1,393**. Stored float and integer counts do not increase. Dense reads, writers, bias, adapters and native background remain required.
- Core identity error is at most $1.07\times10^{-16}$. FP64 full-function and tested parent-intervention errors are at most $5.65\times10^{-15}$.
- Synthetic FP32 full-function and intervention-effect replay is at most $3.03\times10^{-6}$, under the $10^{-5}$ bar.

The follow-up [reachable-input check](SCHUR_GRAPH_REWRITE_NATURAL_REPLAY_V1.json) was executed on all 384 cached native inputs from each of the FineWeb and corpus-shift panels. FP32 output error is at most $1.77\times10^{-7}$; parent-1 zero/swap effect error is at most $4.45\times10^{-6}$. These are surrogate/interface replays, not fresh model accuracy or throughput benchmarks. The artifact is a small changed-coefficient patch referring to the immutable source graph, not another copy of the entire model.

## Why intervention semantics matter

For $q=y^2+2zy=(y+z)^2-z^2$, clamping the parent $z=0$ gives $y^2$ in both programs. Deleting only the explicit shared term $-z^2$ from the rewritten expression instead leaves $(y+z)^2$. At $y=z=1$, the correct clamped output is 1 and the naive one is 4. This live planted control catches the mistake.

The compiler preserves the intervention by changing $z$ **before** forming every shifted square. It cannot reinterpret “remove the node” as “delete whichever explicit terms look shared after rewriting.” Equality of intact functions alone would not verify this boundary. The tested zero, joint-zero and donor-swap interfaces remain intact because the identity holds for arbitrary independent values of all these ports.

This advances executable arithmetic and manipulation semantics, rather than another rank/capture sweep. The saving is modest and does not establish a GPU speedup, a global minimum, a new circuit, or completion of the four properties. A data-independent compiler improvement to the frozen surrogate is now available; its functional approximation to the native model has not improved.

## Decision and continuation

Close the matched independent-reader screen at its valid miss. Keep complete-product contextual alignment as its narrower positive result. Do not rediscover already deduplicated shared products. Use the Schur rewrite as a concrete legal move in future graph search, with source ports retained, numerical checks and literal price attached. Broader graph changes must continue to be judged against the original weight tensor and eventually the four circuit properties.

Continuation was executed: after the synthetic rewrite passed, the natural-input FP32 parent-interface check also ran and passed. The durable program goal remains active. Next mathematical review is due **22:51 UTC**; hourly review remains due **20:27 UTC**.

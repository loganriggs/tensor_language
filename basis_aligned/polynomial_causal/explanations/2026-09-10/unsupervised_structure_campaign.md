# Unsupervised structure campaign — 10 September, 20:06 UTC

The user requested a broad structural search with substantial unlabeled data, enough optimization to establish convergence, and red-team review of negative results. This supersedes treating the short joint32 run as the main search. The four-property goal remains OOD prediction, extraction, selective manipulation, and composition/reuse; a better tensor fit only nominates components for those tests.

## What the first factorization assumed

The native layer already has an exact 4,608-product representation:

$$
B(x)=D[(Lx)\odot(Rx)],\qquad
S_v=\sum_k(UD)_{vk}\operatorname{sym}(l_kr_k^\top).
$$

We approximated all 50,304 $S_v$ jointly with 32 products:

$$
\widehat S_v=\sum_j(Uw_j)_v\operatorname{sym}(a_jb_j^\top).
$$

The loss was $\sum_v\|S_v-\widehat S_v\|_F^2$. For fixed input readers, output writers were solved by linear least squares; Adam then adjusted the readers. Exact Gram contractions avoided constructing a roughly 267 GB FP32 tensor. Readers had unit norm but were not required to be mutually orthogonal. Both input readers of each product were free, signed, dense vectors. The output writes were signed and dense too.

This imposes **few shared products**. It does not explicitly reward common readers between products, sparse computation graphs, output hierarchies, independent blocks, or simple upstream producers. It also gives equal weight to input-coefficient directions regardless of how often the model visits them. The full unembedding weights the output geometry, but the fit contains no natural-input distribution.

Only two individual terms matched across the two 240-step runs at a fixed .95 cosine bar. This is not a census of mechanisms. The random run was still improving at its final step. A newly executed control also gives identical shared-reader programs with individual-term cosines only .5: rotating $(x_0x_1,x_0x_2)$ together with their output coefficients preserves the entire function. A stable block can therefore fail individual-term matching. [Control](../../SHARED_READER_BLOCK_STABILITY_V1_CONTROL.json).

## Structural hypotheses to compare

The list below separates **representation**, **fitting metric**, and **validation** so their effects are not confused. Rows marked planned have not run. Each executed method must report its literal parameters and compute, not only its nominal rank.

| ID | Hypothesis / inductive bias | Representation or test | Initial status |
|---|---|---|---|
| 1 | Few shared bilinear products | Free $(a_j^\top x)(b_j^\top x)$, common vocabulary writers | Short pilot complete; converged larger comparison planned |
| 2 | Signed square features suffice | $(a_j^\top x)^2$ with signed output coefficients | Planned |
| 3 | Many products reuse a small reader dictionary | $a_j=E\alpha_j$, $b_j=E\beta_j$ | First-wave implementation target |
| 4 | A sparse multiplication graph over shared readers | Compute $h=E^\top x$ once; retain selected $h_i h_j$ edges | Planned |
| 5 | Small dense interaction blocks are meaningful units | Several low-rank quadratic blocks, one/few output codes per block | First-wave implementation target |
| 6 | Common input subspace plus sparse interaction core | Symmetric-input Tucker form with explicit core sparsity | Planned; plain dense Tucker is a control |
| 7 | Some subsystems do not interact | Simultaneous block structure in the output quadratic family | Planned; stronger than shared-reader DAGs |
| 8 | Output computations share low-dimensional writer spaces | Factor the output mode, retaining explicit quadratic functions inside each group | Planned; compare existing MLP17 dossiers |
| 9 | Vocabulary groups share computations with corrections | Shared output-group codes plus token-specific remainders | Planned; require cheaper joint implementation |
| 10 | Overlapping output groups matter more than a tree | Graded signed factor usage and overlapping group structure | Dense-support pilot audited; broader comparison planned |
| 11 | Unembedding hierarchy is useful before folding | Learn group/contrast coordinates from U alone, then fold those readers | Planned; no linguistic labels used in discovery |
| 12 | Important structure is simple on natural states | Full-output quadratic fit under unlabeled activation distribution | First-wave data preparation |
| 13 | A constant and linear part hide simpler interactions | Center inputs; separate affine term from remaining products | Planned; charge the affine map and offsets |
| 14 | State covariance obscures shared readers | Compare raw versus covariance-standardized input coordinates | Planned; preserve metric and translate interventions back |
| 15 | Typical fit misses causal directions | Include unlabeled local perturbation/derivative response loss | Planned; not just activation reconstruction |
| 16 | Multiple input regimes use related computations | Shared global dictionary with regime-specific coefficients/blocks | Planned; price gates and test OOD regime boundaries |
| 17 | Sparse feature use across contexts identifies units | Regularize activations or group use on unlabeled states | Planned; no assumed nonnegativity |
| 18 | Non-Gaussian feature statistics identify a useful basis | Higher-order statistics/independence within candidate blocks | Planned; test assumptions before naming components |
| 19 | Upstream simplicity identifies among equivalent decompositions | Prefer readers with a smaller folded producer through attention/earlier MLPs | Planned; last-attention connection remains active |
| 20 | Cross-layer shared variables reveal stronger structure | Coupled decomposition of downstream readers and their upstream producers | Planned |
| 21 | Stable objects are blocks rather than individual terms | Compare subspaces, joint tensor contributions and intervention behavior under gauge changes | Exact counterexample complete; native tests planned |
| 22 | Weak coefficient modes carry meaningful computations | Compare components by held-out causal response, not only coefficient energy | Planned; existing low-variance/high-loss lesson applies |
| 23 | Antipodal symmetry should be explicit | Preserve $f(-x)=f(x)$; use paired-state controls and signed outputs | Native/product forms already even; expanded controls planned |
| 24 | Dense output support is an interface constraint | Test support feasibility and optimal concentration before sparse token wiring | Three-group numerical bound complete |

CP/Tucker and block-term decompositions supply established representation families, not automatic mechanism identification. [Kolda–Bader](https://www.kolda.net/publication/koba09/), [De Lathauwer: block terms and simultaneous block diagonalization](https://ftp.esat.kuleuven.be/sista/delathauwer/reports/ldl-12-61.pdf).

## First substantial data panel

Capture MLP17's normalized input and native output from 1,000 cached corpus sequences of 512 input tokens: **512,000 token positions processed**. Save 64 deterministically sampled positions after position 64 per sequence: **64,000 input/output pairs**. Fixed row splits are 800 training, 100 validation, and 100 test rows, corresponding to 51,200/6,400/6,400 stored pairs. No task labels determine fitting or sample inclusion. All source rows and sampled positions are frozen.

These corpus rows were historically opened. The splits measure generalization within this campaign; they are not fresh document-level or OOD confirmation. The source lacks a dependable document grouping guarantee, which will be stated rather than inferred from row separation. Later promotion needs a separate corpus/construction holdout.

States are stored in FP16 to fit local disk constraints; capture reports the measured rounding floor relative to native FP32 values. Optimization can use FP32/FP64 after loading. No metric may claim accuracy below the measured data floor. The first wave combines weight-only fits (no activations) with explicitly labeled activation-weighted fits, keeping their evidential roles distinct.

## Optimization and convergence policy

The first wave compares free products, shared-reader products, and quadratic blocks, with square/Tucker controls. Start with a materially larger component budget than 32, then compare models by literal storage and computation. Capacity changes are controls, not the main scientific claim. Use at least four starts for each promoted representation, including random starts and a weight-informed start where meaningful.

Do not call a run converged merely because its step budget ended. Record the complete loss history, improvement relative to **captured** energy (not a nearly constant total residual), gradient/stationarity diagnostics, conditioning, learning-rate changes and parameter/function changes. Continue optimization in resumable managed chunks; a chunk boundary is not a convergence verdict. Use lower-rate or second-stage refinement when progress continues. If numerical or resource limits prevent convergence, record **optimization unresolved**, not **structure absent**.

Nonconvex optimization does not provide a global-optimum guarantee. Agreement across starts, planted recoverability controls, convergence diagnostics and independent algorithms strengthen a local result; they do not turn it into a theorem of nonexistence.

## Red-team gate for every negative

Before interpreting a negative as evidence against a structural hypothesis:

1. Verify the actual function, normalization, residual branches, signed coefficients and precision floor.
2. Confirm optimization reached the registered convergence criteria; inspect the trajectory independently of its final scalar.
3. Test a planted example with the proposed structure at the same scale/conditioning when feasible.
4. Check capacity, parameterization and initialization; distinguish representation restrictions from optimizer failure.
5. Compare individual-factor and block/function stability. Do not mistake a gauge change for disappearance of a computation.
6. Inspect coefficient versus natural-state versus intervention metrics. A negative in one metric does not automatically transfer to another.
7. Check data coverage, held-out leakage and how the sampling/consumer grouping was selected.
8. Preserve the original failed test and state the narrow claim it actually rejects. Stronger impossibility claims require a valid mathematical bound or sufficiently broad independent evidence.

This is adversarial examination of the method, not a requirement to obtain a positive answer. A valid negative remains useful. The scientific target is a recovered executable computation with the four properties, rather than a favorable fit or a preferred interpretation.

## Execution ledger

- Completed: short joint32 fit, stable-product causal screen, fixed-support feasibility, shared-reader block counterexample. Their failures remain intact.
- In preparation: managed 1,000-sequence data capture; convergence-controlled larger weight fits; shared-reader/block implementations.
- Pending: joint decomposition of the longer unembedding → last bilinear → last attention path. The earlier position-corrected QK work was a separate two-behavior experiment.

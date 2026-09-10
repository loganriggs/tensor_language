# Unsupervised structure campaign — 10 September, updated 21:53 UTC

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
| 1 | Few shared bilinear products | Free $(a_j^\top x)(b_j^\top x)$, common vocabulary writers | Larger weight chunk complete but unconverged; data fit running |
| 2 | Signed square features suffice | $(a_j^\top x)^2$ with signed output coefficients | Implemented and controlled; native runs pending |
| 3 | Many products reuse a small reader dictionary | $a_j=E\alpha_j$, $b_j=E\beta_j$ | Weight chunk complete but unconverged; data counterpart queued |
| 4 | A sparse multiplication graph over shared readers | Compute $h=E^\top x$ once; retain selected $h_i h_j$ edges | Planned |
| 5 | Small dense interaction blocks are meaningful units | Several low-rank quadratic blocks, one/few output codes per block | Implemented and controlled; data run queued |
| 6 | Common input subspace plus sparse interaction core | Symmetric-input Tucker form with explicit core sparsity | Planned; plain dense Tucker is a control |
| 7 | Some subsystems do not interact | Simultaneous block structure in the output quadratic family | Planned; stronger than shared-reader DAGs |
| 8 | Output computations share low-dimensional writer spaces | Factor the output mode, retaining explicit quadratic functions inside each group | Planned; compare existing MLP17 dossiers |
| 9 | Vocabulary groups share computations with corrections | Shared output-group codes plus token-specific remainders | Planned; require cheaper joint implementation |
| 10 | Overlapping output groups matter more than a tree | Graded signed factor usage and overlapping group structure | Dense-support pilot audited; broader comparison planned |
| 11 | Unembedding hierarchy is useful before folding | Learn group/contrast coordinates from U alone, then fold those readers | Planned; no linguistic labels used in discovery |
| 12 | Important structure is simple on natural states | Full-output quadratic fit under unlabeled activation distribution | 64,000 state pairs captured; product fit running |
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
| 25 | Avoid large cancelling components | Scale-invariant component-energy/cancellation penalty; compare explicit blocks | Fixed-reader path executed; joint-fit objective implemented and controlled |

CP/Tucker and block-term decompositions supply established representation families, not automatic mechanism identification. [Kolda–Bader](https://www.kolda.net/publication/koba09/), [De Lathauwer: block terms and simultaneous block diagonalization](https://ftp.esat.kuleuven.be/sista/delathauwer/reports/ldl-12-61.pdf).

## First substantial data panel

Capture MLP17's normalized input and native output from 1,000 cached corpus sequences of 512 input tokens: **512,000 token positions processed**. Save 64 deterministically sampled positions after position 64 per sequence: **64,000 input/output pairs**. Fixed row splits are 800 training, 100 validation, and 100 test rows, corresponding to 51,200/6,400/6,400 stored pairs. No task labels determine fitting or sample inclusion. All source rows and sampled positions are frozen.

These corpus rows were historically opened. The splits measure generalization within this campaign; they are not fresh document-level or OOD confirmation. The source lacks a dependable document grouping guarantee, which will be stated rather than inferred from row separation. Later promotion needs a separate corpus/construction holdout.

States are stored as scaled FP16 plus per-state FP32 scales to fit local disk constraints; capture reports the measured rounding floor relative to native FP32 values. Optimization can use FP32/FP64 after loading. No metric may claim accuracy below the measured data floor. The first wave combines weight-only fits (no activations) with explicitly labeled activation-weighted fits, keeping their evidential roles distinct.

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

For each future negative, record the narrow failed claim, the strongest plausible methodological explanation, and an executed check that distinguishes them. If that check has not run, label the structural interpretation **red-team audit pending**. Keep the original metric and threshold visible when testing a different representation or metric.

The first larger weight-product fit already illustrates why this matters. At 20:28 its loss was unchanged across successive checks, but relative stationarity was about $10^{-2}$ against the registered $10^{-4}$ bar, and the Gram condition number had risen above $10^6$. That is a stalled optimization trajectory requiring numerical/parameterization investigation, not a converged negative. Candidate checks include a mathematically equivalent objective without its constant term and an explicitly registered penalty against large cancelling components; the executed follow-up checks are reported below.

## What the red-team checks found, through 20:44

**The weight fit is numerically awkward, and natural-input structure looks considerably simpler than its coefficient error suggests.** Neither of the two completed weight chunks converged. Their saved functions nevertheless explain roughly 91–92% of the unembedding-weighted MLP output energy on validation states. The natural-state product fit is still running; its training error is lower than a larger affine baseline, but its first validation error is 0.02585. The quadratic-block fit reaches 0.02439 with fewer parameters. These are candidate representations, not identified circuits.

### Why the product fit stalled

The 128-product weight fit ended its 540-second chunk at squared coefficient error $0.9126275$, after 38,375 objective evaluations. Its saved L-BFGS step length was zero. An independent CPU computation replayed the GPU loss within $4.6\times10^{-12}$; differentiating an equivalent objective agreed with the original gradient within $6.9\times10^{-9}$ relative error. Fresh L-BFGS, removing the constant from the loss, and differentiating the projected objective all improved the loss by less than $5\times10^{-10}$, failing the registered $10^{-8}$ improvement bar. Simply rewriting the scalar loss did not solve the problem. [Audit](../../STRUCTURED_FIT_STALL_V1_AUDIT.json).

The sum of the separate components' squared energies was **14,264.96 times** the squared energy of their sum. This measures cancellation in the chosen representation; it is not a count of mechanisms or a basis-independent impossibility statement. The closest pair had input-quadratic cosine $-0.9999974$ and output-writer cosine $+0.9999996$: nearly the same computation written in opposing directions.

We regrouped that pair exactly. After orienting its readers consistently, write

$$
a_{1,2}=A\pm\delta a,\qquad b_{1,2}=B\pm\delta b,
\qquad w_{1,2}=W\pm\delta w.
$$

Then its output is

$$
2W\left[(A^\top x)(B^\top x)+(\delta a^\top x)(\delta b^\top x)\right]
+2\delta w\left[(A^\top x)(\delta b^\top x)+(\delta a^\top x)(B^\top x)\right].
$$

This exposes two quadratic blocks sharing four readers. On 256 fixed Gaussian inputs, the rewritten full program replayed with relative error $5.3\times10^{-14}$. Its cancellation ratio fell to 1,746.66—an **8.17× reduction**, which misses the preregistered 10× target. The rewrite also uses 130 rather than 128 scalar products. It improves the arithmetic representation of one pair; it does not improve the fitted function, prove convergence, or satisfy the simplicity goal. Collective cancellation beyond nearly identical pairs remains untested. [Regrouping receipt](../../CANCELLING_PRODUCT_BLOCKS_V1_AUDIT.json).

### An explicit new bias against cancellation

For feature Gram matrix $G$ and output metric $M=U^\top U$, consider

$$
\mathcal L_\lambda
=\mathcal L_{\mathrm{reconstruction}}
+\frac{\lambda}{\|T\|^2}\sum_j G_{jj}\,w_j^\top M w_j.
$$

The added term penalizes the sum of separate component energies and is unchanged when an individual feature is rescaled and its writer inversely rescaled. For fixed readers, if $C$ is the target–feature cross matrix, the exact conditional writer solution is

$$
W=C\left(G+\lambda\operatorname{diag}(G)\right)^{-1}.
$$

This is an explicitly different inductive bias. Both the original reconstruction error and the penalty must be reported. The penalty is not invariant to splitting one component into multiple copies, so component count and literal implementation cost remain relevant controls.

On the frozen stalled readers, $\lambda=0.01$ lowered the cancellation ratio from 14,265 to **1.245**, while retaining about **80% of the coefficient energy captured by the unregularized fit**. Absolute captured energy fell from 8.74% to 6.98% of the native tensor. No readers were refitted in this probe. The new joint-fit objective is implemented; dense-function and gradient controls passed for all four representations and both metrics. The joint native weight fit is now preregistered and queued. The new optimizer tracks the penalized objective, keeps the original gradient thresholds, and passed both a planted-optimum/resume control and a nonstationary-stall control. Its screen additionally requires improvement over the already-measured fixed-reader penalty baseline. [Fixed-reader path](../../STRUCTURED_FIT_FIXED_READER_PENALTY_V1_AUDIT.json), [objective controls](../../ENERGY_REGULARIZED_QUADRATIC_V1_CONTROL.json).

### Natural-state baselines and validation

All entries below use the same unembedding-weighted bilinear MLP contribution, with the fixed native output bias subtracted and before final residual normalization and score saturation. Lower squared relative error is better. The affine model was fitted on the training rows only; test rows remain unopened.

| Representation | Stored scalar parameters | Training error in this metric | Validation error |
|---|---:|---:|---:|
| Constant mean output | 1,152 | 0.29557 | 0.29653 |
| Affine map | 1,328,256 | 0.02932 | 0.03156 |
| 128 products fitted to coefficients; unconverged | 442,368 | Not evaluated | 0.09341 |
| 64 shared readers / 128 products fitted to coefficients; unconverged | 237,568 | Not evaluated | 0.08208 |
| 128 products fitted to natural states; unconverged | 442,368 | 0.01623 | 0.02585 |
| 32 quadratic blocks, each combining eight squares; unconverged | 332,032 | 0.02084 | 0.02439 |

The shared-reader fit has worse coefficient reconstruction than the free-product fit but better natural-state validation error in these first starts. This demonstrates that the two metrics can rank partial fits differently. Different initializations and nonconvergence prevent a final method comparison. Both natural-state fits beat the affine baseline on this validation split: about 18% less error for products and 23% less for blocks. Blocks use about 25% fewer parameters than free products and about one-quarter the affine parameters. These are single-start, unconverged results; stability and the four circuit-property tests remain open. [Baselines](../../NATURAL_STATE_BASELINES_V1_AUDIT.json), [weight-product validation](../../STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00_VALIDATION.json), [shared-reader validation](../../STRUCTURED_FIT_V1_weight_shared_reader_s0_CHUNK_00_VALIDATION.json).

The original data-fit runner failed before optimization because it requested `Down.bias`; the native parameter is `Down_bias`. V2 corrects the key and passed a 64-state native-output replay check at relative error $1.79\times10^{-4}$ against a $10^{-3}$ bar. The failure remains preserved, and no data were recaptured. [Repair and scope](../../STRUCTURED_FIT_DATA_V2_REPAIR.md).

## Weight-only output-function bound and its red-team check, 21:03

A different factorization could share output patterns while allowing each input
function to be an arbitrary quadratic, rather than one product. We computed the
best possible coefficient-energy capture for this more permissive class. With
$G$ the exact native product Gram and $M=U^\top U=CC^\top$, the small output
covariance is

$$
K=C^\top DGD^\top C.
$$

Its eigenvalues are the squared singular values of the full output unfolding.
Keeping $k$ arbitrary quadratic output functions can capture at most the sum of
the largest $k$ eigenvalues divided by their total. The first 32 capture **17.82%**;
128 capture **34.05%**. The registered 50% target for 32 fails. The leading four
output functions individually admit best single-real-product captures of only
11.0%, 16.8%, 10.1% and 6.1%, failing the 90% target. Dense toy, trace and eigen
replay checks passed. These bounds do not depend on a nonconvex optimizer getting
stuck. [Full result](../../FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json).

The conclusion is narrow: the full coefficient tensor is broad in this particular
metric. It does not establish that computation on natural states is complex or
that shared circuits are absent. The older dossier's 99.7% output retention at
width512 used a **context-covariance-transformed input metric**, as confirmed in
`mlp_mode_concentration_depth_profile.py`; it was not the same coefficient object.
Its earlier causal and OOD results remain intact.

We red-teamed the new negative using the already-known normalization/trace
identity. A quadratic $x^\top I x$ is constant on a fixed-radius sphere, despite
its full-rank coefficient matrix. For all outputs, with
$t_k=l_k^\top r_k$,

$$
B(x)=\frac{\|x\|^2}{d}Dt
+D\left[\phi(x)-\frac{\|x\|^2}{d}t\right].
$$

The second term has trace-free quadratic coefficients. We retained the actual
$\|x\|^2/d$ factor, so this identity does not assume exact RMS normalization or
discard its epsilon correction. The native split replayed at $3.0\times10^{-15}$.
The isotropic part accounts for only **0.295% of coefficient energy**; removing
it leaves top32 capture **17.58%**, so that particular explanation does not repair
the coefficient-space result. Natural-state weighting still matters substantially,
as the empirical fits above demonstrate. [Normalization red-team receipt](../../FULLU_TRACE_METRIC_V1_AUDIT.json).

## Execution ledger at 21:03 (superseded by the appended update below)

- Completed chunks, all still unconverged: weight products, weight shared readers,
  natural-state products, and natural-state quadratic blocks. Validation receipts
  are saved for all four. No final comparison across restarts yet.
- Running since20:57:45: natural-state shared-reader fit.
- Queued: explicit energy-penalty weight fit, followed by continuation of the
  original natural-state product optimizer with unchanged objective and thresholds.
- Frozen best-model artifacts preserve the first weight-product and data-product
  functions before resumable checkpoints change. Those compact artifacts are
  committed; full optimizer checkpoints and the large captured-state tensor
  remain local-only.
- Pending: remaining starts/representations, converged comparisons, OOD and causal
  promotion, and joint decomposition of the longer unembedding → last bilinear →
  last attention path. The earlier position-corrected QK work remains separate.

## Your questions: one million tokens, optimization, and cost — 21:14 UTC

**Yes: using real model inputs to weight the folded tensor is a useful, still
unsupervised approach. And a new optimization result has just arrived: the
explicitly penalized weight-product fit passed its local convergence checks in
57 seconds.** The earlier unpenalized fit remains unconverged; this is a changed
objective, not a retroactive convergence claim. The larger natural-state fits
remain unfinished.

### What running a million tokens would add

Let $x$ be the normalized input to the last bilinear layer, and write its
bias-free contribution as

$$
B(x)=D[(Lx)\odot(Rx)],\qquad f(x)=UB(x).
$$

$U$ is the entire unembedding. The folded coefficient tensor has one quadratic
interaction matrix per output token. A coefficient-space fit compares those
matrices without asking which combinations of input coordinates occur in text.
Instead, run text through the native model and minimize

$$
\mathcal L_{\rm data}
=\frac{\sum_{n=1}^{N}\|U[B(x_n)-\widehat B(x_n)]\|_2^2}
       {\sum_{n=1}^{N}\|UB(x_n)\|_2^2}.
$$

Common contexts then contribute more often; directions that barely occur carry
less weight. No semantic labels or preselected token groups are necessary. We
still compare **all output-token coordinates**, efficiently using
$M=U^\top U$, rather than materializing an $N\times50{,}304$ score matrix.
This weights the **input distribution**. Weighting output tokens by frequency or
model probability would be an additional, different choice; it is not silently
included here.

There is a precise tensor interpretation. If $\Delta Q_t$ is the error in token
$t$'s quadratic interaction matrix, then

$$
\mathbb E\!\left[\sum_t(x^\top\Delta Q_t x)^2\right]
=\sum_{t,i,j,k,l}(\Delta Q_t)_{ij}(\Delta Q_t)_{kl}
\underbrace{\mathbb E[x_i x_j x_k x_l]}_{\text{input fourth moment}}.
$$

Thus text supplies a metric on **pairs of input coordinates**, which is exactly
where the bilinear interactions live. The input covariance alone does not fully
specify this metric. Under a zero-mean Gaussian approximation with covariance
$S$, it would reduce to

$$
\mathbb E[(x^\top Qx)^2]
=2\operatorname{tr}(QSQS)+[\operatorname{tr}(QS)]^2
$$

for symmetric $Q$. Actual normalized model states need not be Gaussian. Fitting
on sampled real states retains empirical fourth-order information without
storing a $1152^4$ array. Covariance whitening can help conditioning, but does not
replace that empirical objective.

We have already processed **512,000 input tokens**, storing **64,000 layer-state
pairs** and fitting on 51,200 training states. This distinction matters: processed
tokens and optimization examples are different counts. The proposed extension
is at least **1,048,576 processed tokens**, with document-separated training,
validation and test data, and a reproducible sample of layer inputs. Larger and
more diverse data can reveal whether the current patterns generalize and reduce
sampling error; it cannot guarantee that a chosen factorization bias is right.
The million-token extension is being prepared; it has not run at this timestamp.
Any replacement corpus will be named explicitly, rather than assumed to match
the model's training distribution. Rare but causally important contexts still
need their own held-out checks.

### What the product optimizer actually does

For 128 products, the candidate is

$$
\widehat B(x)=\sum_{j=1}^{128}w_j(a_j^\top x)(b_j^\top x).
$$

The vectors $a_j,b_j$ read two directions in the layer input; their scalar
readings are multiplied; $w_j$ writes the result back into residual space.
The token factor is $Uw_j$: every token has a coefficient on the same learned
product. Readers are normalized during evaluation to remove simple scale
ambiguities. Products can still overlap or cancel, and exchanging $a_j,b_j$
leaves their function unchanged.

For each proposed set of readers, form
$F_{nj}=(a_j^\top x_n)(b_j^\top x_n)$ and target rows $Y_n=B(x_n)^\top$.
The output writers are solved by linear least squares at every evaluation:

$$
G=F^\top F,\qquad C=Y^\top F,\qquad W=CG^{-1}.
$$

Here $W$ has the writers as columns; implementations solve a linear system
rather than explicitly constructing the inverse. This conditional solution
also minimizes the full-$U$ loss when $U^\top U$ is positive definite. The
weight-only version uses analytically computed inner products of quadratic
coefficient matrices in place of sampled $F$. Eliminating the linear parameters
this way is called **variable projection**.

The nonlinear reader search currently uses 2,000 full-batch **Adam** steps
(adaptive gradient updates, learning rate 0.03), then **L-BFGS** (an approximate
curvature method) with a strong-Wolfe line search. Each outer L-BFGS call can
take multiple inner iterations and evaluate the loss and gradient multiple
times. We call one such evaluation a **closure**. Therefore “steps,” outer
calls and closures are not interchangeable compute counts.

Fits save checkpoints in roughly nine-minute chunks. A chunk ending does not
mean convergence. We require a plateau over five L-BFGS diagnostic checks,
relative stationarity at most $10^{-4}$ and maximum absolute gradient at most
$10^{-7}$. These are numerical local stopping checks, not a proof of global
optimality, uniqueness or circuit identification. Restarts and function/block
stability remain necessary.

### Why the old run stalled, and what just improved

In the old unpenalized weight fit, separately large components almost cancelled:
their summed energies were **14,265 times** the energy of their joint function.
The saved line-search step was zero. Fresh L-BFGS and equivalent objective
rewrites failed the registered improvement threshold. More calls to that same
stalled optimizer were therefore not useful evidence against structure.

The new objective explicitly prices separate component energies:

$$
\mathcal L_\lambda
=\mathcal L_{\rm reconstruction}
+\lambda\frac{\sum_j G_{jj}\|Uw_j\|_2^2}{\|T\|^2},
\qquad
W=C[G+\lambda\operatorname{diag}(G)]^{-1}.
$$

With $\lambda=0.01$, readers and writers were jointly refitted from the saved
weight fit. The outcome was:

| Measurement | Old unpenalized fit | New explicitly penalized fit |
|---|---:|---:|
| Captured coefficient energy | 8.737% | **8.699%** |
| Cancellation ratio | 14,265 | **1.219** |
| Local convergence checks | Failed | **Passed** |
| Added fitting time for this run | 540 seconds | **57 seconds** |

The new run used 3,931 closures and retained **99.56%** of the old fit's captured
energy. Its relative stationarity was $7.24\times10^{-5}$ and maximum gradient
$1.69\times10^{-9}$. All three preregistered checks—instrument, convergence,
and useful stable fit—passed. This is a meaningful numerical improvement. It
does **not** establish that 128 products capture all the structure, or that the
components are circuits. The 57 seconds are additional cost: this warm start
also depended on the earlier 540-second run. [Native result](../../PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_00.json).

### What the literature suggests trying next

1. **More stable variable projection.** O’Leary and Rust describe solving the
   linear subproblem with QR or SVD and using a nonlinear least-squares solver
   for the remaining parameters. Our current normal equations form $F^\top F$,
   which squares the condition number of $F$. QR/SVD is a direct candidate
   improvement for the empirical fits. A Gauss–Newton version must include how
   the solved writers change with the readers in its reduced residual Jacobian;
   our valid first-gradient shortcut is not a complete second-order method.
   [Variable Projection for Nonlinear Least Squares Problems](https://www.cs.umd.edu/users/oleary/software/varpro.pdf).
2. **Damped Gauss–Newton / Levenberg–Marquardt.** These methods exploit the
   least-squares structure and damp poorly determined update directions.
   Phan, Tichavský and Cichocki study CP fits with collinear factors, a close
   match to our observed numerical difficulty. Our symmetric products and
   output metric require adaptation; their measured speedups are not promises
   for this model. [Low Complexity Damped Gauss–Newton Algorithms for CANDECOMP/PARAFAC](https://arxiv.org/abs/1205.2584).
3. **Matrix-free curvature solves.** Singh and colleagues compare Gauss–Newton
   and alternating least squares using structured, preconditioned iterative
   solves. This matters because our 128-product reader model has 294,912
   nonlinear scalar parameters: its dense FP64 Hessian alone would require
   about 696 GB. Products of a Jacobian or curvature operator with vectors can
   avoid materializing that matrix. [Comparison of Accuracy and Scalability of Gauss–Newton and Alternating Least Squares for CP Decomposition](https://arxiv.org/abs/1910.12331).
4. **Treat diverging cancellation as a representation issue too.** De Silva
   and Lim show that some higher-order tensor low-rank approximation problems
   have no attained best approximation. This does not prove our particular
   problem is ill-posed, but it explains why large cancelling factors cannot
   automatically be cured by more iterations. Explicit energy penalties and
   shared blocks are defensible alternatives, with their changed objectives
   reported openly. [Tensor rank and the ill-posedness of the best low-rank approximation problem](https://arxiv.org/abs/math/0607647).

The energy-penalized fit above has actually run. QR/SVD-based refinement and
matrix-free Gauss–Newton are proposed upgrades, not completed experiments.

### Is running many more steps expensive?

It is practical to run substantial optimization on the GPU. Collecting inputs
is currently much cheaper than repeatedly fitting on them:

| Operation | Measured time | Loss/gradient evaluations |
|---|---:|---:|
| Native model capture: 512,000 tokens | 14.2 seconds | Not a fit; 250 body forwards |
| Unpenalized weight products, first chunk | 540 seconds | 38,375 closures, including stalled calls |
| Natural-state products, first chunk | About 540 seconds | 7,133 closures |
| Natural-state quadratic blocks, first chunk | 541 seconds | 3,464 closures |
| Penalized weight products, warm-start refinement | 57 seconds | 3,931 closures |

For the two data fits this averages roughly 76 and 156 milliseconds per closure,
including chunk overhead. A closure is therefore reasonably cheap, but tens of
thousands across many starts add up. One nine-minute chunk for each of the 32
registered configurations would cost **4.8 GPU-hours**; further chunks add to
that. This is a compute estimate, not a promised convergence time. The existing
512,000-token capture suggests that a million-token forward pass is affordable;
data loading, storage and any extra statistics must also be measured.

The fitting code currently uses FP64 for numerical reliability. Mixed precision
is another candidate speed improvement, provided full-precision replay and
gradient checks hold. More data should be staged with representative batches
for early optimization and fixed, larger panels for refinement and convergence
checks; a noisy minibatch plateau alone would not satisfy our current stopping
criterion. The immediate priorities are better conditioning, larger independent
data coverage, and converged restart comparisons—not treating a fixed step
budget as a scientific negative.


### Execution follow-up, 21:18 UTC

The million-token panel is now prepared and submitted through the managed queue:
2,048 distinct exact-text documents from cached **Pile-10k**, one 512-token input
prefix each, with 1,600/224/224 document splits. It will save 32 sampled inputs
per document (65,536 total), while all 819,200 training positions contribute to
input mean/second-moment estimates. This doubles processed-token coverage and
roughly doubles document count relative to the old panel, **not the number of
stored fitting states**. Long-document/prefix selection and possible near-
duplicates remain limitations. These are preparation facts, not capture results.
[Registered capture](../../MILLION_TOKEN_PANEL_V1_PREREGISTRATION.md).

The natural-state shared-reader fit completed without convergence: validation
squared error **0.01944**, versus 0.03156 for the affine baseline, with 237,568
parameters. The converged penalized weight fit validates at **0.09117**, slightly
better than the old weight fit's 0.09341 but still much worse than the empirical
fits. Numerical conditioning and choosing a relevant input metric are separate
issues. [Shared-reader validation](../../STRUCTURED_FIT_V2_data_shared_reader_s0_CHUNK_00_VALIDATION.json),
[penalized-fit validation](../../PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_00_VALIDATION.json).

The original data-product fit also received another nine minutes with its
unchanged optimizer. Training error moved only from 0.0162327 to 0.0162126 and
stationarity still failed. This is further evidence for testing a better solver
before spending many more identical chunks; it is not evidence of absent
structure. [Continuation receipt](../../STRUCTURED_FIT_V2_data_product_s0_CHUNK_01.json).


### Million-token capture completed, 21:18:19 UTC

The managed run **completed in 27.74 seconds**: 1,048,576 processed tokens from
2,048 distinct document prefixes; 65,536 saved inputs; and mean/second-moment
statistics from all 819,200 training positions. All three registered capture
checks passed. Maximum input rounding error was $1.60\times10^{-4}$; regenerating
the bias-free bilinear output from rounded inputs differed by
$1.10\times10^{-4}$ on the registered training replay panel. The input-only
artifact is 162.4 MB and remains local; source rows, hashes and receipts are
preserved separately. **The new dataset has been captured, but a decomposition
has not yet been fitted on it.** [Capture result](../../MILLION_TOKEN_PANEL_V1_RESULT.json).

The extra nine-minute product continuation also worsened old-panel validation
error from **0.02585 to 0.02944**, despite the small training improvement. The
first chunk's function was frozen before continuation, so it remains available.
This is a reason to compare generalization and optimizer conditioning alongside
training convergence; neither longer fitting nor a smaller training loss alone
selects the useful structure. [Second-chunk validation](../../STRUCTURED_FIT_V2_data_product_s0_CHUNK_01_VALIDATION.json).

The first training-only metric audit confirms substantial input anisotropy: the mean accounts for47.37% of input squared norm; after centering, the leading32 covariance directions account for58.31% of variance. Both moment matrices are numerically positive definite. This supports testing input preconditioning and separating mean/affine effects; covariance modes alone neither identify circuits nor determine quadratic fourth-moment error. [Training metric audit](../../MILLION_TOKEN_TRAINING_METRIC_V1_AUDIT.json).


### Fixed-reader transfer and solver follow-up, 21:26 UTC

A new managed experiment is queued to evaluate five frozen factorized functions
on the Pile validation panel and then refit only their output writers on Pile
training inputs. It reports all sampled positions and positions64..511
separately, because the original training panel excluded the first64 positions.
Constant and affine controls use the same new training inputs. The question is
whether the same input products remain useful across corpus and position
coverage, with or without output recalibration. No test split is used.
[Registered comparison](../../PILE_FIXED_READER_TRANSFER_V1_PREREGISTRATION.md).

A QR-based variable-projection objective is now implemented. It solves the
writers using QR and evaluates the direct residual after whitening output space
by the full unembedding metric. There is no hidden ridge and no derivative
through QR: the envelope theorem supplies the first gradient. Across all four
small representation controls, objective discrepancies were at most
$2.22\times10^{-16}$ and finite-difference gradient discrepancies below
$9\times10^{-12}$. A deliberately ill-conditioned feature matrix with condition
number $10^7$ agreed with the full SVD solution in function space to
$2.38\times10^{-10}$. These are implementation controls, not a native optimizer
speedup or convergence result. The queued native comparison checks QR against
SVD and normal equations before using the new writer fits.
[Controls](../../STABLE_EMPIRICAL_QUADRATIC_V1_CONTROL.json).


### Fixed-reader transfer completed, 21:26:07 UTC

All three registered checks passed. The five-model comparison took4.94seconds
end to end. The same input products were retained; only the output writers were
refitted on Pile training inputs. Lower full-U validation squared error is better:

| Frozen input representation | Original writers | Pile-fitted writers |
|---|---:|---:|
| Data products128 | 0.02136 | 0.01481 |
| Data shared readers64 / products128 | **0.01564** | **0.01402** |
| Data blocks32x8 | 0.02040 | 0.01830 |
| Weight products128 | 0.07520 | 0.01851 |
| Penalized weight products128 | 0.07468 | 0.01871 |

The new training-fitted affine baseline validates at0.02375. On matched
positions64..511, shared readers improve from0.015368 to0.013704, a10.83% error
reduction, passing the registered10% recalibration threshold. The unchanged
shared-reader function also passed the transfer threshold. These observations
support reusable input functions across these two corpus panels and show that
input-distribution weighting can help substantially through the output writer
solve alone. They do not yet identify individual semantic circuits or establish
OOD behavior, extraction, selective removal or composition.
[Full comparison](../../PILE_FIXED_READER_TRANSFER_V1_RESULT.json).

QR and SVD predictions agreed within$7.8\times10^{-14}$; QR and normal equations
within$1.5\times10^{-9}$ on these training matrices. Thus the conditional writer
solve is accurate here even for ill-conditioned features. This does not yet
show that QR improves the nonlinear optimizer. The individual `qr_seconds`
fields time asynchronous host submission and **must not be used as GPU solver
benchmarks**; only the end-to-end wall time is meaningful for this run.

An exact coefficient-function audit then compared old and refitted output
functions with readers unchanged. Shared readers retained cosine0.98575 and
90.57% of the old coefficient norm; blocks retained cosine0.99329 and85.56%.
The weight-only product functions changed more, to cosines0.78290 and0.76370
for unpenalized and penalized fits. This is consistent with different output
weighting being particularly important for the coefficient-trained models.
These are whole-function comparisons, not a claim that individual components
are stable across independent nonlinear fitting restarts.
[Writer geometry](../../PILE_WRITER_GEOMETRY_V1_AUDIT.json).


### Physical model replacement: useful structure, insufficient preservation — 21:43 UTC

The shared-reader program now has an executable implementation that computes
its64 input projections once, mixes them into128 product pairs, and writes the
products back into the residual stream. The native output bias is included. It
uses238,720 scalar coefficients versus15,926,400 in the native last MLP. All
other layers and the unembedding remain native; this is not a comparably large
saving for the whole model and does not supply independent circuit producers.

We physically installed the frozen implementations in MLP17 and ran the224 Pile
validation documents through the model, retaining its actual final RMS and
score saturation. Both no-op and native-factor controls reproduced the native
scores and cross-entropy exactly. The full run took12.34seconds,226bodyforwards /
904sequences. At7,168 sampled next-token positions:

| Installed MLP17 | Mean CE added above native | Mean absolute token CE change | Native top1 agreement | Native-to-candidate KL |
|---|---:|---:|---:|---:|
| Shared readers, original writers | +0.05828 | 0.25275 | 87.25% | 0.05855 |
| Shared readers, Pile-fitted writers | +0.05879 | 0.23836 | 87.65% | 0.05281 |
| Free products, Pile-fitted writers | +0.05414 | 0.22544 | 87.79% | 0.04773 |

Positive CE added is damage; native CE was3.26115. The shared-refit mean CE
increase has a document-bootstrap95% interval[0.04968,0.06796]. The registered
preservation screen required mean absolute token CE change<=0.05 and top1
agreement>=95%; **it fails**. Writer refitting improved shared-reader KL by9.81%,
just short of the separately registered10% threshold; that check also remains
failed. The local reconstruction gains therefore did not establish prediction
preservation. [Physical screen](../../PHYSICAL_QUADRATIC_V1_RESULT.json),
[compiled arithmetic controls](../../PHYSICAL_QUADRATIC_V1_COMPILE_CONTROL.json).

This is a negative result for these frozen replacements at these thresholds,
not evidence that the layer has no useful structure. The native prediction
metric differs from the fitted all-U quadratic metric. To red-team that
explanation, a separate native diagnostic is queued to split numerator and RMS
changes and test a probability-sensitive local error measure. No fitted rescue
or circuit adoption is being claimed while that diagnostic is pending.

For final residual $h$, replacement displacement $\delta h$, unembedding $U$
and $r=\sqrt{\|h\|^2/d+\epsilon}$, the unsaturated scores and their exact
first-order change are

$$
q=Uh/r,\qquad
\dot q=U\delta h/r-q\frac{h^\top\delta h}{\|h\|^2+d\epsilon}.
$$

The actual scores are $s=30\tanh(q/30)$, so

$$
\dot s=[1-\tanh^2(q/30)]\odot\dot q.
$$

For native probabilities $p=\operatorname{softmax}(s)$, the local quadratic
approximation to native-to-candidate KL is

$$
D_{\rm KL}(p\,\|\,p_{\rm changed})
\approx\frac12\left[\sum_t p_t\dot s_t^2
-\left(\sum_t p_t\dot s_t\right)^2\right].
$$

This accounts for RMS, saturation, and the relative probability of each token.
A common additive shift of the final scores changes no probabilities. Thus
uniform squared error in all unembedding coordinates is not the same fitting
objective. The derivative matched autodiff to$8.9\times10^{-16}$ on the CPU
control; at a0.001 perturbation dose, its KL approximation differed by0.151%.
These controls establish the formula, not its accuracy for the actual finite
replacement. [Metric controls](../../TERMINAL_PROBABILITY_METRIC_V1_CONTROL.json),
[registered native red team](../../TERMINAL_PROBABILITY_REDTEAM_V1_PREREGISTRATION.md).

Separately, the matched nonlinear optimizer comparison on the new Pile training
inputs is running. Both arms use the same frozen shared-reader initialization,
same objective and240seconds of fresh L-BFGS, comparing normal equations against
QR/direct residual evaluation. The normal-equation arm has completed: training
error0.01257, validation0.01306, **not converged**. QR is still running. No optimizer
advantage or convergence claim is made before that comparison finishes.
[Registered comparison](../../PILE_QR_REFINEMENT_V1_PREREGISTRATION.md).


## User correction: weight-first search, methods and bottlenecks — 21:53 UTC

**The main discovery route returns to the weights.** The model was trained only
on FineWeb, per the user. Pile is a shifted-corpus/OOD panel, not in-distribution
calibration data. Moreover, once a surrogate has been fitted on Pile training
rows, its performance on Pile validation is within-corpus generalization for
that surrogate, not a clean OOD transfer result. Earlier frozen, unadapted
functions and later Pile-adapted functions must remain distinguished.

I allocated too much of the recent work to data fitting, physical replacement
and probability diagnostics before adequately testing the weight-only structural
hypotheses. No further data/CE/Fisher fitting is queued. The corrected order is:
search different structures in the folded weights; use FineWeb and separately
labelled OOD data to validate them; incorporate data into discovery only after
that weight-first search has been substantially exhausted. Existing results are
preserved, not reinterpreted as completing the intended search.

### Actual coverage, rather than the size of the hypothesis list

| Representation | Assumption being tested | What has actually run |
|---|---|---|
| Shared bilinear products, a symmetric CP-like model | Many token interaction matrices reuse a small set of input products | Weight-only128-product fit, plus an explicit energy-penalized weight fit; only the latter converged locally |
| Shared input-reader dictionary | All products can be assembled from the same small input subspace | Weight-only64-reader/128-product fit; unconverged |
| Signed squares | Products may be more cleanly represented as squared projections with signed output coefficients | Implemented and toy-controlled; no native weight-only campaign result yet |
| Quadratic blocks | Several input interactions share a common output writer | Implemented; data-fitted32x8 blocks ran, but the weight-only case has not |
| Output-mode spectral relaxation | A few output patterns suffice even if each input function is an arbitrary quadratic | Exact weight-only spectrum/bounds ran; these are diagnostics, not an identified factorization |
| Sparse interaction core, general multi-output blocks, simultaneous block structure, hierarchical/coupled decompositions | Different kinds of shared computation beyond few rank-one terms | Mostly plans; not an executed broad comparison |

The four representations in the code are **not four advanced optimizers**. Most
fits use the same custom variable-projection scheme: solve output writers by
least squares, then optimize input factors with Adam/L-BFGS. The QR variant
improves the way the conditional solve is expressed, but is not a different
structural hypothesis. Its completed matched Pile comparison did not converge
or meet the1% validation-advantage threshold: normal0.013064 versus QR0.013039.
That failure does not settle how a structured Gauss–Newton solver would perform.

### Algorithms that fit the different structural hypotheses

- **Shared products / signed squares:** use symmetry-aware alternating least
  squares as a transparent baseline, then damped Gauss–Newton or
  Levenberg–Marquardt with structured matrix-vector products and preconditioning.
  These methods exploit the tensor least-squares problem rather than treating
  every parameter as an unrelated generic optimization variable. The square
  model ties the two input factors and cannot blindly reuse an untied-product
  update. [Structured damped Gauss–Newton](https://arxiv.org/abs/1205.2584),
  [matrix-free Gauss–Newton versus ALS](https://arxiv.org/abs/1910.12331).
- **Multiple interacting blocks:** block-term decomposition with block-specific
  input subspaces and multiple output coordinates, fitted with structured
  nonlinear least squares. Our current one-writer-per-block model is a narrow
  special case. [Block-term algorithms](https://www.tensorlab.com/doc/btd.html).
- **A shared basis with selected interaction edges:** Tucker-style factorization
  with an explicitly sparse core, fitted with alternating subspace/core updates
  and proximal or group-sparsity steps. The sparse core, not just a small Tucker
  rank, expresses which projected inputs interact. Unrestricted Tucker is a
  control, not automatically a circuit decomposition.
- **Separate or overlapping subsystems:** approximate joint block diagonalization
  of the token quadratic matrices, using an appropriate common change of input
  basis. Orthogonal-only methods impose a stronger hypothesis than a general
  invertible basis. [Non-orthogonal tensor/block diagonalization](https://arxiv.org/abs/1402.1673).
- **Shared structure across unembedding groups or multiple folded paths:**
  coupled factorizations with explicit shared factors and separate residuals.
  This is closer to the requested reuse/splitting question than decomposing each
  token independently. [Structured data fusion](https://tensorlab.net/doc/sdf-basic.html).

These are established algorithm families to implement or adapt and benchmark;
I have not yet run a broad state-of-the-art comparison. There is no justified
claim that one generic package or the newest paper is best for this particular
partially symmetric, signed, implicitly represented tensor.

### Assumptions that can hide structure

**A fixed small product count is a hypothesis, not a census.** A computation may
have a compact shared subspace or interaction graph but require many rank-one
terms. A single global64-dimensional dictionary may miss a union of different
local subspaces. Thirty-two blocks with one writer each also force output rank
at most32; the measured coefficient-space output spectrum already limits how
well that restricted class can fit, independently of its optimizer.

**Signs and non-orthogonality matter.** The native products are signed and need
not be independent or orthogonal. Nonnegative decompositions and orthogonal
block algorithms would add assumptions that have not been justified here.
Signed squares are legitimate, but changing products to squares alone does not
discover new information: a product equals a difference of two squares. What
changes is capacity, sharing and optimization geometry.

**The identifiable object may be a block or subspace.** Rotations, rescalings,
factor permutations and exchange of product operands can preserve the same
function. Individual-factor cosine matching is therefore too restrictive as the
only stability test. Conversely, good function reconstruction alone cannot name
or selectively manipulate a semantic circuit.

**Coefficient error is a discovery score, not the final success criterion.**
Weight-based structure remains worth finding even when its importance is uneven
on natural inputs. FineWeb validation and later OOD/intervention checks decide
whether a candidate supports the four requested properties; that does not make
data-fitting the default discovery algorithm.

### Actual bottlenecks and the concrete next implementation

The largest gaps are algorithm coverage and conditioning, not absence of a GPU.
Large cancelling components can flatten optimization, and one start or one
time-limited chunk is not enough to reject a representation. The full50304-token
tensor is about267GB in FP32 if naively materialized. Exact output-space reduction
and the native4608-product representation avoid that allocation; a dense Hessian
over all reader parameters is also unnecessary. Implicit contractions and
preconditioned linear solves are the appropriate computational tools.

As the first concrete return to weight-only solvers, I implemented the exact
linear normal operator for updating one product-reader family while holding the
other readers and output writers fixed. It works directly from L,R,D,U and small
Gram matrices, with no text inputs and no full tensor. A block-Jacobi-preconditioned
conjugate-gradient solve matched a dense direct solution to$3.1\times10^{-16}$
relative error on the control and converged in12 iterations. The independently
computed dense-tensor gradient agreed within$1.2\times10^{-13}$.
This is a tested **building block for weight-only ALS**, not a completed native
ALS benchmark or a solver for every hypothesis on the list.
[Control receipt](../../SYMMETRIC_PRODUCT_ALS_V1_CONTROL.json).

The completed probability diagnostic remains useful background: it reproduced
physical replacement damage, and the local Fisher expression predicted KL within
0.64–2.38%; normalization-only changes accounted for only0.67–1.04% of KL. It
does not supersede the user's weight-first priority or authorize another large
data-fitting branch before the structural search is exhausted.

A recent candidate is the2026 [NPDo tensor block-diagonalization method](https://arxiv.org/html/2605.12932v1). It optimizes blocks in orthonormal mode bases and provides convergence-to-stationarity results under its stated conditions. The orthogonality requirement is a substantive hypothesis for these weights; it is not a general solution for overlapping, non-orthogonal computational subspaces. It has not been implemented or run here.


## Weight-only solver benchmark and broader blocks — 22:08 UTC

**The first native ALS benchmark worked numerically but did not beat the earlier
optimizer. A broader shared-input block representation is now running.** No text
inputs enter either fit. FineWeb validation remains separate; no further Pile,
CE or Fisher fitting has been scheduled.

ALS alternates exact conditional updates of the first input readers, second
input readers and output writers. It minimizes the same explicit cancellation-
penalized weight objective, from the same saved initialization, as the earlier
Adam/L-BFGS fit. The linear subproblems use implicit tensor contractions and
preconditioned conjugate gradients, so the full token tensor is never allocated.

After120.25seconds,517outer sweeps and113698actual conjugate-gradient iterations,
the objective was0.91425633 versus the earlier locally converged0.91404573.
Raw coefficient capture was8.6771%, cancellation ratio1.213. The numerical
prediction passed: all conditional true residuals were below1e-9, no half-step
increased the objective, and checkpoint replay was exact. Convergence and the
registered reference-quality prediction both failed. Relative stationarity was
0.00332, above1e-4. Both methods inherit the same540second warm-start cost.
[Native receipt](../../WEIGHT_PRODUCT_ALS_V1_RESULT.json).

The negative-result check found continued objective improvement of2.03e-5 over
the last102sweeps, despite accurate conditional solves. Thus slower coupled
optimization remains a live explanation. This is not evidence that product
structure is absent, nor a general verdict on ALS or structured Gauss–Newton.
The latter has not yet been implemented for the native fit.
[Executed red-team audit](../../WEIGHT_PRODUCT_ALS_V1_REDTEAM.json).

The new block model represents the folded quadratic tensor as

$$
\widehat T_{vij}
=\sum_{g=1}^{16}\sum_{m=1}^{4}(Uw_{gm})_v
\left[E_g^{\mathsf T}C_{gm}E_g\right]_{ij}.
$$

Here each $E_g\in\mathbb R^{16\times1152}$ reads a16-dimensional input subspace;
its four symmetric matrices $C_{gm}\in\mathbb R^{16\times16}$ specify different
quadratic computations on those same readers. Each computation has its own
output vector $w_{gm}$. Blocks may overlap and need not be orthogonal. This
explicitly allows shared input computation with different output uses. A good
fit would suggest candidate blocks to examine, not identify semantic circuits.

This model has377344parameters:294912input coefficients,8704symmetric-core
coefficients and73728output coefficients. It has256input projections and2176
distinct within-block pair monomials, reused by the four cores. Its output rank
is at most64, so it still has a significant expressivity restriction. Exact
implicit Gram contractions and independently checked gradients passed dense
small-model controls. The native540second first chunk uses the existing
Adam/L-BFGS optimizer; this broadens the representation, not solver coverage.
[Control](../../MULTIOUTPUT_QUADRATIC_BLOCKS_V1_CONTROL.json),
[registered comparison](../../WEIGHT_STRUCTURAL_BASELINES_V1_PREREGISTRATION.md).

A256signed-square arm is also implemented and prepared, but has not been
submitted yet. It has589824parameters. Polarization gives

$$
(a^{\mathsf T}x)(b^{\mathsf T}x)
=\tfrac14\left[((a+b)^{\mathsf T}x)^2-((a-b)^{\mathsf T}x)^2\right].
$$

Thus128products can be expressed with256squares; allowing independent square
writers relaxes the paired-output constraint and changes capacity. A win would
need that qualification. The explicit energy penalty also acts on256square
features versus64block quadratics, so its granularity differs across families.
We report reconstruction and penalty separately. No positivity or sparse-token
assumption is imposed.

The immediate practical bottleneck is disk headroom for resumable optimizer
checkpoints, alongside incomplete algorithm and restart coverage. The managed
runner is healthy. Neither a time-limited first chunk nor one random start will
be labelled an exhausted structural hypothesis.


## Joint Gauss–Newton implemented and queued — 22:15 UTC

**A stronger weight-only optimizer is now implemented and queued for a native
benchmark.** The small exact checks pass; native convergence and fit quality
remain unmeasured. The multi-output block fit is still running and unconverged.
This adds solver coverage for products, not another structural hypothesis.

The difference from ALS is that a proposed step changes both sets of input
readers and the output writers together. Write one product component as

$$
F_j=z_j\otimes\operatorname{sym}(a_jb_j^{\mathsf T}).
$$

The first-order change contains all three coupled contributions:

$$
\delta F_j
=\delta z_j\otimes\operatorname{sym}(a_jb_j^{\mathsf T})
+z_j\otimes\operatorname{sym}(\delta a_jb_j^{\mathsf T})
+z_j\otimes\operatorname{sym}(a_j\delta b_j^{\mathsf T}).
$$

Here $z_j$ is the output writer in exactly transformed coordinates:
$z_j=Cw_j/\sqrt{E}$, where $C^{\mathsf T}C=U^{\mathsf T}U$ and $E$ is native
folded coefficient energy. This preserves the weight objective while avoiding
an additional output-metric conditioning problem in the linear solve.

Let $J$ map parameter changes to these tensor changes. We calculate $Jv$ and
$J^{\mathsf T}(Jv)$ through product contractions. Neither the full token tensor
nor the Jacobian or Hessian is stored. The explicit component-energy penalty
is included as additional residuals $\sqrt{\lambda}F_j$, so its curvature is
included too. This is a **Gauss–Newton approximation**, not the exact nonlinear
Hessian; no claim of global convergence follows.

The damped update solves approximately

$$
\left(J^{\mathsf T}J+\mu D\right)\delta=-J^{\mathsf T}r,
\qquad D=\operatorname{diag}(J^{\mathsf T}J),
$$

using preconditioned conjugate gradients. Damping limits risky steps. Each trial
then normalizes the readers and solves its output writers exactly. We accept
only actual objective improvement with a positive predicted improvement;
rejected steps increase damping. This is joint linearization followed by an
exact conditional writer projection. It avoids pretending that detaching an
optimal writer produces the full reduced Hessian.

The independent dense small-model check found relative errors below
$4.3\times10^{-16}$ for the normal-matrix action, its diagonal and the gradient.
The damped iterative solution agreed with a direct solve to
$4.7\times10^{-11}$. A separate near-planted fit decreased the penalized objective
from0.024885 to0.008871 in six accepted steps, with exact projection replay.
Its numerical-resolution stop is not counted as a convergence certificate.
[Jacobian control](../../SYMMETRIC_PRODUCT_GAUSS_NEWTON_V1_CONTROL.json),
[step acceptance control](../../SYMMETRIC_PRODUCT_LM_V1_CONTROL.json).

The native120second benchmark uses the same frozen initialization and objective
as the previous ALS and Adam/L-BFGS comparisons. It records actual accepted and
rejected steps, inner iterations, unfinished inner solves, gradient thresholds
and reference-quality predictions. The540second initialization cost remains
charged. [Preregistration](../../WEIGHT_PRODUCT_GN_V1_PREREGISTRATION.md).

The algorithmic source is the [matrix-free Gauss–Newton/ALS comparison](https://arxiv.org/abs/1910.12331).
Its implicit tensor contractions motivate this implementation. Our partial
symmetry, explicit component penalty and conditional writer projection are
adaptations checked separately; the paper's performance results do not prove
our native benchmark will succeed.


## What the blocks found: common output versus token contrasts — 22:26 UTC

**The block fit mostly models a shared vocabulary-wide output component. Its
remaining token-specific structure is much less well reconstructed.** This is
a weight-only finding that changes what to factor next; it is not a claim that
we have found independent circuits.

First, the two completed optimizer/representation screens:

| Fit | Time | Coefficient capture | Relative stationarity | Registered verdict |
|---|---:|---:|---:|---|
| Joint Gauss–Newton,128products |120.21s|8.6816%|0.001625|Numerical checks pass; convergence and reference-quality fail|
|16blocks,16readers and4outputs each|540.15s|8.6230%|0.009316|Numerical checks pass; convergence and capture-gain fail|

Gauss–Newton accepted305steps and rejected38, using19977inner iterations.
Its coordinate bridges hold and it slightly improves on the120second ALS fit,
but it remains behind the earlier converged penalizedL-BFGS result. Continued
improvement and unfinished gradients prevent any absence-of-structure claim.
[GN receipt](../../WEIGHT_PRODUCT_GN_V1_RESULT.json),
[GN red team](../../WEIGHT_PRODUCT_GN_V1_REDTEAM.json),
[block receipt](../../WEIGHT_STRUCTURAL_BASELINE_V1_multioutput_block_RESULT.json).

The block geometry is not completely collapsed. Each block's leading output
function accounts for81.4% of its energy on average, leaving other output uses;
11–14 of its16input axes are needed for90% of the input-mode energy. Mean
subspace overlap is4.88%, versus1.39% in an independent random geometry control.
The largest principal cosine is0.974. These compare fitted subspaces, not
semantic tasks or independent fitting restarts.
[Geometry audit](../../MULTIOUTPUT_BLOCK_GEOMETRY_V1.json).

However, the leading output functions across blocks are very similar: mean
absolute cosine0.963. Between80.3% and89.5% of each leading output axis is aligned
with the uniform vocabulary vector. Removing that uniform part for this
comparison lowers mean cosine to0.768. The extra output rows beyond the GPT-2
tokenizer vocabulary contribute under0.5% of each leading axis's energy, so
those rows do not explain the effect. Signed token lists include pronouns and
function words, but these are descriptions of weights, not task identifications.
[Canonical output modes](../../MULTIOUTPUT_BLOCK_OUTPUT_MODES_V1.json).

There is an exact decomposition that exposes this common channel. Define the
mean unembedding row and the centered unembedding by

$$
\bar u=\frac1V\sum_{v=1}^{V}U_{v:},\qquad
U_c=U-\mathbf1\bar u.
$$

For the quadratic part of the last bilinear layer,

$$
Q_{\mathrm{common}}
=\operatorname{sym}\!\left(L^{\mathsf T}
\operatorname{diag}(\bar uD)R\right),
$$

and its folded output separates exactly as

$$
U B_{\mathrm{quadratic}}(x)
=\mathbf1\,x^{\mathsf T}Q_{\mathrm{common}}x
+U_c B_{\mathrm{quadratic}}(x).
$$

The two output components are orthogonal in the coefficient-Frobenius metric,
so their squared energies and squared reconstruction errors add. The native
common channel accounts for7.195% of total folded coefficient energy; the
centered remainder accounts for92.805%. The block fit's relative squared error
is27.97% on the common channel but96.293% on the centered remainder. Equivalently,
it captures72.03% of the common channel and only3.707% of the remaining energy.
This explains why the overall8.623% capture can conceal weak progress on token
contrasts. These are coefficient scores, not probability or causal scores.

This common scalar quadratic has an exact signed eigendecomposition. Its
complexity is still substantial:440signed-square directions are needed to
capture90% of its coefficient energy. At128real products, the best possible
capture for this isolated scalar is73.964%. This is an attained spectral bound,
not an optimizer result. Each symmetric real product has at most one positive
and one negative eigenvalue, so128products can retain at most128eigendirections
of each sign. Selecting the largest of each sign is optimal in Frobenius norm;
pairing a positive direction and a negative direction constructs those products:

$$
p\,uu^{\mathsf T}-n\,vv^{\mathsf T}
=\operatorname{sym}\!\left[
(\sqrt p\,u+\sqrt n\,v)(\sqrt p\,u-\sqrt n\,v)^{\mathsf T}
\right].
$$

The constructed128-product approximation agrees with the spectral error bound
to$1.7\times10^{-16}$. Direct native and block polynomial evaluations agree
with the folded matrices to relative errors below$1.9\times10^{-15}$.
[Common-channel derivation and numerical receipt](../../COMMON_OUTPUT_QUADRATIC_V1_AUDIT.json).

**Preserve this channel; do not silently remove it.** It enters before the
native finalRMS andtanh, so a common shift there is not generally a harmless
shift of the final softmax scores. Bias and incoming residual contributions
also remain part of the full model. The useful next structural comparison is
an explicit common-channel component plus a separately factored token-contrast
remainder, with both components and their costs retained.

The256signed-square weight fit is now running. Its execution uses the same
registered objective, seed and optimization settings; a revised storage guard
uses the measured checkpoint size. No research artifacts were deleted. The
22:18hourly review is recorded; next reviews are23:18strategic and22:49mathematical.


## Signed squares finished; stopping mismatch identified — 22:28 UTC

The256signed-square fit stopped after369.75seconds and10015closures with9.6452%
coefficient capture, versus8.6989%for the earlier penalizedproductfit. It hasmore
parameters, so this is not a matched-capacity win. Numericalchecks passed;
convergence andthe registeredone-percentage-pointgain prediction bothfailed.
[Receipt](../../WEIGHT_STRUCTURAL_BASELINE_V2_square_RESULT.json).

The immediate CPUaudit found a concrete stopping mismatch: finalmaximumgradient
9.51e-11 is belowL-BFGS'sinternal1e-10tolerance, whileourrelative stationarity
0.0003955 exceeds1e-4. Rawreaderrow norms range53–2037, median67; the objective
uses normalizedreaders, so large rawscales can make absolutegradients small.
Repeatedidentical calls cannotfix that stopcondition. The nextpolish should
normalize the rawparameters without changingthe function and use a tighter
internal tolerance, preservingthe externalconvergencebars andoriginalfailure.
[Executed stopping audit](../../WEIGHT_SQUARE_STOPPING_V1_AUDIT.json).

No nativepolish hasrun yet. The managedsquarejob is complete; the stoppingaudit
is the concretecontinuation. The common-output/contrast findings above remain
unchanged andcontinue to motivate a different structuralfactorization next.


## Signed squares converged; sparse-core basis update prepared — 22:41 UTC

**The signed-square fit now converges locally. A new sparse-interaction method
has an exact fixed-basis baseline and a tested basis-update calculation.**
Neither result establishes the four circuit properties.

The stopping repair took17.26additional seconds and492closures, after the
original369.75second fit. All three new predictions passed. Final capture is
9.64522%; cancellation ratio0.99745. Relative stationarity is$3.30\times10^{-5}$
in unit-reader coordinates and$8.09\times10^{-5}$ when transported back to the
original per-row scales. Both pass the unchanged$10^{-4}$ threshold, alongside
the maximum-gradient and plateau checks. The function replay is exact.
[Polish receipt](../../WEIGHT_SQUARE_POLISH_V1_RESULT.json).

The original one-percentage-point improvement prediction remains failed:
9.645% versus8.699% is a smaller increase. The square representation also uses
589824parameters versus442368for128products. Local convergence removes one
uncertainty; it does not prove a global optimum, fair matched-capacity superiority
or stable semantic factors.

The new method represents the centered output tensor with an orthonormal input
frame $Q\in\mathbb R^{1152\times128}$ and a sparse set of quadratic edges.
Each edge joins two projected inputs, or squares one. In coefficient space the
normalized symmetric features are orthonormal:

$$
H_{ii}=q_iq_i^{\mathsf T},\qquad
H_{ij}=\frac{q_iq_j^{\mathsf T}+q_jq_i^{\mathsf T}}{\sqrt2}\quad(i<j).
$$

For a fixed frame, projecting the target tensor onto each $H_{ij}$ gives its
optimal output writer. Selecting the256largest writer energies is then the
exact best256-edge choice in that frame. No nonlinear optimization or hidden
ridge is needed for this conditional problem.

The initial frame comes from the top128eigenvectors of the centered tensor's
input marginal, which sums squared quadratic operators over output coordinates.
This is a spectral initialization: it optimizes one input-mode projection, not
the final sparse-core score. The native calculation used weights alone and took
5.80seconds on CPU. Independent dense controls and native trace/projection
bridges passed.
[Controls](../../SPARSE_ORTHOGONAL_CORE_V1_CONTROL.json),
[native baseline](../../SPARSE_ORTHOGONAL_CORE_V1_RESULT.json).

The256selected edges capture1.401% of centered coefficient energy; all8256edges
in the same frame capture6.406%. The selected graph uses44of128input readers,
with232off-diagonal edges; one input participates in33.9% of selected energy.
The registered5%capture and distributed-graph predictions both fail. These
numbers describe this initial frame, not an optimized sparse-core model.
The input marginal's top128energy is22.484%, an upper bound on the energy that
any128-dimensional input subspace can preserve in both slots. A single small
global dictionary is therefore a substantive restriction, even with better
optimization; larger or overlapping local subspaces remain distinct hypotheses.

Storage for the centered candidate is442368floating-point coefficients plus
512edge indices. The exact common channel is additional: a dense symmetric
1152-dimensional quadratic needs664128independent coefficients. Those costs
must be included when comparing a complete common-plus-contrast representation.
The compact centered artifact stores3.56MB of tensors; a larger original archive
was also retained. Only retired synthetic test fixtures were cleared for space.

The next update must optimize the **frame**, not only its spanned subspace.
Rotating a frame can change which edges are sparse even when the subspace stays
identical. In the control, such a rotation preserved full-core energy to
$3.9\times10^{-17}$ while changing the sparse-core score by0.00881.
For a Euclidean gradient $G$, the orthonormal-frame tangent projection is

$$
G_{\mathrm{tan}}=G-Q\operatorname{sym}(Q^{\mathsf T}G).
$$

This retains the within-subspace rotation component. A QR retraction preserves
orthogonality after a proposed step. Backtracking accepts sufficient ascent
on the old selected edges; reselecting the best edges can only improve that
score, so the old-support score supplies a valid lower bound for acceptance.
The finite-difference gradient check agrees to$2.5\times10^{-10}$ relative error;
the test update increases the true sparse score from0.09217to0.15066 while
preserving orthogonality. [Frame-update control](../../SPARSE_CORE_STIEFEL_V1_CONTROL.json).

This uses the orthogonality-constrained optimization framework described by
[Edelman, Arias and Smith](https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf).
Our top-edge selection makes the objective piecewise smooth, so smooth-manifold
convergence results do not automatically cover support changes. The update
calculation is implemented and checked; a native basis-optimization run and
restart stability remain pending. It would be premature to call the sparse-core
hypothesis exhausted from the fixed-frame baseline.


## Native sparse-frame optimization queued — 22:47 UTC

The native basis-learning experiment is now queued in the managed runner. It
uses Riemannian conjugate gradients: each step combines the current improvement
direction with a transported previous direction, restarting that combination
when it ceases to be an ascent direction. Orthogonality is preserved by the
checked QR update. Edges and output writers are reselected exactly after each
accepted step. The common-output channel remains separate and preserved.

The small recovery control improves capture from27.5%to97.9%in40steps; it is
not at a certified global optimum. Two20stepchunks produce exactly the same
frame and score as one40step run, checking resumable state.
[Optimizer control](../../SPARSE_CORE_RCG_V1_CONTROL.json).

The first nativechunk is240seconds, with128readers and256edges. It must pass
initial/final replay and monotonicity checks; convergence requires both a
plateau and a small tangent gradient, with a nonzero gap between retained and
omitted edge energies. The registered capture target is5%of centered coefficient
energy. A time limit is unfinished optimization, not evidence against a sparse
interaction representation. Native results are pending.
[Preregistration](../../SPARSE_CORE_RCG_V1_PREREGISTRATION.md).


## Sparse frame converged; what the math review establishes — 23:00 UTC

**The sparse-frame optimizer converged locally in74.30seconds, improving
centered coefficient capture from1.401%to3.6695%.** Numerical and convergence
checks passed; the registered5%capture target failed. It took865iterations,
with no score decreases. Final relative stationarity is$8.49\times10^{-5}$,
and the gap between retained and omitted edges is positive. This is a real
optimization improvement within the orthogonal sparse-core family, not a proof
that the family is sufficient or globally optimized.
[Native result](../../SPARSE_CORE_RCG_V1_RESULT.json).

The [22:49 mathematical review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_2249.md)
produced two concrete checks.

**First: can the fitted square factors be recovered from their joint tensor?**
Yes, under the numerical rank conditions observed here. Two independent input
factor matrices and distinguishable output writers meet a classical tensor
uniqueness condition. In the square model the two input factors coincide, but
each still has256independent columns. Three fixed pairs of random output mixtures
recovered every fitted reader with cosine at least0.9999999999999976, given the
fitted input span. This avoids fitting each token slice independently.
[Spectral recovery audit](../../SQUARE_PENCIL_IDENTIFIABILITY_V1_AUDIT.json).

That is uniqueness of this exact fitted tensor. It does not show that another
good approximation to native weights will be the same. Native projected slices
differ substantially from the fitted slices and do not satisfy the same exact
real diagonalization pattern. Multiple starts and perturbation tests are still
needed. Signed squares, distinct-reader products and multi-output blocks also
have different identifiability conditions; the theorem is not transferred blindly.

**Second: do graph components describe isolated native computations?**
For an input projector $P=EE^{\mathsf T}$, native quadratic interactions can be
split into within-span, mixed and outside terms. The mixed energy is

$$
2\left[\operatorname{tr}(E^{\mathsf T}SE)
-\sum_v\|E^{\mathsf T}Q_vE\|_F^2\right],
\qquad S=\sum_v Q_v^2.
$$

For the learned128input frame, the native centered tensor has5.88%inside,
24.14%mixed and69.98%outside energy. Restricting to the92readers actually used by
the fitted graph still leaves16.56%of total native energy in mixed interactions.
The surrogate graph has disconnected components, but its active union has a
substantial native boundary. Extraction must describe these input/output
connections; graph sparsity alone does not make a closed circuit.
[Boundary calculation](../../SPARSE_CORE_NATIVE_PORTS_V1_AUDIT.json).

These checks improve what we can claim and what to test next. They neither name
semantic tasks nor satisfy OOD prediction, extraction, selective removal and
composition. An open component with explicit interfaces can still be useful;
nonzero mixed energy is not a rejection of all circuit decompositions.

An independently initialized sparse-frame run is now queued: Gaussian frame,
seed937, same128readers,256edges, centered weight metric and convergence bars.
Its initial capture is0.0482%, versus1.401%for the spectral initialization. The
registered comparison asks for at least99%of the previous converged fit quality
and full coefficient-function cosine at least0.95. Functions are compared before
raw factor names, so signs, permutations and unused reader directions do not
create a false disagreement. This is the first independent restart in this
family; native results are pending.
[Restart preregistration](../../SPARSE_CORE_RCG_RESTART_V1_PREREGISTRATION.md).


## Restart disagrees; broader block conditioning repaired — 23:06 UTC

The independent sparse-frame restart also converged locally, in 83.42 seconds.
It captured 3.2859% of centered coefficient energy versus 3.6695% for the first
fit. Their full-function cosine is 0.7955, below the registered 0.95 bar, and
the second fit retains only 89.55% of the first fit's capture, below the 99% bar.
Numerical and convergence checks passed; both reproducibility clauses failed.
[Restart result](../../SPARSE_CORE_RCG_RESTART_V1_RESULT.json).

This is initialization dependence among different-quality local fits. It does
not prove multiple equally good global optima or the absence of a useful sparse
representation. The exact best scalar combination of these two fitted functions
captures 3.8967%, only a 0.2272-percentage-point improvement over the first.
That combination approximately doubles centered-model storage before any
deduplication. It is a diagnostic of complementary fit, not an adopted model or
a claim that each difference is a semantic computation.
[Two-function span calculation](../../SPARSE_CORE_TWO_FUNCTION_SPAN_V1_AUDIT.json).

The broader overlapping block model remains unconverged. I have now executed
a conditioning repair for that representation: orthonormalize the readers
inside each block while transferring their coordinate change into its quadratic
cores. This preserves the represented function; it does not force different
blocks to be orthogonal or disjoint. Core normalization is offset in its writer.
On the saved native fit, function replay agrees to relative error 2.53e-15 and
the component-energy penalty agrees to 2.22e-16.
[Gauge repair control](../../MULTIOUTPUT_BLOCK_ORTHOGONAL_GAUGE_V1_CONTROL.json).

The transformed initial state is saved. The next step is a manifold-aware
optimizer for this broader family, first preserving its original full-U objective
and penalty. No newly optimized block fit has run yet.


## Overlapping-block optimizer now running — 23:20 UTC

The broader block model now has a checked manifold optimizer and a native
weight-only run. It keeps the previous representation, coefficient norm and
explicit penalty, so this tests whether better coordinates resolve its
unfinished optimization. No token data or labels enter this step.

For block $g$, the quadratic features are

$$
\phi_{gm}(x)=(E_gx)^\top C_{gm}(E_gx),\qquad
E_gE_g^\top=I,\qquad C_{gm}=C_{gm}^\top.
$$

The rows of $E_g$ define a shared input subspace. Each symmetric $C_{gm}$ specifies
one quadratic interaction within it; each feature has its own output writer.
The QR coordinate change absorbs scaling and internal changes of basis into
these dense cores. It adds no orthogonality restriction between different blocks.
Normalizing each core is offset in the corresponding writer.

Let $G$ be the Gram matrix of the64 quadratic features and $K$ their inner
products with the native residual-output coefficient tensor. The exact
conditional writer solve is

$$
W=K\bigl(G+\lambda\operatorname{diag}(G)\bigr)^{-1},
\qquad \lambda=0.01.
$$

Here $K$ has residual-output rows; the full unembedding metric is used in the
objective. The same metric weights the explicit component-energy penalty, so
it cancels in this conditional solve. Since the frame rows are orthonormal and
the cores have unit Frobenius norm, $G_{jj}=1$ and the regularized Gram condition
is at most6401. That bound prevents a particular linear-solve degeneracy; it
is not a guarantee of fast or global nonlinear optimization.

The remaining variables are optimized jointly by Riemannian conjugate gradient:
project gradients onto the allowed frame/core directions, take a descent step,
and restore the constraints by QR and normalization. A backtracking line search
requires an actual decrease of the full penalized objective. The writer is
solved anew at every evaluation. We test projected gradient size as well as
loss flattening; merely stopping at the time limit is not convergence.

CPU checks passed: finite directional derivative error2.59e-11, tangent constraint
errors below9e-16, and exactly identical40-step versus20+20 resumed parameters.
A small random problem decreased its objective by0.769. These validate the
implementation; they are not native-model results. The managed240-second run
started23:18:59; no completed native result is claimed here.
[Registered predictions](../../MULTIOUTPUT_MANIFOLD_V1_PREREGISTRATION.md),
[CPU controls](../../MULTIOUTPUT_MANIFOLD_V1_CONTROL.json).

The23:18 hourly review is complete. The remaining identification bottlenecks
are local minima, common-output dominance, and whether fitted subcomputations
correspond to stable native functions with explicit interfaces. The earlier
22:49 math review's uniqueness result applies to the exact fitted square tensor,
not to the native tensor or the best approximation problem.
[Hourly review](../../HOURLY_STRATEGIC_REVIEW_2026-09-10_2318.md).


## Block result and overlapping token factors — 23:28 UTC

The overlapping-block manifold run finished240seconds without convergence.
Numerical checks and the registered improvement/retention prediction passed;
convergence failed. Penalized objective improved by5.30e-5 and raw full-tensor
capture rose from8.6230% to8.6285%. The final projected stationarity was0.00368
against the1e-4 bar. This is not a structural negative.

The last500steps still improved the objective by1.22e-6; gradients oscillated,
with best stationarity1.78e-4 across the entire run. No accepted objective
increase or ill-conditioned writer solve occurred. Removing the internal
coordinate freedom helped the representation stay numerically controlled,
while coupled nonconvex optimization remains unfinished. The checkpoint is
resumable. Its final function captures71.98%of the common-output component but
only3.7167%of the centered component, so the earlier common-output story remains.
[Result](../../MULTIOUTPUT_MANIFOLD_V1_RESULT.json),
[negative-result audit](../../MULTIOUTPUT_MANIFOLD_V1_REDTEAM.json),
[common/centered accounting](../../MULTIOUTPUT_MANIFOLD_V1_OUTPUT_SPLIT.json).

The next native job tests a genuinely different structural assumption:
**sparse overlapping token usage of shared quadratic functions**. The MLP17
dossier already records a failed16-leaf whole-token mean hierarchy. Here a token
can use several functions and each function can serve many tokens; there are
no hard token clusters. This targets shared summands rather than proportional
whole-token functions.

First take the centered coefficient tensor's optimal128-dimensional output
projection, computed exactly through its small output covariance. Write

$$
T_{128}(t,i,j)=\sum_{a=1}^{128} A_{ta}H_{aij},
\qquad \langle H_a,H_b\rangle_F=\delta_{ab}.
$$

The quadratic functions $H_a$ are initially unrestricted inside input space;
we have not assumed one product per function. $A_{ta}$ describes how much token
$t$ uses function$a$. Any orthogonal rotation$R$ permits

$$
A'=AR,\qquad H'=R^\top H,\qquad A'H'=AH.
$$

This preserves the projected tensor and its fit error. We choose a rotation
that maximizes the varimax criterion

$$
\mathcal V(R)=\sum_a\left[
\frac{1}{V}\sum_t (AR)_{ta}^4
-\left(\frac{1}{V}\sum_t (AR)_{ta}^2\right)^2\right].
$$

It favors concentrated signed loadings: a factor used strongly by some tokens
and weakly by others. The first experiment uses raw loadings with one global
numerical rescaling, not row normalization or token frequency weights. This
can favor large-weight rows, so we report that limitation and the padded-row
energy. Orthogonal function bases also exclude oblique/overcomplete dictionaries.

The [modern statistical analysis of varimax](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Rohe23.pdf)
provides recovery results under specified latent-factor distributions, including
leptokurtic assumptions. We have not established those conditions for learned
vocabulary weights, so no such recovery guarantee is claimed here.

CPU controls passed: the implicit projection matches dense tensor SVD to1.50e-15;
joint rotation preserves the tensor to3.91e-16; the gradient matches a finite
difference to7.75e-10. A planted overlapping sparse factor mixture is recovered
with minimum axis alignment0.99994 and local convergence.20 uninterrupted steps
match10+10 resumed steps exactly. Diagnostic cadence was repaired before native
registration so rapidly converging controls do not outrun the plateau history.

The native run has now completed in62.24seconds and converged locally. Numerical
and convergence predictions passed; the structural prediction failed overall.
Median effective factor count fell from38.18 to28.62, meeting the25%reduction
clause. But each token's strongest4factors retained only32.39%of loading energy,
below the50%bar (initial25.26%). The fixed rank128 projection captures28.999%of
centered native coefficient energy; rotation leaves this unchanged. Effective
factor count is $(\sum_a A_{ta}^2)^2/\sum_a A_{ta}^4$, not a literal nonzero count.
[Result](../../OUTPUT_VARIMAX_V1_RESULT.json).

The executed red-team audit finds aggregate top8/top16/top32/top64 retention of
43.87%/59.21%/77.56%/94.19%. The median token needs53factors for90%of its own
projected loading energy (10th–90thpercentiles44–60). Keeping only four loadings
per token captures9.392%of the full centered native tensor, because the input
functions are orthonormal and the outside-projection residual is orthogonal.
This exact accounting does not make the resulting program cheap: the global
quadratic dictionary and remainder still cost storage and computation.

The four functions with highest output loading energy need461–476signed-square
directions for90%coefficient energy. Their best single real product captures
8.95–13.60%. These are exact spectral statements about those four scalar forms,
not lower bounds on shared input computations across functions. Sparse token
usage did not automatically make the input computation simple.
[Executed function/sparsity audit](../../OUTPUT_VARIMAX_V1_AUDIT.json).

The strongest remaining assumptions are the fixed128-dimensional output
subspace, orthogonal function axes, one rotation initialization and raw loading
norm weighting. A specific probe found the raw solution far from stationary
for the equal-token criterion (relative gradient0.613). Accordingly, an equal-row
normalized varimax comparison is now queued. Only the rotation objective uses
unit-norm token rows; the original tensor is preserved with unnormalized
loadings and inverse-rotated functions. Report both raw-energy and equal-token
metrics under both rotations. This is a change in the structural prior, not
proof that the new fit will be better, and it does not rewrite V1's failed bar.
[Next preregistration](../../OUTPUT_VARIMAX_NORMALIZED_V1_PREREGISTRATION.md).
No data or circuit identification is claimed. Common output and the outside
projection remainder remain explicit.


## Attention pullback and shared-source constraints — 23:48 UTC

**Equal-token weighting did not resolve the dense token usage.** Its varimax fit
converged in147.82seconds. Median effective factors per token is27.61, compared
with28.62for raw varimax. Equal-token top4energy rises only31.54%→32.06%, still
below50%. Raw-energy top4retention is32.64%. Both numerical and convergence
predictions held; the structural bar failed. Matching quadratic functions
across these different objectives gives median absolute cosine0.894; this is
not an independent-start stability test. The weighting hypothesis was tested
and had modest impact; fixed subspace and orthogonal dictionaries remain limits.
[Result](../../OUTPUT_VARIMAX_NORMALIZED_V1_RESULT.json),
[red-team comparison](../../OUTPUT_VARIMAX_NORMALIZED_V1_REDTEAM.json).

**Folding through attention17's output map alone also produced only small gains.**
This is an actual terminal-path weight fold, unlike the earlier separate QK
experiment. With $b$ the residual before attention output, $z$ the concatenated
head outputs and $O$ the attention output matrix, the MLP-input numerator splits
exactly into

$$
b^\top Q_vb+2b^\top Q_vOz+z^\top O^\top Q_vOz.
$$

The native input RMS squared divides this expression; the bias, residual,
final RMS and tanh remain explicit. $z$ still includes routing and values.
We have not folded the whole computation into a token-to-logit polynomial.

| Weight-coordinate description | Top128 centered output capture | Within-head quadratic energy |
|---|---:|---:|
| Original MLP input | 29.00% | Not assigned attention heads |
| Native attention output map $O$ | 30.54% | 11.86% |
| Left-rotated $PO$, same singular values and head Gram | 29.18% | 11.36% |
| Right-rotated $OP$, same pulled-back spectrum | 30.54% | 11.25% |

All exact checks passed; both structure predictions failed. The learned
alignment gives a small gain, and the native head partition is close to the
scrambled control. This does not reject optimized cross-head groupings or the
full routing/value fold. The apparently huge3833×attention-port coefficient
energy is mostly coordinate gain: after normalizing the output map's overall
scale, the ratio is0.981. It is not evidence that attention dominates behavior.
[Native result](../../ATTENTION_OUTPUT_PULLBACK_V1_RESULT.json),
[red-team scale/alignment audit](../../ATTENTION_OUTPUT_PULLBACK_V1_REDTEAM.json).

The next algebraic step addresses a limitation of treating head outputs as
independent variables. At one source position, every head reads the same
source-state tuple through its own value matrix. If $a_{hp}$ is the head's
routing weight to source$p$ and $x_p$ its shared normalized value-source tuple,
then the lifted pre-value input is

$$
Z_{hi}=\sum_p a_{hp}x_{pi}.
$$

For one source,$Z=ax^\top$ has rank1. Some quadratic coefficient directions
therefore vanish on all such inputs, although they count in an unrestricted
coefficient norm. Crucially, they need not vanish with multiple sources:

$$
Z_{hi}Z_{kj}-Z_{hj}Z_{ki}
=\sum_{p<q}(a_{hp}a_{kq}-a_{hq}a_{kp})
(x_{pi}x_{qj}-x_{pj}x_{qi}).
$$

This separates routing contrasts from source-value contrasts. It supplies a
possible shared arithmetic structure; it does not permit deleting cross-source
interactions. The controlled two-source determinant example equals1, despite
each individual source having zero determinant.

The symmetry split, native-shaped value mixture, one/multiple-source replay and
an efficient128×128-block energy contraction are implemented and tested to
numerical precision. Native coefficient mass in these channels has not yet been
measured. The next weight-only measurement is specified for the four already
fixed leading quadratic functions, with shared-source-coordinate controls.
[Full derivation and executable consequence](../../SHARED_SOURCE_ATTENTION_QUADRATIC_V1_MATH.md),
[CPU controls](../../SHARED_SOURCE_ATTENTION_QUADRATIC_V1_CONTROL.json).
This advances the fuller attention fold; QK normalization, live routing,
residual interfaces and the four circuit properties remain unresolved.

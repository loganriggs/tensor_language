# Unsupervised structure campaign — 10 September, updated 21:28 UTC

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

# Unsupervised structure campaign — 10 September, updated 20:44 UTC

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
| 3 | Many products reuse a small reader dictionary | $a_j=E\alpha_j$, $b_j=E\beta_j$ | Weight chunk complete but unconverged; data counterpart prepared |
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

**The weight fit is numerically awkward, and natural-input structure looks considerably simpler than its coefficient error suggests.** Neither of the two completed weight chunks converged. Their saved functions nevertheless explain roughly 91–92% of the unembedding-weighted MLP output energy on validation states. The natural-state product fit is still running; its training error is lower than a larger affine baseline, but its validation result is not yet available. These are candidate representations, not identified circuits.

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

On the frozen stalled readers, $\lambda=0.01$ lowered the cancellation ratio from 14,265 to **1.245**, while retaining about **80% of the coefficient energy captured by the unregularized fit**. Absolute captured energy fell from 8.74% to 6.98% of the native tensor. No readers were refitted in this probe. The new joint-fit objective is implemented; dense-function and gradient controls passed for all four representations and both metrics. Joint optimization with this penalty is still pending. [Fixed-reader path](../../STRUCTURED_FIT_FIXED_READER_PENALTY_V1_AUDIT.json), [objective controls](../../ENERGY_REGULARIZED_QUADRATIC_V1_CONTROL.json).

### Natural-state baselines and validation

All entries below use the same unembedding-weighted bilinear MLP contribution, with the fixed native output bias subtracted and before final residual normalization and score saturation. Lower squared relative error is better. The affine model was fitted on the training rows only; test rows remain unopened.

| Representation | Stored scalar parameters | Training error in this metric | Validation error |
|---|---:|---:|---:|
| Constant mean output | 1,152 | 0.29557 | 0.29653 |
| Affine map | 1,328,256 | 0.02932 | 0.03156 |
| 128 products fitted to coefficients; unconverged | 442,368 | Not evaluated | 0.09341 |
| 64 shared readers / 128 products fitted to coefficients; unconverged | 237,568 | Not evaluated | 0.08208 |
| 128 products fitted to natural states; running | 442,368 | About 0.017 at 350 Adam steps | Pending checkpoint |

The shared-reader fit has worse coefficient reconstruction than the free-product fit but better natural-state validation error in these first starts. This demonstrates that the two metrics can rank partial fits differently. Different initializations and nonconvergence prevent a final method comparison. The natural-state product fit's early training result also exceeds a constant or affine explanation; it still needs validation, stability and the four circuit-property tests. [Baselines](../../NATURAL_STATE_BASELINES_V1_AUDIT.json), [weight-product validation](../../STRUCTURED_FIT_V1_weight_product_s0_CHUNK_00_VALIDATION.json), [shared-reader validation](../../STRUCTURED_FIT_V1_weight_shared_reader_s0_CHUNK_00_VALIDATION.json).

The original data-fit runner failed before optimization because it requested `Down.bias`; the native parameter is `Down_bias`. V2 corrects the key and passed a 64-state native-output replay check at relative error $1.79\times10^{-4}$ against a $10^{-3}$ bar. The failure remains preserved, and no data were recaptured. [Repair and scope](../../STRUCTURED_FIT_DATA_V2_REPAIR.md).

## Execution ledger

- Completed: short joint32 fit, stable-product causal screen, fixed-support feasibility, shared-reader block counterexample. Their failures remain intact.
- Completed: managed 1,000-sequence capture V2; all 64,000 state pairs saved, rounding errors below 0.016%. V1 direct-FP16 storage overflow is preserved.
- Implemented and checked: four representations under weight and data objectives, with four starts each (32 configurations); resumable Adam/L-BFGS optimization and explicit stationarity/plateau gates.
- Completed chunks, not converged fits: weight-product native start and weight shared-reader random start. Both checkpoints replay; neither passed stationarity. CPU stall, regrouping, regularization-path and validation checks are complete.
- Running since20:39:01: repaired V2 natural-state product fit. V2 natural-state block fit is queued; the data shared-reader counterpart is prepared next. Other configurations remain pending, not completed.
- Implemented and controlled, but no joint native fit yet: explicit component-energy regularization. Convergence tracking must use its penalized objective while reporting reconstruction separately.
- Pending: joint decomposition of the longer unembedding → last bilinear → last attention path. The earlier position-corrected QK work was a separate two-behavior experiment.

# Project proposal: discover circuits by decomposing composed interaction paths

Requested by Logan, 11 September 2026. **Status: proposal, with existing evidence and two new exact accounting checks; the proposed joint sparse fits have not run.**

## High-level proposal

Instead of first finding a good decomposition of each layer and then connecting the pieces, **compose several adjacent computations and search for a simple representation of the resulting interactions**. The starting objects are the unembedding folded through the last bilinear layer, then through attention output/value maps or the preceding bilinear layer. Crucially, include interactions between those incoming contributions.

The intuition is sensible: a layer might write many directions, while the downstream computation reads only particular combinations. Composing their weights can expose cancellations, shared factors, or small interacting subspaces that separate layer fits obscure. This is a hypothesis about this model, not a guarantee that deeper composition becomes simpler.

My recommendation is **a joint sparse interaction model with shared readers and writers, compared against block-term and explicitly shared arithmetic-graph models**. Sparse Tucker is a useful first baseline. It should not be a prerequisite for discovering hierarchy: we should also search for reused intermediate computations during fitting, and allow a dense small block when it describes a simple operation more economically than many sparse scalar entries.

The deliverable is an executable collection of interacting components with specified inputs, operations, outputs and shared dependencies. To count as circuits, they must predict unseen behavior, support extraction at a declared boundary, permit selective removal, and compose or reuse correctly. A sparse tensor alone does not satisfy those requirements.

**First experiment:** compare separate versus joint decompositions of the residual/residual, residual/attention and attention/attention operators, using the full unembedding objective and identical total implementation budgets. Follow with the preceding-bilinear and mixed attention/bilinear paths. Keep normalization explicit from the beginning. Discover from weights; freeze candidates before text validation.

## What has already been done—and what is new

We have done parts of the proposed algebra. We have not established that the joint composed-path representation is sparse or recovered its circuits.

| Existing work | What it establishes | What remains new here |
|---|---|---|
| [September 10 backward-folding derivation](../2026-09-10/unembedding_folding_in_math.md) | Derives residual/attention cross terms, attention-head pairs, and the quartic term from two bilinear layers. | A systematic joint decomposition and comparison of these objects. |
| [Full-unembedding attention-output study](../../ATTENTION_OUTPUT_PULLBACK_V1_PREREGISTRATION.md), [result](../../ATTENTION_OUTPUT_PULLBACK_V1_RESULT.json), [red-team](../../ATTENTION_OUTPUT_PULLBACK_V1_REDTEAM.json) | Computes full centered-unembedding coefficient spectra and head-pair energies after folding through the last attention output map. | Optimized cross-head sparse cores, shared factors across path blocks, and explicit QK/value composition. |
| [MLP16 producer study](../../PARENT1_MLP16_PRODUCER_V1_MATH.md) | Folds three downstream readers into MLP16; those fixed producer functions are not simple low-rank quadratics. | Joint optimization of downstream combinations and the composed function, rather than requiring each fixed reader to simplify separately. |
| [Latest research update](research_update_2026-09-11_2142.md) | Covers sparse dictionaries, LL1, shared-reader graphs and a frozen local suffix predictor, including their misses. | Applying such assumptions to the larger composed interaction object. |
| [New normalization accounting](../../NATIVE_SUFFIX_INPUT_NORMALIZATION_V1.json) | Separates bilinear numerator interaction from input-RMS correction for the existing conditional suffix component. | Discovery and validation of a new sparse path program. |
| [New six-path oracle](../../NATIVE_SIX_PATH_ORACLE_V1.json), [code](../../native_six_path_oracle_v1.py) | Executes the exact six source-pair expansion for the entire native MLP17 output, checked on 128 cached endpoints. | A general coefficient-contraction optimizer and sparse fitted representation. |

The earlier attention-output study increased top-128 output coefficient capture from **29.00% to 30.54%**, missing its registered improvement bar. Only **11.86%** of attention/attention coefficient energy was inside individual head blocks, versus **11.25%** after scrambling head coordinates. This rejects that particular strong head-concentration assumption. It does **not** test optimized interacting subspaces or the full routing/value computation.

The new six-path check reconstructs the native quadratic output with **5.23 × 10⁻¹⁵ relative error**; mixed-pair finite-difference checks are within **2.85 × 10⁻¹⁴**. These are algebra/execution checks, not sparsity or behavioral results. The previous report remains the latest retrospective; this document is the new proposal.

## 1. The exact final-layer object

Use column vectors. The residual width is $d=1152$, native bilinear width is $m=4608$, and the unembedding has $V=50304$ rows. For the last bilinear layer,

$$
B_{17}(x)=D_{17}\big[(L_{17}x)\odot(R_{17}x)\big]+b_{17}.
$$

Here $L_{17},R_{17}\in\mathbb R^{m\times d}$ read two sets of linear features, $\odot$ multiplies corresponding features, and $D_{17}\in\mathbb R^{d\times m}$ writes their products into the residual stream.

Let $w_v=U^\top e_v$ be the unembedding reader for vocabulary row $v$. The symmetric quadratic matrix for its bilinear numerator is

$$
S_v=\frac12\left[
L_{17}^\top\operatorname{diag}(D_{17}^\top w_v)R_{17}
+R_{17}^\top\operatorname{diag}(D_{17}^\top w_v)L_{17}
\right].
$$

Only the symmetric part matters because the same vector occupies both input slots. Before native input normalization, the scalar numerator is $x^\top S_vx$. Stacking the $S_v$ gives the familiar third-order tensor, indexed by output token and two residual coordinates.

The actual last-layer input is a sum of sources. Write

$$
y=g+p+a,
\qquad
p=\lambda_{17,0}B_{16}(x_{16}),
\qquad
\rho(y)^2=\frac{\|y\|^2}{d}+\epsilon.
$$

Here $p$ is the scaled preceding-MLP output, $g$ is the remaining incoming residual including re-entry, and $a$ is the final attention output. The previous MLP input $x_{16}$ has its own native normalization. Attention itself depends on the incoming residual; these are explicit computational ports, not independent random variables.

The full native final state and logits are

$$
h=y+B_{17}\!\left(\frac{y}{\rho(y)}\right),
\qquad
\ell_v=30\tanh\!\left(\frac{w_v^\top h}{30\rho(h)}\right).
$$

Thus the contribution $y^\top S_vy/\rho(y)^2$ is only the MLP numerator route through the final readout. Direct residual writing, MLP bias, final normalization and logit capping remain in the executable model.

## 2. Interaction paths are pairs of incoming branches

Expanding the quadratic gives exactly six unordered pairs:

$$
\begin{aligned}
y^\top S_vy={}&g^\top S_vg+p^\top S_vp+a^\top S_va\\
&+2g^\top S_vp+2g^\top S_va+2p^\top S_va.
\end{aligned}
$$

The three mixed terms answer the user's question about “the interactions.” Their two orderings are already combined by the principled symmetrization. They should not be counted twice as two independent circuit discoveries.

A useful candidate could therefore be “information produced by MLP16 interacts with information retrieved by attention17, and the product writes to a family of output directions.” Neither native module alone would be that circuit.

More generally, if source $i$ writes $E_i z_i$, define

$$
K_{v,ij}=E_i^\top S_vE_j.
$$

For $i<j$, its contribution is $2z_i^\top K_{v,ij}z_j$; the reverse block is $K_{v,ji}=K_{v,ij}^\top$. A rectangular mixed block should **not** be symmetrized within itself when its two ports have different dimensions or meanings.

This suggests an interaction graph whose nodes are learned source features and whose edges are bilinear operations. Further upstream substitution turns it into a directed acyclic graph (DAG): intermediate results can feed several later operations. An interaction edge and a complete input-to-output circuit are different levels of description.

## 3. How far attention can be folded

First write $a_t=\sum_h O_h z_{h,t}$. Folding through the output projection gives

$$
K_{v,hk}=O_h^\top S_vO_k,
\qquad
K_{v,gh}=S_vO_h,
\qquad
K_{v,ph}=\lambda_{17,0}D_{16}^\top S_vO_h
$$

for attention pairs, residual/attention pairs, and the product-coordinate portion of the previous-MLP/attention pair. Bias contributions remain separate or can be represented by a constant coordinate.

**A limit on the output-only step:** if the concatenated output map $O$ is invertible, replacing $S_v$ by $O^\top S_vO$ is only an invertible change of input coordinates. It cannot lower exact tensor CP rank or multilinear ranks. Approximate spectra, coordinate sparsity and the cost of particular constrained implementations can change, but raw gains may reflect rescaling. Deeper value/routing composition, upstream reachability constraints, and shared computation are the substantive opportunities. Include alignment and coordinate-gain controls instead of treating any folded-spectrum gain as a discovery.

This is only the first depth. The native head value aggregate is

$$
z_{h,t}=\sum_{s\le t}\gamma_{h,ts}
\left[(1-\mu_h)V_h\bar r_s+\mu_h v^{(0)}_{h,s}\right],
$$

where $\bar r_s$ is the native normalized residual and $v^{(0)}$ is the separately produced first-layer value stream. The learned mixing coefficient can be negative. This model has no softmax or attention-row normalization. Its routing coefficient is the product of two QK scores:

$$
\gamma_{h,ts}=
\left(\frac{\widehat q_{1,h,t}^\top\widehat k_{1,h,s}}{128}\right)
\left(\frac{\widehat q_{2,h,t}^\top\widehat k_{2,h,s}}{128}\right).
$$

Hats denote the model's native Q/K normalization and rounded rotary-position operations. They must be reproduced, not replaced with unnormalized matrices.

Substituting this expression exposes interactions indexed by head **and source position**. For example, the current-value part of the attention-pair route contains

$$
\sum_{h,k}\sum_{s,u\le t}
\gamma_{h,ts}\gamma_{k,tu}(1-\mu_h)(1-\mu_k)
\bar r_s^\top V_h^\top O_h^\top S_vO_kV_k\bar r_u.
$$

There are also current/first-value and first-value/first-value terms. Keeping only $s=u$ or only $h=k$ would impose an unsupported independence assumption.

We should compare three depths: output projection only; output plus value maps with live routing as an explicit input; then routing-aware composition that retains **both QK factors together**. For the last depth, the numerator of each QK score can be expanded through MLP16 as in the original note, while all normalization factors and positional operators remain explicit. This is not a globally polynomial transformer once those denominators are included.

## 4. Folding into the preceding bilinear layer

Define its product coordinates

$$
z(x)=(L_{16}x)\odot(R_{16}x),
\qquad
p=A z(x)+b_p,
\quad A=\lambda_{17,0}D_{16}.
$$

Absorb $b_p$ into $\widetilde g=g+b_p$. The composed numerator is

$$
\begin{aligned}
(\widetilde g+Az+a)^\top S_v(\widetilde g+Az+a)
={}&\widetilde g^\top S_v\widetilde g
+2\widetilde g^\top S_vAz
+z^\top A^\top S_vAz\\
&+2\widetilde g^\top S_va
+2z^\top A^\top S_va
+a^\top S_va.
\end{aligned}
$$

The $z/z$ term is quartic in the normalized previous-layer input $x$: it multiplies two already quadratic features. Keeping $z$ as a shared intermediate preserves a compact execution graph without materializing a fourth-order coefficient array for every vocabulary row.

There are two distinct discovery objectives:

- **Product-port objective:** treat entries of $z$ as formal coordinates and decompose $A^\top S_vA$. This preserves explicit interventions on those ports but can overestimate complexity because reachable $z(x)$ are constrained.
- **Composed-function objective:** substitute $z(x)$ and simplify the resulting quartic function. If represented as a coefficient tensor, use its fully symmetric four-input form: permuting the four copies of $x$ does not create a different function. This can expose cancellations hidden in product coordinates.

We should test both. A cheap composed function does not automatically preserve every original product-port intervention. Conversely, failure to find sparse structure in independent product coordinates is not evidence that the realizable composed function is complex. This distinction is central to the proposal.

## 5. Candidate representations and their assumptions

A coupled Tucker model for source-pair operators is

$$
K_{v,ij}\approx
\sum_{c=1}^{C}\sum_{\alpha=1}^{r_i}\sum_{\beta=1}^{r_j}
W_{vc}\,G^{ij}_{c\alpha\beta}
(B_i)_{:\alpha}(B_j)_{:\beta}^{\top}.
$$

$B_i$ learns features within source $i$, $W$ learns output directions shared across paths, and $G^{ij}$ specifies which feature pairs write to which output directions. Sparse cores mean many of these interactions are absent. Sharing $B_i$ across all its partner paths is the crucial difference from independently fitting every block.

| Method | Useful assumption | Limitation to test |
|---|---|---|
| Coupled sparse Tucker | Few interactions among shared source/output features. | Coordinate-dependent sparsity; a dense block can still have simple arithmetic. |
| Block-sparse Tucker | Small groups of features interact densely, most groups do not interact. | Group sizes and assignments can hide arbitrary complexity. |
| CP with general bilinear products | Shared scalar products write to output directions. | May require many factors for a naturally multi-dimensional operation. |
| LL1/block terms | One output direction receives a low-rank matrix interaction between two feature spaces. | Separate terms can duplicate readers; several-output blocks may be needed. |
| Hierarchical Tucker or tensor train | A high-order composed function has small interfaces across selected tensor partitions. | Tree/chain structure is imposed; a small tensor rank is not automatically a reusable semantic intermediate. |
| Shared arithmetic DAG | Intermediate linear combinations and products are reused across paths and outputs. | Harder discrete search, non-identifiability and optimization degeneracies. |

I would begin with coupled sparse/block-sparse Tucker and shared block terms, then compare a DAG model initialized from them **and** a jointly optimized DAG alternative. Post-fit simplification is useful but should not be the only way hierarchy can enter.

The sparse model should support shared and private features: a source can use one reader across several interactions and additional readers for only one branch. Output directions should also be learned jointly. We should not demand that a factor have a human-readable token category before testing it.

## 6. What sparsity and independence would mean

“Sparser” must refer to a counted implementation: nonzero interactions, independently stored readers/writers, intermediate state, multiplications and adapters. Fewer than vocabulary-size output factors alone is a weak criterion: the linear unembedding image already has dimension at most 1152. The desired gain is a simpler input-to-output computation, not that rank fact.

Sparsity can change under a basis rotation or rescaling. Use normalized or orthonormal feature frames when appropriate, charge the transformations, and check function/intervention equivalence across restarts. Also include a nonorthogonal alternative: an orthogonality constraint may exclude a useful overlapping feature system.

There are three different notions of independence:

1. **Algebraic separation:** an interaction operator is zero, or two blocks have disjoint computational support at the chosen boundary.
2. **Distributional separation:** features vary independently, or nearly so, on actual inputs.
3. **Causal selectivity:** editing one component preserves specified other behaviors.

None automatically proves the next. The experiments must label which one they measure.

Expansion itself can increase cost. The new six-path oracle uses nine native elementwise product contributions—three self terms and two for each mixed pair—versus one product bank for the original factored execution. It is an inspection representation, not a proposed faster implementation. A learned path program must recover enough sparsity or sharing to pay for that expansion, or retain the original factored sums for execution.

## 7. Normalization is a shared computation, not a disposable detail

For the existing suffix component, the just-completed check compared the four inputs $y$, $y+\Delta p$, $y+\Delta g$, and $y+\Delta p+\Delta g$. With the base denominator fixed, the scalar interaction is exactly

$$
\frac{2\Delta p^\top S_v\Delta g}{\rho(y)^2}.
$$

The actual interaction adds the correction from the changing input denominator. For the exact native reference, that correction's physical-write norm is **44.1%** of the full interaction norm for verbs and **9.72%** for nouns. These ratios are not additive variance shares; terms can cancel. The registered “at most25% in both families” prediction failed.

Nevertheless, fixing that denominator preserves all tested interaction signs. The final-margin interaction RMS becomes **1.48×** the actual value for verbs and **1.02×** for nouns. The quadratic interaction therefore survives without changing normalization; normalization partly offsets it, especially for verbs. Algebraic accounting error is below **9.39 × 10⁻¹⁴**, and prior native-margin replay differs by at most **2.39 × 10⁻⁶ nats**. [Code and full scope](../../native_suffix_input_normalization_v1.py).

This is a reason to retain both the numerator interactions and a shared normalization node. It is not evidence that all layers or tasks share this particular mechanism. The new proposal will distinguish numerator-only edge edits, edits that propagate into the shared denominator, and complete node edits; these are different interventions.

## 8. Fitting objective and convergence plan

Use the full unembedding, without materializing the vocabulary-by-input-by-input tensor. Its weight-space output metric can be represented exactly through $U^\top U$; if centering vocabulary rows, retain the common output component separately. This folds all token directions into the loss rather than choosing a few grammatical labels.

A proposed discovery objective is

$$
\min_{W,B,G}\quad
\mathcal E_{\mathrm{composed}}(W,B,G)
+\eta\mathcal E_{\mathrm{ports}}(W,B,G)
+\lambda\sum_{i\le j}\|G^{ij}\|_1
+\mu\,\mathcal C(W,B,G).
$$

$\mathcal E_{\mathrm{composed}}$ measures the error in the actual composed coefficient function. $\mathcal E_{\mathrm{ports}}$ measures fidelity of the separately exposed interactions when that intervention interface is required. $\mathcal C$ prices shared readers, writers and edges. The displayed penalty is a proposal, not an already implemented optimizer or a claim that one weighting is universally correct. Compare explicit budgets and a Pareto curve rather than selecting a convenient penalty after inspecting behavior.

For quadratic blocks, reuse exact Gram contractions. For higher-order composed objects, first establish the contraction cost and symmetry convention on small cases. If exact contraction is too expensive, use reproducible coefficient sketches with an independent error audit. Random synthetic inputs are a different, explicitly declared function-space metric—not automatically an unbiased Frobenius coefficient estimator. No language-data fitting is needed for either approach.

Optimize continuous parameters and structure in alternating steps: solve linear output/core subproblems accurately where possible; optimize normalized readers; update sparse support; and jointly refit. For a fixed support, variable projection can eliminate linear coefficients, but penalized subproblems require their appropriate solver rather than an ordinary least-squares shortcut. Use multiple initializations, continuation in sparsity, and support exchange. Reuse our existing contractions and fitting machinery where applicable.

Convergence requires a fixed-objective stationarity test, support stability or a recorded remaining support improvement, bounded coordinate scales, and an independent continuation check. A line-search stop or a flat objective is insufficient. Test planted recovery from independent starts and cancellation-heavy fixtures first. **No global recovery guarantee is assumed.** If optimization remains unconverged, report that before interpreting a poor fit as absent structure.

## 9. Staged experiments and opposing predictions

| Stage | Experiment and deliverable | Evidence that changes the decision |
|---|---|---|
| 0: exact representation | Extend the checked source-pair executor to coefficient contractions and native full-logit replay. Register which normalizers and ports remain external. | Any identity or intervention mismatch invalidates the instrument before fitting. The current six-path identity is completed; the general optimizer is not. |
| 1: output-map composition | Fit residual/residual, residual/attention and attention/attention jointly and separately, with full-U metric and matched total storage/compute budgets. | Joint structure should improve the accuracy/price frontier and reuse actual readers. A gain from uncharged adapters or coordinate gain does not count. |
| 2: previous MLP and mixed paths | Add MLP16/MLP16, MLP16/residual and MLP16/attention blocks; compare product-port and fully composed objectives. | Simplicity visible only after valid upstream substitution supports the composition hypothesis. Inability to retain old interventions must be reported, not hidden. |
| 3: deeper attention | Add current/first-value branches and both QK factors with native positions and denominators. Compare with the output-only boundary. | A smaller output-only core that requires an equally opaque routing program has not produced closed extraction. Shared route/value features must improve the total charged program. |
| 4: arithmetic structure | Compare sparse, block and jointly shared-DAG models; allow repeated subexpressions and distributive rewrites. | Reuse must save independently specified computation and preserve chosen ports, or explicitly declare a new intervention boundary. |
| 5: frozen circuit validation | Freeze weights/topology; validate on new FineWeb documents and newly specified task panels, then separately labeled corpus shifts such as Pile. | Promotion requires the four properties below. Task labels annotate discoveries after fitting; they do not secretly enter the discovery objective. |

Use managed GPU execution for native fitting and large contractions; use CPU for small exact controls and receipt analysis. There is no assumed two-CPU-hour restriction. Benchmark one objective/gradient step and memory footprint before setting GPU run budgets. The immediate disk constraint favors compact cores, seeds and sufficient receipts over large expanded tensors or repeated checkpoints.

Before each native fit, register concrete budgets, starts, tolerances and behavioral bars from its pilot price. This document specifies the research comparison; it is not a claim that every future run has already been preregistered or queued.

## 10. Four-property promotion criteria

| Property | Required evidence | Failure we must not relabel as success |
|---|---|---|
| OOD prediction | Predict feature values and signed intervention effects on frozen new documents/constructions; report corpus shifts separately. | Good training-weight fit or performance on repeatedly inspected examples alone. |
| Extraction | Execute the component from a declared upstream boundary, with all required residual, routing and normalization inputs listed and priced. Progress toward replacing those dependencies. | Supplying the original component output, or calling native-background conditional prediction a standalone circuit. |
| Removal | Specify a node or edge edit, test intended changes and broad unrelated controls, including lexical probability and grammatical alternatives. | Passing only a binary contrast while substantially changing other token probabilities. |
| Composition and reuse | Combine components and predict interactions; intervene on a shared node and check every claimed consumer. | Weight cosine, common support or exact addition without a tested shared computation. |

Compare candidate programs with matched-cost independent factors and generic controls. Inspect restart stability as a prerequisite for naming a particular basis vector; operationally equivalent programs may still be valid when individual coordinates are not unique.

## 11. Related work and the precise connection

- **Sharkey (2023), [A technical note on bilinear layers for interpretability](https://arxiv.org/abs/2305.03452).** Establishes the third-order tensor treatment of bilinear MLPs and its connection to transformer-circuit analysis. This is a direct algebraic foundation; our proposed contribution is a joint search over composed interaction objects.
- **Pearce et al. (ICLR 2025), [Bilinear MLPs enable weight-based mechanistic interpretability](https://arxiv.org/abs/2410.08417).** Studies weight-based spectral analysis of bilinear MLPs and reports interpretable structure and small circuits. It motivates weight-first discovery, but does not by itself establish a sparse multi-layer decomposition for this checkpoint.
- **Kolda and Bader (2009), [Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf).** Provides CP/Tucker definitions, algorithms and identifiability context. Tucker supplies shared mode bases and an interaction core; it does not automatically deliver core sparsity or a semantic hierarchy.
- **De Lathauwer (2008), [Decompositions of a Higher-Order Tensor in Block Terms—Part II](https://epubs.siam.org/doi/pdf/10.1137/070690729); Domanov and De Lathauwer (2018), [On uniqueness and computation of rank-(1,Lr,Lr) decompositions](https://arxiv.org/abs/1808.02423).** Block terms match the idea that one output direction can receive a multi-dimensional bilinear interaction. Their uniqueness conditions must be checked; shared/overlapping paths and polynomial symmetries do not automatically satisfy them.
- **Grasedyck (2010), [Hierarchical Singular Value Decomposition of Tensors](https://publications.rwth-aachen.de/record/170375).** Hierarchical tensor formats organize multilinear subspaces along a dimension tree. They are a useful comparator for the composed quartic/high-order object. A selected dimension tree is not learned multi-parent arithmetic reuse.
- **Oseledets (2011), [Tensor-Train Decomposition](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf).** Tensor trains factor high-order arrays into a chain of small cores. Relevant to avoiding explicit high-order tensors; the chain ordering and intermediate ranks are assumptions, not a guarantee of circuit interpretability.
- **Golub and LeVeque (1979), [Extensions and Uses of the Variable Projection Algorithm](https://faculty.washington.edu/rjl/pubs/GolubLeVeque1979/GolubLeVeque1979.pdf).** Eliminating linear unknowns from separable nonlinear least squares motivates exact output/core solves inside reader optimization. It does not solve discrete topology selection or guarantee the global optimum.

The sparse-core, shared/private-path and joint-DAG combination above is our proposed experimental design. It should not be described as a theorem from these papers, an established best method for this model, or a completed literature-exhaustiveness claim.

## Recommended decision

Proceed with the joint path comparison. The strongest new question is whether **shared computation becomes simple only after upstream and downstream weights are composed**, including mixed terms and constrained intermediate features. Start from the exact full-output operators, keep all normalization dependencies explicit, compare several structural assumptions with adequate optimization, and validate frozen discoveries afterward. Success would give us a better unit of decomposition than a native layer or head—and a concrete route toward the four circuit properties.

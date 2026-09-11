# Research update: better weight-based factorization, remaining instability, and a useful validation surprise

**11 September 2026, 13:27 UTC; completion addendum through 13:30. Requested report for Logan.** This covers the work since the [09:02 requested methods update](weights_first_methods_2026-09-11_0902.md), including results subsequently described in the shorter automatic updates. The previous broad report from 10 September is also [in this folder](research_update_2026-09-10_1842.md). This report concerns Codex's unembedding/MLP17 research; it does not count Claude's parallel circuit work as mine. MLP17 is the model's last bilinear layer, numbered from zero.

## The high-level rundown

**We made a substantial improvement in weight-based fitting, but have not yet obtained a stable decomposition into circuits with your four properties.** The strongest new result is that the learned representation preserves the model's predictions on a small FineWeb check much better than a similarly sized native-product baseline—even though that baseline reconstructs the weight tensor better.

The sequence was:

1. **We repaired how the shared features are used.** Better sparse connection selection raised full polynomial coefficient capture from about 43.8% to 53.6%, without learning new feature directions in that repair.
2. **We changed discovery from a proxy to the actual composed computation.** Jointly fitting the shared readers and their outputs raised capture from about 54.0% to 64.7%. These runs used weights, including the entire unembedding, rather than text.
3. **We tested the restrictive connection graph.** Allowing connections to change added another 0.31 percentage point, reaching about 65.0%. This helped, but did not resolve disagreement between independently initialized fits.
4. **We investigated cancellation and stability.** A penalty greatly reduced large opposing components while preserving almost all the fit. The first penalized run still did not converge. Two similar reconstruction scores do not yet imply that we recovered the same computation.
5. **We looked for a shared computation inside the unstable fits.** One scalar function was reproducible enough to examine. It partly recovers previously documented pronoun-related products; it is not a newly discovered gender circuit. Its activation also has a strong punctuation association.
6. **We challenged a discouraging weight-only comparison with frozen validation.** Native product selection wins on coefficient error, but the learned dictionaries win clearly on the checked FineWeb predictions. That is a concrete reason not to discard the structure because of the weight score alone.
7. **We derived a different structural family to try next:** overlapping groups that each share one input reader and have a small set of partner readers/output directions. The conditional mathematics is derived; its new reader-update implementation and a multi-group native fit are not completed or queued at this report's cutoff.

**No new million-token discovery sweep was added.** Your intended order remains: find structure from weights, freeze the candidates, then validate with FineWeb. Pile is a separate distribution-shift test, and previously Pile-adapted candidates cannot claim untouched Pile validation.

## What we are actually factorizing

At the last layer's already-normalized input, the native bilinear computation is

$$
q(x)=D\big[(Lx)\odot(Rx)\big]+b.
$$

Here $x$ has 1,152 coordinates. The matrices $L$ and $R$ each contain 4,608 **readers**: vectors whose dot products with $x$ extract linear features. The symbol $\odot$ means multiply corresponding entries. The matrix $D$ writes the resulting products back into the residual stream. The unembedding $U$ converts residual directions into scores for 50,304 vocabulary rows.

Folding in all of $U$ gives a quadratic interaction matrix for every token $v$:

$$
T_v=\sum_{j=1}^{4608}(UD)_{vj}\operatorname{sym}(l_jr_j^\top),
\qquad
\operatorname{sym}(A)=\frac{A+A^\top}{2}.
$$

Thus $[Uq(x)]_v=x^\top T_vx+(Ub)_v$. Tokens already share the native products; discovery asks whether a different organization exposes simpler, reusable computations.

We optimize the entire collection of token matrices implicitly. There is no need to allocate the approximately 66.8-billion-entry tensor. Product-matrix inner products reduce to dot products:

$$
\langle\operatorname{sym}(ab^\top),\operatorname{sym}(cd^\top)\rangle_F
=\frac{(a^\top c)(b^\top d)+(a^\top d)(b^\top c)}{2}.
$$

Together with $U^\top U$, this evaluates the full-vocabulary objective exactly, up to floating-point arithmetic.

Our **coefficient capture** is

$$
1-\frac{\sum_v\|T_v-\widehat T_v\|_F^2}{\sum_v\|T_v\|_F^2}.
$$

It measures reconstructed squared coefficient energy. It is not token accuracy or the fraction of the model explained. This folded quadratic is only one branch of the full model: residual addition, final RMS normalization, and the $30\tanh(\cdot/30)$ logit cap remain essential.

## How the better factorization worked

The representation learns 2,304 shared linear features, collected in a matrix $B$. Each native left/right reader uses 128 of those features:

$$
z=Bx,\qquad
\widehat F(x)=UD\big[(C_Lz)\odot(C_Rz)\big].
$$

The sparse matrices $C_L,C_R$ specify the connections and their strengths. An **overcomplete dictionary** means there are more available features than input dimensions; features can overlap and need not be orthogonal.

The first improvement was an encoder repair. The previous procedure used Lasso, which penalizes absolute coefficient magnitudes. If it selected fewer than 128 nonzero connections, arbitrary zero-valued ties filled the remaining positions. We instead filled those slots by **orthogonal least squares**: choose the next feature giving the greatest exact conditional reduction in reconstruction error, then refit all selected values. This is a greedy search, not an optimal search over every possible support.

The larger improvement came from changing the fitting objective. Earlier dictionary learning approximated individual native reader vectors; it did not directly optimize how their errors interact after multiplication and output writing. The new method minimizes the full folded polynomial error. At each step, it solves exactly for the best $D$ given the readers, then updates $B,C_L,C_R$ with L-BFGS-B, a limited-memory method that estimates useful curvature from recent gradients. Eliminating the linear output unknowns is called **variable projection**.

| Stage | Full coefficient capture, two starts | Interpretation |
|---|---:|---|
| Learned dictionary with old encoder | 43.77%, 43.78% | Sparse inference was leaving useful connections unused. |
| Same learned features, repaired encoder | 53.60%, 53.61% | Large improvement from using existing features better. |
| Repaired initialization plus exact output solve | 53.954%, 53.957% | Starting point for direct polynomial fitting. |
| Direct full-tensor joint fit | 64.685%, 64.687% | About 10.73 percentage points gained; both one-hour runs unconverged. |
| One full connection-exchange sweep | 64.998%, 64.998% | About 0.312 percentage point more; not graph convergence. |

The coupled optimizer on the older reader proxy gave negligible further improvement and remained unconverged. We therefore did not automatically repeat it. These stages reuse the same main dictionary family; changing the solver is not evidence that another structural hypothesis has been exhausted.

The full exchange compares replacing connections against just refitting the existing ones. It revisits all 9,216 readers, recomputing conditional interactions as it goes. Same-connection refitting added less than 0.000028 percentage point; changed connections added about 0.312. The four-arm experiment took about 84 seconds overall. One sweep does not solve the combinatorial graph-search problem.

The learned program stores 9,142,272 matrix coefficients, 1,179,648 connection indices, and 1,152 bias values. The unembedding and remaining native model are additional retained dependencies. [Encoder results](../../OVERCOMPLETE_OLS_REENCODE_V1_RESULT.json), [joint fit](../../PROJECTED_SPARSE_DICTIONARY_FIT_V1_RESULT.json), [connection exchange](../../FULL_SUPPORT_EXCHANGE_V1_RESULT.json).

## Why similar fits still do not identify the same circuits

We compared the complete coefficient functions, which avoids being fooled by simple factor permutations or sign flips. Their cosine similarity was only **0.7448**, increasing to **0.7468** after connection exchange, below the registered 0.9 target. Equal capture can mean explaining different parts of the tensor.

We also tested a tempting repair: could changing only the output weights align the functions while barely worsening reconstruction? An exact conditional-projection identity gives a conservative bound. With at most 0.001 capture loss for each frozen-reader fit, attainable cosine is bounded above by **0.8448**. This rules out that particular inexpensive output-only repair; it does not constrain changing readers or discovering another family. [Bound and numerical check](../../OUTPUT_ONLY_STABILITY_BOUND_V1_AUDIT.json).

**Cancellation** is another issue. Products can have large individual outputs that oppose one another in their sum. We measured summed component energy relative to the native tensor's energy and added a penalty:

$$
\mathcal L=
\frac{\|T-\widehat T\|_F^2}{\|T\|_F^2}
+\lambda\frac{\sum_j\|\widehat T_j\|_F^2}{\|T\|_F^2}.
$$

This discourages unnecessarily large components. It changes a preference within the family; it is not a new family or a proof that every cancellation is spurious.

An exact fixed-reader solve first reduced component energy fourfold for only **0.0879 percentage point** of capture loss. The subsequent joint penalized fit's first 20-minute start reduced energy from **3.069 to 0.613** times native energy—roughly fivefold—while retaining **64.6851%** capture. However, its joint stationarity measure remained about **0.00152**, above the required **0.00001**. It is an improved representation, not a converged one. The second start finished while this report was being published; see the completion addendum below.

We check gradients rather than declaring convergence from a flat loss curve. One early native finite-difference check failed; using a tenfold smaller differencing step reduced its discrepancy about 100-fold and passed. The original miss remains recorded. This distinguishes a derivative-check numerical issue from a structural failure. [Component budget](../../OUTPUT_COMPONENT_BUDGET_V1_AUDIT.json), [first penalized result](../../PENALIZED_PROJECTED_FIT_V1_SEED_0.json).

## The shared scalar function: a partial recovery of known structure

A scalar function is one weighted combination of output coordinates. We searched for functions that agree across the two fits, first with the same output reader and then allowing a separate reader for each fit. The latter uses **canonical correlation**: normalize each function space by its own coefficient inner product and find the most correlated pairs.

Only **one** pair exceeded 0.95 correlation, rather than the hoped-for 16. Its correlation was **0.9643**; each approximation had about **6.6–6.7% squared coefficient error** relative to its corresponding native scalar function. Those native functions agreed at cosine **0.99993**. This is a small positive within a failed broad-coverage prediction.

We then formed that native scalar's quadratic matrix and eigendecomposed it. Pairing its largest positive and negative eigenvalue directions gives optimal small banks of real two-reader products for this fixed scalar matrix. One product captured **81.0%**, four **90.7%**, and sixteen **92.2%** of its coefficient energy. The scalar itself accounts for only about **0.108%** of the whole tensor's energy; these are not whole-layer percentages.

Frozen validation on the existing 64-by-128 FineWeb state cache was less favorable. One and four products each left about **49% relative mean-squared scalar error**. Sixteen reduced this to **9.94%**, with centered correlation **0.973**. The extra small coefficient tail mattered disproportionately on actual inputs: its relative natural-input energy was **15.44 times** its relative coefficient energy. This diagnoses the mismatch without fitting factors to the data.

The output direction loads positively on forms such as “he/his/him” and negatively on “she/her.” Checking the dossier found earlier pronoun-related products. Their two-product input span explains about **81–82%** of the new scalar; their writer span explains only about **62%** of its output direction. This is partial recovery of prior structure, not a new gender circuit. In addition, all 32 cached rows containing an opening-parenthesis token had lower mean scalar activation there than elsewhere in that row. That is a post-selected association, not a causal punctuation experiment, but it makes a simple input-gender interpretation doubtful. Earlier failed sufficiency and selective-removal tests remain in force.

[Canonical comparison](../../CANONICAL_FUNCTION_AGREEMENT_V1_AUDIT.json), [native products](../../SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json), [FineWeb scalar check](../../SHARED_FUNCTION_FINEWEB_V1_AUDIT.json), [tail analysis](../../SHARED_FUNCTION_TAIL_ENERGY_V1_AUDIT.json), [prior overlap](../../SHARED_FUNCTION_PRIOR_ALIAS_V1_AUDIT.json), [punctuation check](../../SHARED_FUNCTION_PUNCTUATION_V1_AUDIT.json).

## The strongest practical result: coefficient ranking reverses on text

We tested whether the learned dictionary merely benefits from its size. A baseline selects **2,645 native products** by their full-unembedding component energy, then solves their output weights exactly. It uses slightly fewer matrix coefficients and needs no sparse graph indices. It captures **68.63%** of the tensor, beating the learned dictionaries' **65.00%**. A random native-product baseline captures **59.39%**. Thus the dictionary does not yet demonstrate a matched-size coefficient advantage.

But that is not the same as preserving the computation on text. We froze all three programs and evaluated the actual final normalization and logit cap at **128 positions from 64 previously opened FineWeb sequences**:

| Frozen program | Added cross-entropy, nats | KL divergence, nats |
|---|---:|---:|
| Selected native products | +0.3546 | 0.2940 |
| Learned dictionary, start 0 | +0.0119 | 0.0212 |
| Learned dictionary, start 937 | +0.0358 | 0.0224 |

**Cross-entropy** is the average negative log probability of the observed next token; added cross-entropy is damage relative to the original model, so lower is better. **KL divergence** measures the change in the entire next-token probability distribution. The original model's mean cross-entropy on this panel was 3.2136 nats.

The dictionary wins clearly on both measurements. The numerical replay discrepancy was tiny: relative logits $4.39\times10^{-7}$ and mean cross-entropy $2.04\times10^{-7}$.

The reason these rankings can differ is mathematical:

$$
\underbrace{\sum_v\|E_v\|_F^2}_{\text{coefficient error}}
\quad\text{and}\quad
\underbrace{\mathbb E_x\sum_v(x^\top E_vx)^2}_{\text{natural-input error}}
$$

weight errors differently. The second quantity depends on fourth-order input statistics; normalization and the logit cap further affect predictions. The result supports continuing weight-based discovery and validating frozen candidates. It does not require changing discovery to a data-trained objective.

The limitation is important: this is a **small, historically inspected panel**, not fresh validation or OOD evidence. Mean absolute per-position cross-entropy changes are still about 0.133 nats for the dictionaries. Small average damage does not prove selective preservation. The registered prediction that all three programs would meet preservation bars failed because the native baseline did not; the dictionaries' individual passes do not erase that overall miss. [Weight baseline](../../MATCHED_PRICE_NATIVE_V1_AUDIT.json), [frozen prediction check](../../MATCHED_PRICE_FINEWEB_V1_AUDIT.json).

## What the math cycles contributed, and what is genuinely new next

The 10:51 mathematical review produced two useful distinctions and executable controls. First, a whole interacting quadratic block can remain unchanged while its internal feature axes rotate. The individual axes may therefore be arbitrary coordinates even when the block is a meaningful computational unit. A tested tool measures how much a proposed reader change can be canceled by output reweighting. Second, gradient size can change dramatically under equivalent rescalings; a tested rowwise measure avoids that particular ambiguity. Neither control establishes native circuit identification. [Review and derivations](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1051.md).

Those ideas helped motivate the later comparison of complete functions, the output-only stability bound, and the search for a shared scalar. Exact linear solves also made it practical to separate poor output fitting from poor input features. These are useful consequences, but the cycles have not delivered the desired complete decomposition.

The proposed next structural family is

$$
\widehat F(x)=\sum_{j=1}^{m}(a_j^\top x)M_jx,
\qquad \|a_j\|=1,\quad \operatorname{rank}(M_j)\le r.
$$

Each group shares a reader $a_j$ across a limited set of partner readers and output directions. Groups may overlap. This directly expresses reuse of a common input feature across several products, rather than assuming all units are independent one-product outputs or forcing one global orthogonal basis.

For fixed $a_j$, prior code already gives an exact conditional rank-$r$ partner update through a transformed singular-value decomposition. For fixed $M_j$, we have now derived a sphere-constrained quadratic reader update: solve a small spectral problem and a scalar norm equation, including the nullspace case. **That new update is derived, not yet implemented or tested.** Exact conditional updates do not guarantee a global optimum of the joint multi-group problem. Existing one-group experiments are prior art; they should not be counted as a new multi-group run.

Remaining bottlenecks are nonlinear conditioning, sparse graph search, cancellation, unequal component sizes, and how to define stable units despite coordinate freedom. A native penalized gradient probe took about 1.48 GPU seconds; many evaluations, line searches, and repeated output-system solves accumulate. The giant token tensor is not the main memory bottleneck because it stays implicit. We have tried established solver families; we have not established that every family received the best possible optimizer or enough optimization to reject its structure.

## Where this leaves the four properties

| Desired property | New evidence | What remains missing for these weight-derived candidates |
|---|---|---|
| Held-out/OOD prediction | Frozen FineWeb scalar and probability checks; an informative ranking reversal. | Fresh document validation, separately labelled corpus shift, and prediction of circuit-specific effects. |
| Extraction/sufficiency | Executable replacement layers and a compact native scalar approximation. | An isolated target computation with explicit background that passes sufficiency tests. |
| Selective removal | Exact feature-removal formulas and reusable intervention interfaces. | Selective behavioral effects; existing pronoun sufficiency/removal misses are not repaired. |
| Composition/reuse | A shared-feature program and one partially stable scalar function. | Stable interpretable groups whose separate and joint edits have predictable effects. |

The evidence justifies further structural work. It does not yet establish a four-property circuit or a model-wide decomposition. No new QK-native experiment completed in this reporting interval; the current full fit is **MLP17 plus the whole unembedding**, not a fitted unembedding → MLP17 → attention path. Earlier backward-attention and joint-QK results remain in their topic notes and the [method index](../../WEIGHT_ONLY_METHODS_INDEX.md).

## Files, running work, and disk space

Requested fuller updates now live in **`explanations/for_logan/`**. Start at [LATEST.md](LATEST.md); [README.md](README.md) lists the requested reports newest first. Four identifiable earlier requested reports were moved here, with old-path navigation stubs and rebased links. Routine automatic updates remain in the dated folders. Historical report wording records its original snapshot, not today's live queue.

At the initial 13:27 cutoff, the second penalized start was running. It finished at 13:29:35; the managed runner subsequently completed its canary, and the inspected queue is empty. Both penalized starts are complete and unconverged. No duplicate runner or direct GPU job was launched. The shared-reader group proposal is not yet a native queued experiment.

Disk was down to about 146 MiB free. I removed two confirmed inactive older VS Code server distributions and npm's reinstallable download cache, reclaiming **1,553,666,048 bytes, about 1.45 GiB**. Free space rose to about **1.59 GiB**. The active editor, installed extensions, model/data/research artifacts, Git history, agent histories, and running experiment were retained. A cache directory referenced by the live editor was skipped. This cleanup did not move research data into volatile memory. [Cleanup receipt](../../DISK_CLEANUP_2026-09-11_1327.json).

## Completion addendum — 13:30 UTC

The second penalized start finished at **64.6855% capture**, **0.61356 component energy**, and missed convergence. The final two-start function cosine is **0.74484**, still far below 0.9. Numerical validity, objective improvement, and the capture/energy tradeoff passed; convergence and stability failed. Reduced cancellation alone has therefore not repaired this family's identification problem. No identical continuation was queued. [Final two-start receipt](../../PENALIZED_PROJECTED_FIT_V1_RESULT.json).

I also executed a cheap allocation check for the proposed shared-reader groups, using the existing native spectral cache. At the same matrix-coefficient budget, 122 groups with partner rank 32 can capture at most **46.20%** of coefficient energy under this family, even with perfect optimization. Rank 16 permits 240 groups and has a looser **76.11%** upper bound; the rank 4/8 allocations are not excluded by this bound. This helps avoid an allocation that cannot meet the current weight score. It is not a bound on natural-input fidelity or general circuit structure. The new conditional-reader solver remains unimplemented. [Executed allocation check](../../SHARED_READER_GROUP_ALLOCATION_V1_AUDIT.json).

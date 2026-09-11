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

<a id="factorization-explained"></a>
## Requested explanation: what counts as a factor, and are we doing CP or LL1?

**Added 11 September, 13:50 UTC.** Your starting picture is right: take the bilinear layer, fold in the unembedding, view the result as a third-order tensor, and slice it along the output axis. Each slice is a complete quadratic interaction matrix for one output token. We then look for a simpler set of computations shared across those matrices.

**We are not committed to finding only a smaller CP decomposition.** CP/product factors, LL1 blocks with a shared output, and groups with a shared input reader are different ways to organize the same polynomial. Your proposed “multiple input spaces, provided they write to one output direction” is precisely the kind of grouping LL1 is designed to express. The latest shared-reader experiment instead shares an *input* direction. I should have made this distinction explicit earlier.

### 1. A token slice is a function, not yet a discovered factor

The exact object is

$$
T_{vij}=\sum_{k=1}^{4608}(UD)_{vk}
\frac{L_{ki}R_{kj}+R_{ki}L_{kj}}2,
\qquad F_v(x)=\sum_{i,j}T_{vij}x_ix_j.
$$

The three axes are **output token, first input coordinate, second input coordinate**. Both input coordinates belong to the same vector $x$. Consequently, $T_{vij}=T_{vji}$: exchanging the two input indices cannot change the polynomial.

For example, a token slice could encode

$$
F_v(x)=2x_1x_2-3x_3^2+x_2x_4.
$$

That entire quadratic is the slice. Calling the slice a “factor” would skip the discovery problem. We want to find common subexpressions that many slices use, and describe how each token combines them.

In the generic shared-function notation,

$$
F_v(x)=\sum_{g=1}^{G}c_{vg}\phi_g(x).
$$

The function $\phi_g$ is one proposed shared computation. The column $c_{:g}$ tells us how it writes across **all** output tokens. Tokens can use several functions, with positive or negative coefficients. This naturally allows overlapping token sets; it does not force a token clustering or a hierarchy.

The central question is what restrictions make each $\phi_g$ simple. If arbitrary dense quadratics are allowed, reducing the number of columns can just hide the complexity inside each function.

### 2. Product factors and CP: one pair of readers per contribution

The product-based model is

$$
\widehat F(x)=\sum_{k=1}^{K}c_k(a_k^\top x)(b_k^\top x).
$$

Each term reads two linear features, multiplies them, and writes one output direction $c_k$. Every token shares the same input products but has its own coefficients $c_{vk}$. We jointly learn the factors across outputs; we are not independently eigendecomposing each token and matching the results afterward.

There is a terminology subtlety. An ordinary rank-one **CP term** is $c_k\otimes a_k\otimes b_k$. Our symmetric coefficient tensor uses

$$
c_k\otimes\operatorname{sym}(a_kb_k^\top)
=\frac12c_k\otimes a_k\otimes b_k
+\frac12c_k\otimes b_k\otimes a_k.
$$

Thus a general real two-reader product corresponds to a tied pair of ordinary CP terms. A square, with $a_k=b_k$, is one partially symmetric CP term. Counting products, ordinary CP terms, and squares as if they were the same rank can mislead comparisons. The standard CP definition is a sum of rank-one outer products. [Kolda–Bader survey](https://www.kolda.net/publication/koba09/).

The native layer already gives an exact **4,608-product** representation. A new product decomposition asks whether different readers and writers need fewer products, or yield more useful and stable computations at comparable cost. Restricting each output slice to a low matrix rank independently is a different problem; it does not force sharing across tokens.

### 3. LL1: several input interactions share one output direction

In our output-first axis order, the output-sharing block model is

$$
\widehat T=\sum_{g=1}^{G}c_g\otimes Q_g,
\qquad
\widehat F(x)=\sum_{g=1}^{G}c_g\underbrace{x^\top Q_gx}_{\phi_g(x)}.
$$

If $Q_g$ has matrix rank at most $L_g$, the block has multilinear ranks at most $(1,L_g,L_g)$. Put the output axis last and these become $(L_g,L_g,1)$—the usual **LL1** convention. Tensorlab writes this as an outer product of a low-rank matrix and one vector. Its documented solvers include generalized-eigenvalue initialization and nonlinear least-squares refinement. [Tensorlab LL1 documentation](https://tensorlab.net/doc/ll1.html).

For a symmetric quadratic, one convenient parameterization is

$$
Q_g=A_gH_gA_g^\top,\qquad H_g=H_g^\top,
$$

where the columns of $A_g$ span that group's input space and $H_g$ specifies interactions within it. The computation is: read $A_g^\top x$, evaluate the small quadratic form, write $c_g$ times the resulting scalar. Different groups' input spaces may overlap. The output direction can also overlap other groups' output directions.

Alternatively, to retain explicit two-reader products,

$$
\phi_g(x)=\sum_{s=1}^{p_g}(a_{gs}^\top x)(b_{gs}^\top x).
$$

All $p_g$ products now share the same output vector $c_g$. This is the output reuse you described. The resulting symmetric $Q_g$ has rank **at most $2p_g$**, not necessarily $p_g$. In an unconstrained unsymmetric LL1 parameterization, symmetrizing $AB^\top$ can likewise double its input matrix rank. We must keep that distinction when translating software's LL1 rank to our same-input polynomial.

LL1's group count alone does not price the computation. One group containing a full-rank 1,152-dimensional quadratic can be expensive. We should report both **how many output groups** and **how complicated each input function is**, including shared input features between groups.

An LL1 block can be expanded into smaller product terms. Its benefit is the hypothesis that those terms belong together because they write the same scalar variable into the same output direction. The internal basis may rotate without changing the block, so the whole block can be more meaningful than its individual factor columns.

### 4. The newest method shares an input reader instead

The new family I implemented is

$$
\widehat F(x)=\sum_{g=1}^{G}(a_g^\top x)M_gx,
\qquad M_g=W_gV_g,\quad\operatorname{rank}(M_g)\le r_g.
$$

Here one feature $a_g^\top x$ is reused by several partner readers. The partner products may write **different output directions**, supplied by columns of $W_g$.

This is not output-sharing LL1. Before symmetrization it has one rank-one *input* mode; after symmetrization, the group's input space is contained in the span of $a_g$ and the partner readers. Its multilinear ranks are bounded by $(r_g,r_g+1,r_g+1)$ in output-first order, with additional shared-reader structure. These are upper bounds, not assertions that every group attains those ranks.

I chose this family to ask whether the model reuses a common input feature across several downstream operations. Your LL1 proposal asks the complementary question: do several input operations cooperate to write one common output variable? **Both are reasonable structural hypotheses. Success or failure of one does not settle the other.**

### 5. The same computation can have several sensible organizations

Consider a toy vector output with two output directions $c_1,c_2$:

$$
F(x)=c_1(x_1x_2+x_3x_4)+c_2x_1x_3.
$$

| View | The units it exposes |
|---|---|
| Individual products | Three contributions: $c_1x_1x_2$, $c_1x_3x_4$, $c_2x_1x_3$. |
| Output-sharing blocks | Two scalar functions: $x_1x_2+x_3x_4$ writes $c_1$; $x_1x_3$ writes $c_2$. Their symmetric input ranks are four and two. |
| Shared-input-reader groups | $x_1(c_1x_2+c_2x_3)$ reuses $x_1$; the remaining group is $x_3(c_1x_4)$. |

All three are exact. Each exposes a different kind of reuse. Which organization deserves to be called a circuit depends on how its variables are used and what its interventions predict—not just which grouping has the shortest name or the fewest top-level terms.

### 6. Why “fewer than vocabulary size” is too weak by itself

The vocabulary has 50,304 rows, but the layer already has only 4,608 native products. More strongly, since all token matrices are linear combinations through $U$, their linear span has dimension at most **1,152**, the residual width:

$$
T_v=\sum_{o=1}^{1152}U_{vo}S_o,
$$

where $S_o$ is the quadratic matrix for residual output coordinate $o$. So an exact representation with at most 1,152 shared **unrestricted quadratic functions** is available before discovering any new structure. Those functions can still be dense and opaque.

That is an upper bound on the number of unrestricted scalar functions, **not** an upper bound of 1,152 on product/CP rank. Each dense scalar quadratic may require many products. A smaller approximate output basis can be obtained by matrix SVD of the output unfolding, but that only minimizes error at the specified output rank; it does not make the input functions simple or identify semantic units.

Our meaningful comparison should therefore include:

- the number of groups/output variables;
- the ranks, products, or small cores needed to compute each variable;
- reusable readers and intermediate results shared between groups;
- all learned constants, adapters, and remaining native background;
- reconstruction and frozen behavioral fidelity;
- stability and the four eventual circuit properties.

A token hierarchy is an additional hypothesis about the loading matrix $[c_{vg}]$. It need not be imposed before discovering the input functions. A coherent output loading can involve antonyms or positive/negative contrasts rather than a conventional cluster.

### 7. What we have actually optimized so far

The strongest completed native dictionary fit is **not a search for minimum CP rank**. It retains 4,608 product slots, but generates their readers from 2,304 reusable features, with 128 connections per reader. It optimizes those features, connection strengths, and output writes against the full folded tensor. Its approximately 65% coefficient capture measures that family at that budget. It does not answer whether a much smaller output-sharing LL1 representation exists.

Earlier work includes free product banks, signed squares, output-function dictionaries, block/core fits, and one-output local group approximations. Those are relevant prior results, but they do **not** establish that a broad, adaptive-rank, jointly optimized, symmetric LL1 model of the whole tensor has been exhausted. We should not relabel the recent shared-input experiment as that missing experiment.

The latest multi-group results are still **planted-problem tests**. Near-initialized exact block updates work. Some random/spectral starts fail. A joint solver can generate huge opposing groups; a whole-group energy penalty controls that growth. With the penalty, three of eight random starts recover the planted grouping closely. One selected candidate has relative coefficient error 0.0001604 and matched group cosine 0.999752, but the four-of-eight recovery-rate prediction fails. This is a demonstrated initialization problem, not evidence against structure.

A native 64-group/rank-8 **shared-input** cost pilot is implemented in draft, but it is not yet bound, fully checked, or queued at this appendix's cutoff. It must not be described as a native LL1 result. Your clarification also makes a direct output-sharing LL1 comparison a distinct candidate for the next mathematical review.

### 8. The fitting criterion, signs, and remaining assumptions

The default discovery loss compares the complete symmetric coefficient tensor, using every unembedding row. It stays implicit through factor contractions and $U^\top U$; we need not materialize the giant vocabulary tensor. We can work in at most 1,152 output coordinates using a metric-preserving factor of $U^\top U$. This is an exact representation of the folded objective, not an arbitrary token subsample.

A coefficient objective favors some errors differently from natural model states. That is why frozen FineWeb validation remains necessary. It does not require using FineWeb to learn the factors. The latest small-panel result already showed a reversal: native selection wins on coefficient capture but loses badly on prediction preservation.

All factors are real and signed. Nonnegative CP/NMF assumptions would therefore be inappropriate without a justified reformulation. Also, every homogeneous quadratic satisfies $F(-x)=F(x)$. This antipodal symmetry is automatic; by itself it neither identifies a unique basis nor proves semantic clusters. The bias and full network's residual/normalization operations must be accounted for separately.

We also must distinguish three claims: a low-error representation **exists**; our optimizer **finds** it; its groups are **identified circuits**. The planted failures explicitly separate the first two. Your four properties—OOD prediction, extraction, selective removal, and composition/reuse—address the third.

### 9. A self-contained brief for browser Codex

I created **[factorization_browser_brief_2026-09-11.md](factorization_browser_brief_2026-09-11.md)** with the equations, dimensions, actual methods/results, constraints, key literature, and concrete questions. Paste that file's contents into browser Codex; it contains the essential context without requiring access to this workspace. It asks specifically about symmetric LL1/output-sharing blocks, adaptive block ranks, scalable solvers, and ways to distinguish poor optimization from absent structure.

A relevant method-selection lead is Rontogiannis, Kofidis and Giampouras's work on jointly estimating block counts and ranks using hierarchical sparsity and iteratively reweighted least squares. That is a literature candidate to inspect and adapt, not a method already run here or a proven recovery guarantee for this model. [Primary paper](https://arxiv.org/abs/2002.09759).

**Implementation note,13:54:** the scheduled mathematical review has now produced a small symmetric LL1 conditional-update tool: fixed output direction → best signed low-rank input quadratic, and fixed quadratic → best output direction. Dense numerical checks pass; two near-initialized planted blocks recover to4.63e-13error. This is a CPU algebra/control result, not a native LL1 fit. [Math review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1351.md).

Logan: I have some other thoughts, haven't read this carefully though. 

We can find shared inputs, shared outputs, but both (tucker?) might not be possible given a simple example that shows you can't have both maximally in every situation (right?). Then for finding other structure, like hierarchical or DAG (multiple parents are allowed), I think you have to do other things, right?

**Yes—with one distinction: Tucker can represent shared inputs and shared outputs simultaneously. What you cannot assume is that one organization minimizes every kind of computation at once.**

There are three separate questions.

**1. What is shared?**

A symmetric Tucker representation is

$$
h=B^\top x,\qquad
F(x)=C\,s(h),\qquad
s_g(h)=h^\top H_gh.
$$

It shares input readers through \(B\), and output directions through \(C\). The core \(H\) specifies their interactions.

That representation can be exact. But a dense core can contain nearly all the original complexity. **Shared spaces alone do not establish simple shared computations.**

**2. Which factoring choices compete?**

Your example captures this:

$$
F=p(ab+cd)+q(ac)
  =a(pb+qc)+p(cd).
$$

Here \(a,b,c,d\) are scalar input features; \(p,q\) are output vectors.

* The first expression combines interactions before writing through \(p\).
* The second combines output-valued expressions before multiplying by \(a\).

These are different arithmetic organizations of the same function. Which is cheaper depends on whether you count scalar products, vector scaling, additions, or reader evaluations.

Crucially, **computing the reader \(a\) once and reusing it is different from algebraically pulling the multiplication by \(a\) outside a sum**. The former is compatible with the first expression too.

So your intuition about competing factorizations is right. A claim that both cannot be “maximal” needs a specified operation set and cost. Sometimes the apparent conflict comes from demanding disjoint groups; overlapping DAGs already remove that restriction.

**3. How do we discover hierarchy or a DAG?**

Tucker supplies a fixed template:

$$
\text{linear readers}\;\to\;\text{quadratic interactions}\;\to\;\text{output mixing}.
$$

Discovering a richer computational organization requires searching for **relations among the factors**, such as:

| Structure              | What to look for                                                         |
| ---------------------- | ------------------------------------------------------------------------ |
| Reused intermediate    | The same expression occurs in several computations                       |
| Hierarchy              | Several expressions share a subexpression that itself has reusable parts |
| Multiple parents       | One computation depends on several previously discovered intermediates   |
| Alternative arithmetic | Distributive rewrites expose cheaper combinations                        |

For example,

$$
u=a+b,\qquad v=c+d,\qquad
s_1=uv,\qquad s_2=ue,\qquad F=ps_1+qs_2.
$$

Now \(u\) is reused, and \(s_1\) has two computational parents, \(u\) and \(v\). To discover this, we must factor the **reader computations themselves**, not merely group tokens by their loadings.

There is also a limit to what this last-layer object reveals: it is quadratic in \(x\). It exposes linear-feature reuse and quadratic interactions; it does not reveal how earlier layers constructed those features. That deeper hierarchy requires folding backward.

**I would therefore treat tensor factorization as a way to propose useful intermediates, followed by arithmetic-DAG search to organize and reuse them.** Token-support overlap helps characterize their consumers; algebraic dependencies establish the computational graph.

<a id="hierarchy-and-dag-discovery"></a>
## Follow-up: can we discover the hierarchy during fitting?

**Added 11 September, after reading Logan's browser-Codex addition above. Yes. Finding factors first and reorganizing them afterward is one practical strategy, not a mathematical requirement. We can make shared intermediate computations part of the object we fit. A sparse Tucker core is a useful candidate, but neither a prerequisite nor a complete solution.**

The browser response correctly distinguishes an arithmetic organization from a shared subspace. I would change its final recommendation from a strictly sequential pipeline to an **alternating search**: propose factors or groups, propose reusable computations, refit their numerical weights jointly, and reconsider the graph. Freezing an arbitrary factorization too early can hide a simpler organization.

### What does a hierarchy or DAG mean here?

There are at least three different objects we should distinguish:

| Object | What an edge means | What establishes it |
|---|---|---|
| Token grouping or hierarchy | Tokens share loadings, or one group is nested within another | Structure in the output coefficients; useful descriptive organization |
| Arithmetic DAG | A computed value is consumed by another operation | An executable expression with shared nodes and explicit dependencies |
| Hierarchy across model layers | An earlier computation constructs information consumed later | Actual upstream composition, interfaces, and behavioral/intervention evidence |

In a DAG—**directed acyclic graph**—a product can have two parents, and a shared value can have several consumers. With arrows pointing from inputs toward outputs, “several parents” describes an operation combining inputs; reuse is often **several outgoing edges from the same parent**. These are compatible and do not require disjoint groups or a tree.

For example:

$$
u=a+b,\qquad v=c+d,\qquad
q_1=uv,\qquad q_2=ue,\qquad
F=pq_1+rq_2.
$$

Here $u$ is computed once and feeds two products. The parents of $q_1$ are $u$ and $v$. The output combines both products. A further shared sum of quadratic nodes is also possible; it need not be stored as a separate node unless doing so helps reuse, price, or the intended intervention interface.

The layer is quadratic in its input $x$. Therefore our simplest degree-respecting grammar allows linear combinations of linear nodes, products of two linear nodes, and linear combinations of quadratic nodes. Repeated multiplication of nonconstant quadratic nodes would produce higher degrees that then need cancellation. We should not introduce that machinery by default. Linear computation can still have several reusable stages. How the network originally produced $a,b,c,d,e$ is a separate upstream question. Folding backward must retain RMS normalization, attention normalization, residual paths, and other real operations; the whole model cannot simply be declared a higher-degree polynomial.

### Dense Tucker can contain the answer without displaying it

Write a symmetric Tucker model as

$$
h=B^\top x,\qquad
s_g=\sum_{i,j}H_{gij}h_i h_j,\qquad
F=Cs.
$$

$B$ selects shared input features, $C$ selects shared output directions, and the core $H$ records their interactions. This is the standard factors-plus-core organization; CP is a more restricted organization. [Kolda–Bader survey](https://www.kolda.net/publication/koba09/).

**Yes, we can search for a DAG inside a dense Tucker core.** If the representation is exact, changing to Tucker coordinates does not lose information. But truncating the input or output spaces can remove the very relations we want to find. The cost of applying $B$ and $C$ also remains part of the program.

An important ambiguity is the choice of coordinates. For any invertible matrix $R$, replace $h$ by $h'=Rh$ and each quadratic matrix by

$$
H'_g=R^{-\top}H_gR^{-1}.
$$

The function is unchanged, although a sparse matrix can become dense. Output-basis changes also mix the slices. Thus, **a dense core in one basis does not demonstrate dense underlying computation**, and the basis must remain adjustable during structure discovery.

I executed a small exact control for this point. Starting from $(a+b)(c+d)$ and $(a+b)e$, I applied invertible input and output coordinate changes. Both resulting output slices contained all **15 possible quadratic monomials in five variables**. Exact symbolic factoring still recovered two products with the same linear parent, with zero polynomial reconstruction error. This shows that making the core sparse first is unnecessary in this example. It is a structured integer toy, not a scalable algorithm for noisy native weights, and it does not uniquely recover the original output basis. [Code](../../dense_core_dag_v1_control.py) · [Receipt](../../DENSE_CORE_DAG_V1_CONTROL.json).

### Sparse Tucker helps, but core sparsity is not arithmetic simplicity

A sparse-core objective rewards many $H_{gij}$ being zero. That can expose an interaction graph: which learned input features interact and which outputs consume them. Here “sparse Tucker” means sparsity in the **learned core**, not merely software designed for a sparse observed tensor.

However,

$$
(a+b)(c+d)=ac+ad+bc+bd
$$

has four nonzero expanded interactions but needs only one variable-by-variable multiplication after forming the two sums. A sparse-core score in the fixed coordinates $a,b,c,d$ counts four interactions; a learned reader/product program can expose one product. The reader additions and any learned constants must also be charged. The converse matters too: a sparse core may have little repeated arithmetic to share.

So we should compare several representations under the same reconstruction target and an explicit computation price:

| Representation | What it directly encourages | What remains to discover |
|---|---|---|
| Dense Tucker | Shared input/output spaces | Cheap structure inside the core and adapters |
| Sparse Tucker with learned bases | Few active feature-pair/output interactions | Factored sums and deeper reuse not captured by zero counts |
| CP or symmetric products | A bank of linear-by-linear products | Reuse within readers and output mixing |
| Output-sharing LL1 | Several interactions compute one output variable | Shared computations between different blocks |
| Explicit shared arithmetic graph | Reused linear features, products, and combinations | The graph topology and numerical parameters jointly |

“Hierarchical Tucker” has a specific tensor-decomposition meaning: a hierarchy of mode subspaces. Its name alone does not mean a semantic hierarchy or an arbitrary arithmetic DAG. Our original tensor has only three modes; splitting coordinate indices into more modes introduces a choice of tensorization that itself needs justification.

### A concrete way to search during fitting

For this last-layer problem, my preferred next formulation is an **explicit shared arithmetic graph fitted to the implicit weight tensor**. We already have an exact native product representation, so constructing a huge dense Tucker tensor first is unnecessary. Tucker coordinates or LL1 groups can provide alternative initializations or manageable subproblems.

One useful graph template is

$$
z_i=\sum_j B_{ij}x_j+\sum_{k<i}A_{ik}z_k,
$$

$$
q_\ell=
\left(\sum_i P_{\ell i}z_i\right)
\left(\sum_i Q_{\ell i}z_i\right),
\qquad
s_g=\sum_\ell V_{g\ell}q_\ell+\sum_{h<g}J_{gh}s_h,
\qquad \widehat F=Cs.
$$

The index restrictions make the graph acyclic. $z_i$ are linear intermediates; $q_\ell$ are products; $s_g$ are quadratic combinations. Sparse connections and explicit reuse are essential: if all these matrices are unrestricted and dense, this just hides complexity in extra layers. The graph can share a linear parent between many products and share a product or sum between many output groups. It need not force the CP and LL1 organizations to compete as disjoint partitions.

The conceptual objective is

$$
\min_{\mathcal G,\theta}
\frac{\|T-\widehat T(\mathcal G,\theta)\|_F^2}{\|T\|_F^2}
+\lambda\,\operatorname{Cost}(\mathcal G,\theta).
$$

$\mathcal G$ is the graph, and $\theta$ its constants. The norm covers the folded unembedding, using our existing implicit contractions. Cost counts stored constants, graph edges, additions, variable products, and execution/storage costs under stated weights. **A reused node is charged once to compute, with its uses and storage accounted for.** Node count alone would reward hiding an arbitrary dense matrix inside one node. We should also show the separate cost/error measurements, rather than obscure tradeoffs in one scalar score.

An implementable search would alternate these steps:

1. **Fit weights for a fixed graph.** Use the exact coefficient objective and gradients; eliminate linear output coefficients by least squares when its cost is manageable.
2. **Propose graph changes.** Merge proportional readers, replace approximately shared reader combinations with one parent, share product nodes, split or merge output groups, and try distributive factoring of suitable sums. Include proposals that change the basis, since exact duplicate detection alone will miss approximate learned relations.
3. **Refit and compare.** Score the changed graph against the full target and actual computation price. Approximate merges need a measured error budget; they are not algebraic identities.
4. **Keep competing organizations and test recovery.** Different graphs can compute the same function. Use planted examples with overlap, basis changes, and noise before treating failed native search as absent structure. After freezing candidates, validate on FineWeb and then the four circuit properties.

Equality saturation, as implemented by **egg**, is a relevant tool for retaining many equivalent expressions under supplied rewrite rules. It could handle the exact distributive-rewrite part. It does not by itself discover approximate real-valued weight identities or fit their coefficients; our extraction cost must also account for shared DAG nodes rather than charge repeated expression trees independently. This is a proposed use, not an installed native search we have already run. [Primary egg paper](https://arxiv.org/abs/2004.03082).

The hard parts are graph search, approximate sharing, nonunique bases, and honest computation pricing. A continuous sparsity penalty is a search aid, not a guarantee that we found the simplest graph. Algebraically equivalent whole functions can also assign different meanings to deleting an internal node. That is why extraction, selective removal, OOD prediction, and composition are still needed to identify useful circuits.

### Where this leaves the current experiments

The existing feature dictionary already learns a limited form of shared readers. CP, LL1, and the shared-input groups each search a restricted graph family. **We have not yet implemented the general joint hierarchy/DAG search described above.** The symbolic control demonstrates one exact route from a dense description to reuse; it does not close that gap.

While this discussion was underway, the native matched-budget pilot completed. At about 1.25 million parameters per arm, output-sharing LL1 captured **11.66% / 11.64%** of coefficient energy, versus **8.53% / 8.54%** for the shared-input family. All four fits reached their 120-second limits; these are optimization/cost pilots, not converged comparisons or circuit discoveries. LL1 passed the registered 10% capture bar; the shared-input family missed it. Whole-function agreement between starts was 0.901 for LL1 and 0.864 for shared input, which does not establish agreement of individual groups. [Completed pilot receipt](../../MATCHED_SHARED_GROUPS_V1_RESULT.json).

**Recommendation for browser Codex:** consider native products and symmetric LL1 as initial graph proposals, keep the input/output bases adjustable, and search for shared arithmetic while refitting to weights. Compare sparse-core Tucker as one structural hypothesis. Do not require sparsification before DAG discovery, and do not assume a final pass over frozen factors can recover every useful organization. The immediate method question is which graph moves expose shared input and quadratic nodes reliably, with acceptable error and literal cost, before any data-guided fitting.

### Clarification at 17:32: what has now been implemented?

The recommendation above still applies. We now have a **restricted shared graph**
whose shared input readers, private input readers, output directions, and quadratic
cores are jointly fitted to the weights. Its topology is proposed before that
numerical fit. We have not implemented the general alternating search over graph
topologies, reusable sums, and distributive rewrites described above. In particular,
jointly optimizing weights on a fixed graph is not the same as discovering arbitrary
hierarchy during optimization.

An exact analysis now finds that 10 of the fitted spectral graph's 12 shared readers
need at least two distinct output/partner branches to retain 95% of their own
coefficient energy. This makes the meaning of branching more concrete: most do not
collapse to one product merely because we change their output basis. It is still
conditional on the selected reader and graph, and is not evidence of two semantic
tasks. [Derivation, results, and frozen behavioral-screen protocol](../../SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md).

For choosing the next representation, **dense Tucker is a possible coordinate
system, sparse Tucker is a structural hypothesis, and an arithmetic DAG is the
executable object we ultimately want to search.** There is no requirement to pass
through them in that order. The essential additional search operations are to
change the basis, expose common factors or sums, share them, and jointly refit the
changed graph against the original folded weight tensor. Each approximate rewrite
must earn its error/cost tradeoff; each claimed circuit must subsequently earn its
behavioral interpretation.

### Clarification at 19:21: graph discovery need not wait for factorization to finish

The practical recommendation is to **alternate representation fitting and graph changes**. Start from native products, LL1 groups, or Tucker coordinates; propose a common reader, product, or sum; refit the affected computations against the original weight tensor; then retain the change only if its error and literal execution cost justify it. Sparse Tucker is one competing assumption, not a required intermediate stage. An exact dense Tucker representation preserves the function, whereas a truncated one can discard relations before the graph search sees them.

Our implementation has now gone beyond fitting a completely fixed topology in one narrow way: it tested adding the existing shared reader `parent1` to a third consumer. The matched refits included every affected consumer and allowed their shared/private readers, cores, and output directions within a common output span to move. Both local optimizations converged, but the added edge cost approximately $2.51\times10^{-5}$ in normalized objective, above the registered $10^{-6}$ allowance, despite saving 1,137 stored numbers. We therefore did not adopt that edge. This is a completed local graph-edit test, not a general topology-search algorithm or evidence against other DAGs. [Result and scope](../../RESIDUAL_PARENT_EDGE_V1_MATH.md).

The missing broader search includes shared sums inside readers, reusable quadratic combinations, and distributive rewrites that change which products are computed. Also distinguish **finding a cheaper equivalent program** from **recovering the model's unique internal hierarchy**: equivalent expressions can define different internal deletions. The four behavioral properties are needed to decide which proposed intermediate computations deserve a circuit interpretation.

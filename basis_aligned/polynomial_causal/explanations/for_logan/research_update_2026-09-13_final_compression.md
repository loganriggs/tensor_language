# Final instance update: compression of composed circuit interactions

13 September 2026. Covers the work since the [02:55 full update](research_update_2026-09-13_0255_joint_attention_predictor.md), through the 14:34 research checkpoint. Written before the instance expires. [Next-session instructions](../../../../CODEX_RESEARCH_SESSION_STARTUP.md).

## 1. What happened, at a high level

**Compression of a composition helped. We have a modest, behaviorally tested storage improvement and several exact simplifications. We do not yet have a substantially smaller or faster complete circuit.**

Your proposal was to compress interactions or retained terms, allowing intermediate computations to be shared. We investigated exactly three settings: the recent layer9–10 interaction, the recent last-attention/last-bilinear interaction, and the full unembedding folded into the last bilinear layer as an unsupervised control.

The most useful sequence was:

1. **Use how the inputs are produced.** The layer9–10 interaction originally appeared to need ten quadratic products. The actual upstream response restricts those inputs, allowing six products, then five for a particular same-input term. One output vector can be shared across contexts. A further three-term approximation passed conditional behavioral checks.
2. **Check whether the simplified term matters.** Removing that small cross interaction from the ordinary model barely changed the spelling capability. Its larger upstream parent and remainder mattered much more. We therefore moved compression to that larger parent rather than treating excellent prediction of a tiny term as a complete explanation.
3. **Fit the computation, not just its internal coordinates.** A sparse reader fitted to reproduce a basis failed broader text checks. Optimizing the same-sized sparse reader against the complete product of the two attention scores worked substantially better. The best candidate passes all five corpus-average preservation tests, including 96 additional prefixes, and preserves tested joint removals and donor swaps.
4. **Price the executable result.** The best sparse parent removes 25% of reader entries, but retains all 64 reader directions. After correction matrices and the rest of the interface are charged, the saving is **1.806% of that parent interface**. Actual sparse CPU execution is slower and uses more memory. A portable package reproduces the validated candidate, but still needs model-generated inputs.

This supports your intuition that compositions can expose structure missed by independently compressing their parts. The current evidence supports local arithmetic simplification and a small storage saving. It does not establish a general discovery algorithm for sparse semantic circuits.

## 2. The three settings and their present verdicts

| Setting | What we compressed | Best supported result | Important limit |
|---|---|---|---|
| Layer9–10 interaction, then its larger head9.8 parent | Generated bilinear response products; shared parent reader used by both QK scores | Exact product reuse; a 25%-entry-pruned parent passes corpus-average and composition checks | Only 1.806% parent-interface storage saving; no speed gain; individual failures |
| Head17.2 → MLP17 → selected output readers | The mixed residual/head interaction tensor and shared output features | About 30.81% less packed storage than the dense folded tensor; 2.72–8.12% error relative to the retained term’s own effect | Broader full-head inputs fail; sparse CPU execution loses; essentially no node sparsity |
| Full unembedding → MLP17 | Shared quadratic functions plus group-specific functions for tokens | Shared/private structure improves on matched-storage global low-rank baselines | Large errors, incomplete convergence, failed behavioral preservation; no adopted replacement |

Head and layer labels here use the repository’s zero-based numbering: head17.2 is head index 2 in the final block. “QK” means the query–key score calculation. This model multiplies **two** such scores; it does not use ordinary softmax attention.

## 3. The strongest new compression: a shared sparse parent reader

### What is the parent?

The larger head9.8 computation has a scalar output amplitude, written into the residual stream along a fixed vector. Earlier work split that amplitude into a child and a remainder:

$$
p=c+r.
$$

Here a parent is an explicitly defined computation containing the child, not a semantic label inferred from a clustering plot. The child and remainder share its writer. Removing the larger parent/remainder damaged roughly 21–36% of the native paired spelling capability on the tested panel; the tiny downstream cross interaction was much less consequential. These are different intervention objects and should not be conflated.

The parent is defined using an **even part** of the attention calculation: average a calculation with the same calculation after reflecting a chosen source-input subspace. “Even” refers to invariance under that sign reversal, not to positive weights.

Let the residual width be $d=1152$, and let $B\in\mathbb R^{1152\times64}$ have orthonormal columns. Its projector and reflection are

$$
P_B=BB^\top,\qquad H_B=I-2P_B.
$$

$P_B y$ extracts the selected component of a source vector $y$; $H_B y$ reverses that component and leaves the orthogonal complement alone.

For one query/source position pair, write the two attention scores as $x^\top M_1y$ and $x^\top M_2y$, where $x,y\in\mathbb R^{1152}$ and the position-dependent matrices $M_i$ incorporate the query/key weights and rotary position transformation. The even numerator is

$$
F_B(x,y)=\frac12\left[
(x^\top M_1y)(x^\top M_2y)
+(x^\top M_1H_By)(x^\top M_2H_By)
\right].
$$

This formula shows why fitting QK1 and QK2 independently is not the right objective: both scores participate in the same product. The complete native parent additionally uses values, normalization, source accumulation and its fixed writer. Those parts remain in the execution and price; the fitting objective below covers the displayed numerator.

### What changed in the representation?

We replaced $B$ with a sparse, full-column-rank matrix $S\in\mathbb R^{1152\times64}$. Its columns are not necessarily orthogonal. The correct projector is therefore

$$
P_S=S(S^\top S)^{-1}S^\top,\qquad H_S=I-2P_S.
$$

Using $SS^\top$ without the correction would implement a different and generally invalid reflection. The correction must also be charged in storage and execution.

The successful mask retains 55,296 of 73,728 entries: **25% edge pruning, zero reader-node pruning**. A mask specifies which coefficients may be nonzero. We retain both QK consumers and share the sparse source projection between them.

First we rotated the original basis and fitted sparse entries to minimize basis reconstruction error. Conditional updates used hard thresholding and an orthogonal Procrustes solve: keep the allowed largest entries, then solve for the rotation that best aligns the original basis with the sparse one. Ten starts reached the declared local stopping condition quickly. This improved basis fit but still failed a legal-text preservation test at 16.52% error.

The better fit changed the objective to the coefficient error of **$F_S-F_B$**, including both repeated query slots and both repeated source slots. We evaluated that polynomial coefficient norm through small Gram-matrix contractions, without materializing a $1152^4$ tensor. This was weight-only fitting: no text examples, token probabilities or loss gradients trained $S$.

Two fixed masks and five perturbation sizes gave ten starts. Each used a limited-memory quasi-Newton optimizer, L-BFGS, with a line search and normalized columns. All ten met the registered local stationary criterion in about **19 seconds total**. The best mask had two distinguishable solution basins across the campaign; this is evidence of adequate local optimization for these starts, not proof of a globally best sparse graph. Explicit small-tensor and finite-difference controls checked the loss and its gradients.

The best composed objective was 41.60% lower than the rotated-basis starting candidate’s objective. That number is a reduction in squared coefficient loss, **not** the percentage of circuit behavior explained. New position-lag checks improved too, but do not replace native execution.

### Did the frozen candidate preserve behavior?

We compared the full model after the original parent intervention with the full model after the compressed parent intervention. The model’s downstream computation was actually rerun. For corpus tests, we examined all 50,304 output logits, rather than only a chosen spelling pair.

An effect is the difference between the edited model and the ordinary model. Relative preservation error divides the discrepancy between compressed and exact effects by the size of the exact effect. The probability-weighted version emphasizes logits that the unedited model gives more probability. It is a **validation metric**, not data used for fitting. KL divergence separately measures discrepancy between the resulting probability distributions.

On 40 development prefixes, the five corpus-average weighted errors were 3.60–5.06%, all below the 10% bar. The decisive legal-text error fell from 16.52% to 4.02%.

We then froze the candidate and tested 96 additional, distinct 128-token prefixes:

| Corpus | Prefixes | Weighted effect preservation error |
|---|---:|---:|
| FineWeb | 32 | 5.07% |
| Discussion | 16 | 2.69% |
| Reference | 16 | 2.36% |
| Biomedical | 16 | 3.10% |
| Legal/patent | 16 | 3.13% |

FineWeb is the model’s training corpus. The other four domains come from the existing Pile panels and are labeled OOD checks. The 96 prefixes have distinct exact token hashes from the development 40; they come from historical project caches and are **not** claimed to be historically untouched, near-duplicate-free, or disjoint from all pretraining text.

Removing any one document still left the corpus-average error below 5.65%. However, **12 of 96 individual prefixes exceed 10% error**. The largest relative failures have very small original effects, but that does not explain every failure. We retain them in the receipt rather than declaring uniform preservation.

Signed parent/remainder interventions, including negative and doubled strengths, preserve aggregate target/control effects to about 2%. Two small control effects change sign, so the strict all-signs criterion **fails**. Their reference effects are approximately 0.00059 and 0.00016 nats. A nat is a natural-log probability unit; small magnitudes contextualize these misses but do not erase them.

### Does it still compose and support removal/swapping?

We retained the exact child and defined the compressed remainder by $\widehat r=\widehat p-c$. We tested seven joint strengths, including opposite signs and doubled branches:

$$
\Delta h=-(a c+b r)w,\qquad
\widehat{\Delta h}=-(a c+b\widehat r)w.
$$

$w\in\mathbb R^{1152}$ is their shared writer. The actual combined edit was propagated through the model. We did not add separately measured output effects, which would ignore downstream nonlinear interactions. Across 48 lexical prompts, all registered aggregate composition bars pass; the largest weighted error is 2.19%. Child-only outputs match exactly. Seven nontrivial prompt/strength cells exceed 10%, so this too is an aggregate result.

We also installed the compressed reader in the existing selective head8.2/head9.8 producer pair. The second head receives the context actually changed by the first intervention. On the historical 72-row regional and 32-row newline-control panel, the registered removal, donor-transfer, capability and control bars pass. Target effect preservation error is at most 1.57%, and control error at most 2.07%. There is one small donor-control sign reversal in the additional audit. The old broader full-sector newline failures remain part of the project history; this selected-pair result does not overturn them.

These checks support reuse of this particular compressed parent inside tested existing programs. They do not establish that arbitrary independently discovered semantic circuits can now be combined.

### What did we actually save?

At a common FP32 precision, the reader representation decreases from 294,912 to 246,800 bytes after charging mask, values, correction and shape: **16.31% locally**. Charging the rest of the declared parent interface reduces the saving to **1.806%**. No whole-model percentage is claimed.

Actual compressed-sparse-row execution is **1.9–4.4 times slower** than a dense implementation that already shares its source projection across both QK scores. It also uses more resident memory because sparse indices and execution layout cost space. A prepacked-layout countercheck did not rescue speed. This is a CPU result; no GPU speedup has been demonstrated.

The [portable package](../../extracted_circuits/sparse_even_key_producers_8_2_9_8_v1/README.md) contains a roughly 5.97 MB weights file and a standalone Torch executor. Native score arrays exactly match the validated compressed implementation, and isolated loading without project imports passes. At load time it reconstructs a corrected dense basis: the package is a storage/export result, not a sparse fast kernel. Native normalized contexts, token positions and the surrounding model remain required.

[Full derivation and primary receipt index](../../DIRECT_PARENT_KEY_COMPRESSION_V1_MATH.md) · [Additional corpus result](../../COMPOSED_SPARSE_NEW_DOCS_V1_RESULT.json) · [Composition](../../SPARSE_HIERARCHY_COMPOSITION_V1_RESULT.json) · [Selective pair](../../SPARSE_SELECTIVE_PAIR_V1_RESULT.json).

## 4. Other compression attempts, and why they were not adopted

**More regular sparsity:** keeping exactly two coefficients out of each group of four would be more hardware-friendly and save about 4.58% of the parent interface. Ten-start fitting and a learned-mask/refit follow-up made real coefficient improvements, with converged promoted fits. Nevertheless the improved mask failed native FineWeb and legal preservation at 12.30% and 39.11%. Legal failure persists when any one document is removed. The candidate fails; this is not proof that all regular sparse representations must fail.

**A separately compressed upstream composition:** ten converged starts still missed the coefficient target. More fundamentally, an apparent 10% saving assumed its original producer could be removed. Other consumers still needed that producer. Keeping both would increase joint storage by 82.29%. This is a concrete example of why reusability must be priced over the whole surviving graph. [Receipt and analysis](../../COMPOSED_READER_REFIT_V1_MATH.md).

**The layer9–10 response algebra:** the upstream perturbation follows a constrained rational curve because a bilinear numerator is divided by an input-dependent RMS normalization. Respecting that curve reduced ten symmetric downstream products to six, then five for its same-input quadratic part. One fixed vector is shared across contexts; a three-term approximation passed the 160-prefix historical panel and 48 new lexical prompts. Its own-component errors on the fresh panel were 0.41–1.05% for targets and 1.09–2.97% for controls. Product preparation can be faster and intermediate banks smaller, but full-branch execution still loses to direct evaluation. [Explained three-term result](../../CONIC_AMPLITUDE_POLYNOMIAL_V1_MATH.md) · [Shared term and native relevance](../../SHARED_POLYNOMIAL_TAIL_V1_MATH.md).

**Setting 2, last-attention interaction:** with $z\in\mathbb R^{1152}$ the background residual and $a\in\mathbb R^{128}$ the retained head coordinates, the selected-output mixed term is

$$
M(z,a)=UD_{17}\left[(L_{17}z)\odot(R_{17}Wa)
+(R_{17}z)\odot(L_{17}Wa)\right].
$$

Here $W\in\mathbb R^{1152\times128}$ writes head coordinates to the residual stream; $L_{17},R_{17}\in\mathbb R^{4608\times1152}$ are bilinear input readers; $D_{17}\in\mathbb R^{1152\times4608}$ writes their products; and the selected $U\in\mathbb R^{12\times1152}$ reads twelve output coordinates. Folding gives a tensor of shape $12\times1152\times128$. This particular compression is of the **mixed numerator**, not a complete fitted unembedding→MLP17→QK/value path with all dependencies removed.

Sparse output/head coordinates save 30.81% against the dense folded tensor and preserve its retained regional contribution within 2.72–8.12%. Nearly every input product and all reader nodes remain active. On broader native full-head inputs, aggregate errors rise to 10.24% on regional text and 20.51% on FineWeb. Matched-output and non-tiny-effect checks preserve the failure. Shared output-subspace fits, balanced common/difference objectives and private corrections did not produce a better adopted result. A good overall coefficient fit can preferentially preserve common token writes while missing spelling contrasts. [Scope and counterchecks](../../SPARSE_INTERACTION_EXECUTOR_V1_MATH.md) · [Shared-output fitting](../../INTERACTION_SHARED_WRITES_V1_MATH.md).

**Setting 3, full unembedding:** we fitted global shared quadratic functions plus token-group-private functions. At two budgets, coefficient errors were 83.61% and 74.41%, versus 85.85% and 77.74% for matched-storage global baselines. Both improve, but all promoted fits remain unconverged and native preservation fails. Some impressively accurate token rows were duplicates; a separate bound shows at least 93.97% of the overall improvement comes from outside that duplicate subset. Thus the modest representation gain is real, while useful vocabulary-wide factorization remains unresolved. Both capacities together took about 37 minutes of measured fitting time. [Full result and red-team](../../FULLU_SHARED_LOCAL_FIT_V1_MATH.md).

## 5. Did the math reviews help?

Yes, particularly when the mathematics changed the object being compressed.

- **Generated-input identities** exposed reuse that unrestricted tensor decompositions miss. They produced exact smaller product banks and motivated the successful three-term conditional approximation.
- **Joint two-QK coefficient mathematics** made it cheap to optimize a sparse reader for the actual even attention numerator. This is the clearest practical gain from changing the objective: broader preservation improved at the same storage budget.
- **Literal joint pricing** rejected a misleading standalone compression before adopting an extra dependency.
- **An exact coordinate-chart calculation** in the final math review removed a small redundant reader block without changing the subspace. It saved about 1.99% of prepared FP32 state but ran slower. This is a useful engineering null, not a new semantic decomposition.

Other reviews were informative negatives: output-subspace lower bounds, shared-source product spectra, and a joint query/key/value moment calculation did not reveal a cheap large-rank collapse. None proves the absence of a sparse arithmetic graph with a different structure.

The latest hourly review was 14:28 UTC; the latest three-hour mathematical review was 14:29. Their next deadlines were 15:28 and 17:29 on this instance. The hourly review also found too much publication overhead and inaccurate phase labeling. The next session should record routine checks as receipts plus short board entries, and reserve fuller narrative for hourly/major/user-requested updates.

## 6. What the next session should do

**Resume setting 2 before expanding the successful parent’s validation indefinitely.** Its current sparse tensor was fitted on unrestricted residual/head inputs. The best setting1 result suggests a concrete question: does preserving the operator on inputs generated by the retained three QK/value contractions expose a smaller or more faithful shared graph?

First inspect the existing head17.2 and MLP17 dossiers and the shared-output failures. Then derive and test a weights-only contraction-error objective for that actual retained producer, including the mixed contributions of both QK factors and values. Keep the native normalization boundary explicit. Compare against the existing sparse operator at matched total cost. Start with an exact small/native-weight CPU control; only launch a fit if the objective and dependency price are valid. **This new fit has not been implemented or queued.** The instance-expiry request interrupted orientation to it.

The second priority is an efficient implementation of the successful sparse parent only if hardware or a changed representation gives a plausible reason to beat shared dense execution. Repeating the failed irregular CPU kernel is low priority. The third setting remains an open control, but its unfinished large fits should not consume the first new-session block without a specific convergence or representation improvement.

Keep normalized Frobenius tensor similarity and a search over proposed arithmetic graphs as Logan’s deferred idea, to use when it fits the situation well. Do not let a large new graph-search framework displace these cheaper discriminating tests.

The project’s four desired properties remain the goal: OOD prediction, extraction, selective removal, and composition/reuse. We have improved conditional evidence for a small compressed interface. Producing a substantially smaller, independently executable and generally reusable model remains unfinished.

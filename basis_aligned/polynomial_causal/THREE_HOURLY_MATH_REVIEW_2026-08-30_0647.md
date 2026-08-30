# Three-hour mathematical review — 2026-08-30 06:47 UTC

## Bottom line

The newest useful mathematical move is **not another low-rank fit**.  It is to ask
whether a proposed structured tensor program has the right number of *observable*
degrees of freedom after its exact symmetries are removed, and whether its named
components can therefore be identified and edited separately.

This review implements that question as a gauge-quotiented Jacobian-rank and
algebraic-matroid toy.  The code detects the expected gauge nullspace in regular matrix
and CP factorizations, stays invariant under exact regauging, and rejects a deliberately
duplicated/non-identifiable CP decomposition.  It also selects a minimal local set of
physical tensor entries whose differentials span all other observable differentials.

The next two mathematical moves are:

1. a dependent-input functional ANOVA of the MLP0 token/context map, which can make
   “mean + token + context + interaction” a distribution-relative canonical split;
2. compressed simultaneous perturbations of the newly lawful signed causal-response
   interface, which can reduce GPU forwards if the source-to-target causal Jacobian is
   actually sparse.

Neither is allowed to earn credit from local MSE alone.  Their gates are crossed-pair
prediction, held-out intervention prediction, selective removal, and literal program
price.

## Current evidence, jobs, and blockers

The committed head at inspection was `07e8e443`.  The latest completed Codex artifacts
are:

- a lawful additive response-cell schema that retains signed and absolute member/off
  sums plus counts;
- a frozen source-document split: 1,000 rows from 688 source documents, divided into
  343 FIT and 345 EVAL documents with zero overlap;
- exact sparse Möbius machinery and its dense failure control;
- the geometry-to-causality counterexample at M16.

The strict whole-project balance has not moved:

- certified stored model content: **5.348245316%**;
- named causal CE: **10.923302467%**;
- unexplained CE: **4.72714 nat = 89.076697533%**;
- terminal actions jointly passing extraction, selective removal, and OOD: **0/68**.

The six-component substrate census finished while this review was running, after 879
seconds, but its artifact and logs were still uncommitted concurrent work and are not
staged here.  Its registered provisional summary is:

- a dominant shared direction appears in 3/6 components, missing the registered
  majority bar of 4/6;
- the “remove the common direction and reveal selective children” reversal remains
  unique to A8;
- shared variance and mean absolute cosine correlate at 0.9961, confirming internal
  consistency of the geometry instrument, not causal meaning.

The GPU was idle after that job.  The blocker to causal BTD remains scientific rather
than computational: the signed response tensor has a schema and a leakage-free split,
but its model outcomes have not yet been collected.  Existing concentration ratios
cannot be factorized lawfully because ratios are nonadditive and discard sign, scale,
counts, and document variation.

## Ranked move 1 — quotient-Jacobian rank and an algebraic-matroid observation basis

### Exact bilin18 object

Apply this to any proposed structured replacement before calling its atoms “circuits”:

- the proposed mean/class/token/context MLP0 program;
- a fitted shared/private CP or block-term response model;
- a hierarchical/DAG tensor program;
- a tensor-similarity fit of an MLP's folded third-order tensor.

Let its parameters be \(\theta\), its exact gauge group be \(G\), and let

\[
\Phi(\theta)
\]

be **physical observables**, not internal coordinates.  Suitable observables are folded
polynomial coefficients, exact tensor entries, or the lawful signed causal-response
cells.  Compute

\[
J_\Phi(\theta)=\frac{\partial \Phi}{\partial\theta}.
\]

Every tangent to the exact gauge orbit must lie in \(\ker J_\Phi\).  At a regular point,
\(\operatorname{rank}J_\Phi\) is the local dimension of the represented physical
family.  If the nullspace is larger than the known gauge-orbit tangent space, the
candidate contains extra locally unidentifiable directions: two purported components
can trade content without changing the physical program.

For the matrix product \(W=AB\), with \(A\in\mathbb R^{m\times r}\) and
\(B\in\mathbb R^{r\times n}\),

\[
(A,B)\mapsto (AG,G^{-1}B)
\]

is an \(r^2\)-dimensional gauge.  At full rank, the expected image dimension is

\[
r(m+n)-r^2=r(m+n-r).
\]

For a generic rank-\(R\) third-order CP program, raw factor count is
\(R(I+J+K)\), while two continuous rescalings per component are gauge.  The expected
local dimension is therefore

\[
R(I+J+K-2),
\]

when this does not exceed the ambient tensor dimension and the point is regular.  CP
permutation is discrete, so it does not reduce local dimension.  The distinction
between generic and specific identifiability is important; a practical primary source
is Chiantini, Ottaviani, and Vannieuwenhoven,
[An Algorithm for Generic and Low-Rank Specific Identifiability of Complex
Tensors](https://doi.org/10.1137/140961389).

The independent rows of \(J_\Phi\) form a local algebraic-matroid basis.  Selecting a
row basis chooses physical observable coordinates whose first-order changes determine
all the others.  This gives a principled experiment-selection target after grouping
coordinates by their actual GPU acquisition cost.

### Assumptions that may fail

- Jacobian rank is local.  It does not prove global uniqueness; disconnected or
  discrete alternative decompositions can remain.
- Rank can fall at singular parameter points, including duplicated or zero atoms.
- A finite collection of causal cells may omit a downstream consumer, so
  identifiability relative to that collection is not whole-model identifiability.
- Near-zero non-gauge singular values make an edit statistically unstable even if the
  exact rank is full.
- A row basis ignores grouped acquisition cost: one bilin18 intervention returns many
  response coordinates together.  A real design must select groups, not pretend cells
  cost independently.

### Prediction beyond reconstruction

If the test passes, then after quotienting the known gauge:

- independently named atoms have locally distinguishable physical effects;
- non-gauge parameter changes predict distinct folded coefficients or signed causal
  responses;
- a selected observable basis predicts every other small response perturbation;
- removing a named atom has an invariant physical meaning rather than depending on an
  arbitrary factor rotation.

If extra null directions remain, selective component extraction/removal is not well
defined, even when tensor reconstruction is exact.  That is a direct test of whether a
“simpler” decomposition is useful.

### Cheapest falsifying experiment

Run randomized JVP/VJP rank estimation on the next fitted shared/private response
program and explicitly generate every known gauge tangent.  Reject the structural
story if:

1. a known gauge tangent changes a physical observable;
2. nullity exceeds the measured gauge-orbit dimension stably across tolerances;
3. the selected observable basis fails to predict held-out small perturbations;
4. the rank or basis changes materially when documents double.

This is CPU work once a fit exists.  Only after it passes should any factor receive a
semantic label or removal test.

### Executed known-answer result

`quotient_jacobian_minimality.py` and five tests now pass:

- matrix product: 45 raw parameters, gauge dimension 9, measured/expected physical
  dimension 36, and maximum gauge-image magnitude \(1.98\times10^{-16}\);
- regular CP: 45 raw parameters, gauge dimension 6, measured/expected physical
  dimension 39, and maximum gauge-image magnitude \(2.75\times10^{-16}\);
- an exact CP regauge changes the tensor by at most \(2.22\times10^{-16}\) and leaves
  rank 39 unchanged;
- duplicating one CP component lowers rank from 39 to 26, correctly detecting a
  singular, non-identifiable representation;
- 39 selected physical entries span all 120 tensor-entry differentials, with relative
  first-order replay error \(1.66\times10^{-15}\).

The receipt is `quotient_jacobian_minimality_toy_receipt.json`.  It is explicitly a CPU
algebra gate, not a bilin18 identifiability or simplicity result.

## Ranked move 2 — dependent-input functional ANOVA for MLP0 token and context

### Exact bilin18 object

Let \(T\) denote token identity/embedding, let \(C\) denote the attention-derived
continuous context at layer 0, and let \(F(T,C)\) be MLP0's folded residual-stream
write.  Seek

\[
F(T,C)=\mu+F_T(T)+F_C(C)+F_{TC}(T,C).
\]

The existing TT/X/CC polynomial expansion is algebraically exact, but its terms need
not be orthogonal or distribution-canonical; indeed its CE pair interactions are
large.  Ordinary independent-input ANOVA is also invalid because tokens and contexts
are strongly dependent in language.

The relevant established object is a **hierarchically orthogonal functional
decomposition** for dependent inputs.  Under conditions on the joint distribution, it
chooses the terms uniquely by zero-mean and hierarchical orthogonality constraints.
Chastaing, Gamboa, and Prieur prove existence/uniqueness under bounded-density
conditions and develop estimators in
[Generalized Hoeffding–Sobol Decomposition for Dependent
Variables](https://doi.org/10.1214/12-EJS749).

For vector-valued MLP writes, use the empirical downstream/Fisher inner product rather
than treating every residual coordinate equally.  After this split, factor only the
token main effect \(F_T\) into shared lexical atoms plus token residuals; use a separate
low-rank or sparse-routed model for \(F_{TC}\).  This is the requested mixture of
structures rather than forcing all MLP0 content into one SAE or one Tucker core.

### Assumptions that may fail

- Natural text may lack overlap: for many tokens, only a narrow context distribution
  exists.  Density/positivity conditions can fail and make token versus context
  effects non-identifiable.
- Empirical conditional-expectation fits can leak document identity or memorize rare
  tokens.
- Orthogonality is relative to a chosen distribution and metric.  It may change under
  code, arithmetic, or synthetic OOD text.
- Orthogonal output components need not have additive CE effects through the nonlinear
  suffix.

### Prediction beyond reconstruction

- \(F_T\) should transport across held-out contexts for the same token;
- \(F_C\) should transport across matched token replacements;
- \(F_{TC}\) should be required specifically when both token and context are changed;
- a lexical atom learned inside \(F_T\) should support class-level extraction/removal
  with less collateral than a flat activation SAE atom;
- the terms and their downstream effects should remain stable when source documents
  double and on registered token/context swaps.

### Cheapest falsifying experiment

Use already cached MLP0 token/context/write rows with the frozen source-document split.
Fit the four-term decomposition only on FIT documents.  On EVAL documents, test:

1. empirical hierarchical orthogonality;
2. native held-out write and downstream CE;
3. token-context crossed-pair prediction;
4. matched removal of each term versus a random equal-energy direction;
5. stability when the FIT documents are halved/doubled.

Reject the canonical story if crossed-pair error is no better than the existing exact
TT/X/CC proposal at matched price, or if the split changes materially under a source
resample.

## Ranked move 3 — compressed simultaneous perturbation of the signed causal Jacobian

### Exact bilin18 object

Let \(\alpha\in\mathbb R^p\) scale \(p\) registered source interventions, and let
\(F_t(\alpha)\) be a signed target-circuit response on a fixed source document.  The
local causal-response operator is

\[
J_{t,s}=\left.\frac{\partial F_t}{\partial\alpha_s}\right|_{\alpha=0}.
\]

For a Rademacher mixture \(\omega_k\), a central difference gives

\[
\frac{F_t(h\omega_k)-F_t(-h\omega_k)}{2h}
=\omega_k^\top J_{t,:}+O(h^2).
\]

One mixed forward pair yields this measurement for every target \(t\).  If each target
depends on only \(s\) of the \(p\) sources and the random design has an appropriate
restricted-isometry/incoherence property, sparse recovery can replace \(2p\) separate
central-difference forwards by roughly \(O(s\log(p/s))\) mixed forwards.  Borkar,
Dwaracherla, and Sahasrabudhe analyze this combination of simultaneous perturbation and
compressed sensing in
[Gradient Estimation with Simultaneous Perturbation and Compressive
Sensing](https://jmlr.csail.mit.edu/papers/v18/15-592.html).

This is not the previous sparse Möbius experiment.  Möbius coefficients describe
finite subset interactions.  Here the first-order signed Jacobian is recovered first;
its support then chooses which pairs/triples deserve finite Möbius measurements.

### Assumptions that may fail

- Source effects may be dense; the six-component geometry census already warns that
  shared services are common.
- A small tangent effect can become large under full deletion, and vice versa.
- Large \(h\) introduces nonlinear interaction bias; small \(h\) can be lost in bf16
  or sampling noise.
- Simultaneously perturbing different layers may create off-manifold intermediate
  states.  Central-difference stability across several \(h\) values is mandatory.
- Shared supports across documents/targets are not guaranteed.

### Prediction beyond reconstruction

The recovered operator must predict:

- held-out random intervention mixtures;
- individually measured source-to-target signed effects;
- which finite source pairs exhibit nonzero Möbius interaction;
- which source removals are target-specific versus shared collateral;
- support stability across FIT/EVAL documents and OOD domains.

### Cheapest falsifying experiment

First run a CPU polynomial toy with sparse and dense Jacobian controls plus a quadratic
contamination sweep in \(h\).  If that passes, preregister 12–16 Rademacher mixtures
over the 49 currently supported source circuits, use the frozen FIT documents for
recovery, and reserve EVAL documents plus several individual sources for falsification.
Stop if held-out mixture error or support instability is high; then collect ordinary
source rows instead of forcing sparsity.

## Why tensor-cross is not in the top three

Cross/pseudoskeleton approximation is mathematically attractive: selected rows and
columns can recover or approximate a low-rank matrix, and TT-cross extends this to
multidimensional arrays.  See Goreinov, Tyrtyshnikov, and Zamarashkin,
[A Theory of Pseudoskeleton Approximations](https://doi.org/10.1016/S0024-3795(96)00301-1),
and Oseledets and Tyrtyshnikov,
[TT-Cross Approximation for Multidimensional
Arrays](https://doi.org/10.1016/j.laa.2009.07.024).

But the current causal acquisition cost is asymmetric.  One source intervention
already returns almost an entire target/document response row; requesting a target
column means running all source interventions.  Naive cross approximation therefore
does **not** reduce the expensive source-forward count.  It becomes useful only if we
obtain cheap side information for unseen source rows or change the interface so
individual tensor fibers are genuinely queryable.  The mixed-perturbation Jacobian
above matches the actual forward-cost geometry better.

## Reconsideration and pruning of the requested mathematics

| Mathematical family | Decision now |
|---|---|
| Tensor and arithmetic-circuit rank | Promote only quotient-Jacobian image dimension and exact executable operation count. Raw MLP polarization rank is smoothly full and already pruned. |
| Simultaneous factorization/shared dictionaries | Wait for lawful signed response cells; then require held-out cell and edit prediction. Existing ratio matrices are not additive. |
| Polynomial invariant theory/gauge quotients | **Promoted and executed** through explicit gauge tangents, quotient rank, and physical-observable row bases. |
| Algebraic complexity | Use quotient degrees plus literal multiply/add/storage price. Neither alone proves causal usefulness. |
| System identification/minimal realizations | Finite-Hankel triage was ranked in the 06:10 review and is not relabeled new. It remains appropriate for parity/successor, not generic MLP0. |
| Hankel/automata methods | Retain as a circuit-family classifier; copying and bracket depth may show rank growth rather than a small finite state. |
| MDL/prequential coding | Retain as price/generalization tie-breaker after causal validity. It cannot rescue a non-identifiable or noncomposable factorization. |
| Causal abstraction/bisimulation | The projected-abstraction toy already exists. Apply only after lawful response collection and hidden-consumer tests. |
| Blackwell/Le Cam deficiency | Already proposed on 2026-08-28, so it is not counted as new. It remains a strong eventual task-universal sufficiency criterion once actions/outcomes are rich enough. |
| Information bottleneck | Generic IB is still pruned: mutual information can discard edit-relevant structure and does not price an executable tensor program. |
| Sparse program synthesis | Promote only the falsifiable sparse signed-Jacobian group test; no unconstrained search over semantic programs. |
| Approximation certificates | Global Lipschitz products remain likely vacuous near RMSNorm. Quotient rank is an exact structural certificate, but finite CE/OOD consequences remain separate gates. |
| Norm minimization before HOSVD | Closed as a conditioning tool: the exact prior toy showed gauge balancing, while the real copy edge gained only a small amount. It is not semantic identification. |

## Priority and next action

1. **Apply quotient-Jacobian rank to the first fitted lawful shared/private response
   model.**  It is the cheapest way to prevent a mathematically exact but uneditable
   decomposition from being mistaken for circuits.
2. **Fit dependent-input token/context ANOVA on cached MLP0 rows.**  This is the best
   route to a canonical mixture of lexical and contextual structures.
3. **Build the sparse simultaneous-perturbation toy and preregister a small real
   mixture pilot.**  It can materially lower response-collection cost, but only if its
   dense and finite-step controls pass.

The executed action in this review is priority 1's proof/code gate.  No GPU/model row,
protected outcome, or concurrent artifact was opened or modified.  The next real use
waits for a fitted lawful response program; until then the new result is an exact tool,
not progress on the strict whole-model percentage.

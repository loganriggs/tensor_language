# Three-hour mathematical review — 2026-08-30 06:10 UTC

## Project state used for this review

This review is conditioned on the current receipts rather than on a generic list of
factorizations.

- The strict whole-model ledger remains **5.348245316% certified storage** and
  **10.923302467% named causal CE**.  The unexplained balance is **4.72714 nat**, or
  **89.076697533%**, and **0/68** desired terminal actions yet jointly pass extraction,
  selective removal, and OOD transport.
- MLP0's completed TT/X/CC factorial says its useful structure is strongly
  compositional: on SELECT the pair dividends are `+1.721576`, `-1.153678`, and
  `-1.032800` nat, while the remaining three-way dividend is only `+0.024390` nat.
- The equality-copy tensor is exact and extractable and transports to code, but its
  narrow induction-only collateral certificate fails.  This is direct evidence that a
  correct primitive can serve several behavior branches.
- The a8 and a16 circuit geometries differ.  At a8, a common direction explains 91.6%
  of the fitted direction variance and selective private residuals remain.  At a16,
  the analogous common direction explains only 48.9%; projecting it out generally
  hurts.  A single global “shared parent plus leaves” decomposition is therefore
  falsified even though a component-conditioned version remains plausible.
- Learned rank-64 DAS subspaces recover only about 21–35% of component damage.
  Low-rank directions are enriched but not complete circuit mechanisms.
- At inspection time the RTX 5090 was idle.  A newline row-preparation test was the
  only active project-side Python job; no scientific GPU outcome was pending.

The central missing interface is no longer “find a small local MSE basis.”  It is:

> represent which tensor programs are reusable services, which uses are private, and
> which combinations have irreducible causal interactions—then make that
> representation predict interventions not used to fit it.

## Ranked move 1 — sparse Möbius tomography of circuit interactions

### Exact bilin18 object

Let the proposed executable circuit programs be indexed by
`1,...,n`.  For a subset `S`, run one registered intervention arm and record a scalar
outcome

\[
F(S)= -\mathrm{CE}(S),
\]

or a registered circuit-specific logit/effect score.  Here `S` can mean “these
programs are present in an otherwise deleted background” or “these programs are
removed from native,” but the convention must be frozen and cannot be mixed.

The Boolean-lattice Möbius coefficient is

\[
m(T)=\sum_{U\subseteq T}(-1)^{|T|-|U|}F(U),
\qquad
F(S)=\sum_{T\subseteq S}m(T).
\]

`m({i})` is a main effect.  `m({i,j})` is the part of the joint effect that cannot be
predicted by adding the two main effects.  Higher-order coefficients are irreducible
three-way and larger causal interactions.  The collection of nonzero `m(T)` is an
**interaction hypergraph**: reusable independent services are separate vertices;
inseparable compositions are hyperedges.

This is exactly the calculation already used for the MLP0 factorial, generalized from
three branches to a circuit library.  It provides a direct target for the user's
proposed sparse/DAG tensor program: hierarchy is earned only if the measured
interaction hypergraph is sparse and stable.

### Established mathematics and operational theorem

Full-cube Möbius inversion is exact in `O(n 2^n)` arithmetic operations.  If only `s`
low-degree coefficients are nonzero, the problem becomes sparse polynomial learning
in the AND/subset basis.  Schapire and Sellie gave exact learnability results for
sparse multivariate polynomials with membership queries in
[Learning Sparse Multivariate Polynomials Over a Field with Queries and
Counterexamples](https://doi.org/10.1006/jcss.1996.0017).  A directly relevant recent
algorithmic treatment is
[Adaptive Sparse Möbius Transforms for Learning Polynomials](https://arxiv.org/abs/2602.06246),
which uses adaptive group-testing ideas and explicitly addresses the high coherence of
the AND basis.

The important warning is mathematical, not cosmetic: ordinary random compressed
sensing assumptions do **not** automatically hold because subset-containment columns
are strongly correlated.  Orthogonal matching pursuit (OMP) is therefore only a cheap
diagnostic baseline, not a licensed recovery theorem for bilin18.

### Assumptions that may fail

- The ten proposed circuits may have a dense interaction hypergraph.  Then the method
  honestly says there is no cheap sparse program at that granularity.
- CE is nonlinear and depends on the chosen intervention background.  Present-in-empty
  and remove-from-native coefficients need not agree.
- A scalar average can hide document- or context-routed interactions with cancelling
  signs.  Coefficients must transport split-half and by domain, and later be evaluated
  at a vector of circuit outcomes rather than CE alone.
- Proposed circuits overlap in their native owners.  A subset arm must specify how a
  shared tensor is counted and executed once; otherwise the set function is ill-defined.
- Query-efficient recovery can fail even when the truth is sparse if interaction
  magnitudes are small, noisy, or the query design is coherent.

### Prediction beyond reconstruction

Fit coefficients using only a registered subset of intervention arms.  The method must
predict the CE, target effects, and collateral of **unseen circuit combinations**.  A
useful sparse hypergraph also predicts which removals commute and which private use
branch can be deleted without disturbing another consumer.  Its simplicity price is
the stored tensor services plus the nonzero interaction terms and executed products,
not merely a local rank or MSE.

### Cheapest falsifying real-model experiment

Start with four already executable programs rather than ten: previous-token, equality
copying, bracket closure, and newline.  Freeze one intervention convention and a
degree-two/three budget.  Measure 16–32 strategically chosen subset arms on already
open SELECT rows, fit the sparse interaction model, and predict withheld arms.

Fail this move if it does not beat an additive-main-effect model and a matched dense
ridge baseline on withheld CE **and** circuit-specific collateral, or if support/sign
does not replicate across document halves.  Do not run the 1,024-arm ten-circuit cube
before this small gate.

### CPU gate executed during this review

The new auditable implementation is
[`sparse_mobius_interaction_tomography.py`](sparse_mobius_interaction_tomography.py),
with four tests in
[`test_sparse_mobius_interaction_tomography.py`](test_sparse_mobius_interaction_tomography.py).

It passed three distinct checks:

1. Exact Möbius/zeta inversion round-trips an arbitrary five-component cube.
2. A planted 12-component system with 8 nonzero terms among 299 candidate degree-at-
   most-3 terms is recovered exactly from **202** subset queries, versus **4,096** for
   the full cube.  Held-out RMSE is `7.17e-15`.
3. The same 8-term decoder is rejected on a dense 299-term control: normalized
   held-out RMSE is **0.4692**.

The implementation also reanalyzed the completed MLP0 cube.  It reproduces every
registered coefficient to at most `8.9e-16` nat.  Removing only the third-order term
causes a maximum arm-prediction error of `0.01660` nat on FIT and `0.02439` nat on
SELECT; the largest pair term is `1.70640` and `1.72158` nat respectively.  This is a
real known-answer example where a degree-two interaction program is accurate, though
not yet a query-efficiency result because all eight MLP0 arms were already measured.

The numerical receipt is
[`sparse_mobius_interaction_tomography_toy_receipt.json`](sparse_mobius_interaction_tomography_toy_receipt.json).
Its claim boundary explicitly excludes model, row, protected-outcome, circuit-promotion,
and ten-circuit recovery claims.

## Ranked move 2 — component-conditioned block-term decomposition

### Exact bilin18 object

Do not impose one global hierarchy on all circuits.  For each owner component `c`, form
a three-way response tensor

\[
\mathcal R_c[\text{circuit},\text{intervention},\text{response coordinate}],
\]

where interventions include mean removal, interchange, and learned subspace removal,
and response coordinates include member effect, matched negative, and downstream
consumer effects.  Fit a block-term decomposition

\[
\mathcal R_c \approx \sum_b \mathcal G_{c,b}
\times_1 U_{c,b}\times_2 V_{c,b}\times_3 W_{c,b},
\]

with a component-specific number and multilinear rank of blocks.  A block may be a
shared service; sparse circuit loadings within that block are its private uses.  a8 is
the positive shared-plus-private case.  a16 is the essential negative control that
should select mostly private or different blocks.

Block-term decompositions generalize CP and Tucker decompositions and have uniqueness
results under rank/independence conditions; see De Lathauwer,
[Decompositions of a Higher-Order Tensor in Block Terms—Part II](https://epubs.siam.org/doi/10.1137/070690729).
If the circuit slices instead lie in a union of subspaces, sparse self-expression offers
an alternative grouping principle; see Elhamifar and Vidal,
[Sparse Subspace Clustering](https://doi.org/10.1109/TPAMI.2013.57).

### Assumptions that may fail

- BTD uniqueness conditions can fail when blocks overlap strongly or response modes
  are rank deficient; numerical canonicalization is not semantic identifiability.
- Mean/interchange/DAS responses may not span the finite edits that matter at terminal
  interfaces.
- A fitted common block may be merely a high-amplitude nuisance, as the a8 common
  direction was nonselective before residualization.
- Component-by-component fitting can double-count a service shared across components.
  Cross-component composition remains a later constraint.

### Prediction beyond reconstruction

From a subset of circuits and interventions, the blocks must predict held-out entries
of the circuit-by-intervention response tensor and the effect of deleting two inferred
private leaves together.  A shared block is valuable only if it is stored once and its
private leaves are more selectively editable than matched flat low-rank directions.

### Cheapest falsifier

Use the already collected a8 and a16 response matrices.  Freeze model-selection and
block ranks on half the circuits/interventions; compare (i) one flat SVD, (ii)
independent per-circuit factors, and (iii) shared-plus-private BTD on held-out response
cells.  It must choose a shared substrate at a8 but not hallucinate the same structure
at a16, and it must improve predicted pair interventions or selective removal per
stored float.  Otherwise prune BTD before any new GPU collection.

## Ranked move 3 — finite-Hankel minimal-realization triage by circuit family

### Exact bilin18 object

For a particular sequential circuit service, make a matrix whose rows are prefixes and
whose columns are registered suffix continuations/interventions:

\[
H[p,s] = F(ps).
\]

`F` is not the model's raw hidden state; it is the verified terminal response of the
candidate service.  Candidate families are deliberately different:

- quote parity: two-state hypothesis (inside/outside quote);
- ordered successor: small finite transition hypothesis;
- bracket closure: depth-dependent stack-like hypothesis;
- equality copying: associative-memory hypothesis whose state may grow with vocabulary
  and lag.

For a rational series, finite Hankel rank equals the number of states in a minimal
weighted finite automaton.  This is the operational theorem behind spectral weighted-
automata learning; a compact primary algorithmic reference is Balle and Mohri,
[Spectral Learning of General Weighted Automata via Constrained Matrix
Completion](https://proceedings.neurips.cc/paper_files/paper/2012/file/6602294be910b1e3c4571bd98c4d5484-Paper.pdf).
For tensorized recurrent maps, Rabusseau, Li, and Precup connect linear 2-RNNs and
weighted automata in
[Connecting Weighted Automata and Recurrent Neural Networks through Spectral
Learning](https://proceedings.mlr.press/v89/rabusseau19a.html).

### Assumptions that may fail

- Transformer services are approximate, continuous, and context-dependent, not exact
  stationary automata.
- A small observed Hankel block can underestimate rank; rank must be tracked as prefix,
  suffix, depth, lag, and vocabulary support grow.
- A low scalar-response rank can hide distinct internal states needed for different
  interventions.  Several verified outputs are required.
- Bracket and copying services may require pushdown or associative memory, so a finite
  automaton can be the wrong mathematical class.  That negative answer is still useful
  triage.

### Prediction beyond reconstruction

A stabilized low Hankel rank predicts OOD response to unseen prefix/suffix
compositions and gives a lower bound on the number of linear states required by that
service.  Rank growth predicts that a fixed-state extraction will fail as bracket
depth, copy lag, or vocabulary diversity increases.  This separates services that
should compile to a tiny state machine from those that require a tensor memory.

### Cheapest falsifier

Run a model-free symbolic gate first: parity must stabilize at rank 2; bounded-depth
brackets should grow until the depth bound; equality matching should grow with the
number of independently addressable tokens.  Then fill small empirical Hankel blocks
from already-open circuit receipts and test whether singular-rank growth and withheld
suffix prediction match those registered expectations.  If empirical rank is unstable
under row resampling or fails withheld suffixes, do not use a state-count claim.

## Ideas pruned in this review

- **More global CP/Tucker/HOSVD of a raw MLP tensor:** exact slice rank and poor
  rank-64 causal recovery already say this cannot be the whole program.  It remains a
  local parameterization tool, not the next experiment.
- **One universal hierarchy/DAG:** a16 is a direct counterexample to the a8 geometry.
  Only component-conditioned structure survives.
- **Another activation SAE or weight dictionary scored by local MSE:** MLP1 already
  gives a strong conditional dictionary result, but pricing the native router and
  predicting composition are the unresolved issues.  Repeating the fit is redundant.
- **Consumer commutant blocks, routed tensor dictionaries, and projected causal
  abstraction:** these remain promising but were the three executed moves of the
  00:45 review, so they are not counted again as new mathematics.
- **Balanced realization alone:** consumer-weighted balancing was already ranked at
  22:48.  The finite-Hankel proposal above is narrower and falsifies the state-machine
  class separately for each verified service.

## Result-to-price ranking and immediate decision

1. **Sparse Möbius interaction tomography** — highest direct information gain because
   it measures the exact composition/removal failure that local decompositions miss;
   CPU implementation and two-sided toy gate now pass.  Next cost: 16–32 arms on four
   already executable circuits, not a ten-circuit powerset.
2. **Component-conditioned BTD** — almost free because a8/a16 response artifacts
   already exist, and it converts their contrast into a held-out predictive test.
3. **Circuit-family Hankel triage** — cheap and highly falsifiable; likely to give both
   positive (parity/successor) and negative (copy/stack) structural results, but it
   depends on sufficiently complete verified terminal responses.

The first move is advanced from idea to tested CPU machinery in this review.  It does
not change the strict whole-model ledger.  Its next promotion gate is an outcome-blind
four-circuit subset-arm preregistration with explicit overlap accounting and held-out
combination prediction.

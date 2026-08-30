# Three-hour mathematical review — 2026-08-30 00:45 UTC

## Bottom line

The most promising new mathematical move is to search for **common reducing
subspaces of downstream consumer metrics**, not low rank of one native weight tensor.
This turns Logan's proposed joint decomposition into a precise algebraic question:
can the family of things that read an early write all be put into the same small block
structure?  If so, those blocks are candidates for independently editable circuits.
If the family has only the trivial common block, that version of the hypothesis is
falsified even if an SAE reconstructs activations well.

The second move is the sparse routed interaction-tensor program proposed in the newest
discussion.  It is now backed by a known-answer implementation, but has not yet been
tested on a folded real MLP atom bank.  The third move is to evaluate any lossy code as
an approximate **causal abstraction**: interventions must commute with compression up
to a measured error, including unseen compositions and held-out consumers.

The strict whole-model accounting has not changed:

- certified removable storage: `29,196,288 / 545,904,054 = 5.348245316%`;
- named causal CE: `0.57968 / 5.30682 = 10.923302467%`;
- unexplained CE: `4.72714 nat = 89.0767%`;
- terminal extraction/removal/OOD actions: `0/68`.

## Evidence inspected in this review

### Native and compiled structure

- All 18 native MLP polarization slices have numerical rank 1,152.  Their rank-768
  coefficient tails vary smoothly (`0.1210--0.1325`), with MLP10/MLP9 only `1.0113x`.
  Raw tensor rank and ordinary HOSVD therefore do not explain the compiled program's
  layer-10 causal rank knee.
- The compiled program is highly heterogeneous.  Its MLP tables buy extra rank mainly
  at layers 10--17, whereas attention rank is comparatively homogeneous and shallow
  stream maps are over-bought.  Uniform sweeps failed on two heterogeneous axes and
  succeeded on the homogeneous axes.
- A live suffix exhibits a sharp, coverage-invariant incompatibility when context-free
  early MLP writes are followed by live attention.  The damage saturates near 10.7 nat,
  attention 5 acts mostly as a presence/control interface, and attention 6 has a small
  but high-dimensional content term.  This structure largely disappears inside the
  all-table program because later sites no longer read live state.

### Early-MLP replacement and composition

- A free-factor rank-512 MLP2 program recovers `68.27%` of MLP2-deletion CE at equal
  executable product count, but is not faithful: centered-logit NRMSE `0.14538`, local
  write NRMSE `0.6866`, and top-1 agreement `87.52%`.
- Adding C512 exposure to local MLP2 MSE training reduces the C512×MLP2 interaction by
  only `13.2%`; its advantage over an equal-compute native-only control is unresolved.
- The three frozen MLP2 programs share a diffuse document-level interaction pattern,
  but the `91.2%` first document mode is shared ordering of document effects—not a
  residual-space direction or circuit.
- MLP1's weight-action dictionary remains the strongest positive conditional result:
  many global atoms, only 8--64 used per position, and much better CE recovery than
  matched activation-weighted SVD.  It is not yet an honestly priced conditional
  executor because its current router still computes all native gates.

### Rayleigh status and exact blocker

The audited Rayleigh v2 DESIGN collector completed in `261.260653 s`.  Its receipt and
ledger hashes are frozen, and HELDOUT remains unopened.  The required independent audit
of the DESIGN scorer returned **NO-GO**, also without opening protected values.

The scorer currently calls `protected_snapshot()` before publishing its
`frozen_before_design_ledger_open` authority, and that snapshot deserializes and
semantically inspects the DESIGN ledger.  Failure provenance can also claim the ledger
was unopened after such a pre-authority failure, and protected-input drift can prevent
publication of the failure artifact itself.  This is an execution-lifecycle bug, not a
scientific result.  The repair must bind hashes before authority, publish and re-read
authority, and only then deserialize DESIGN; synthetic ordering, race, drift, and
receipt-last tests are required before a fresh audit.

## Ranked new move 1 — consumer-common blocks from the commutant algebra

### Exact bilin18 object

Choose an early error/write coordinate space, initially a 32--128 dimensional sketch of
MLP2 product or folded atom coordinates.  For each downstream consumer $c$ and document
$d$, let $J_{c,d}$ map a small edit $e$ at that interface to the consumer response, and
let $W_{c,d}$ be its meaningful output metric: categorical Fisher for logits, normalized
Euclidean energy for a residual consumer, or a registered circuit-specific metric.
Form the positive-semidefinite pullback

$$
G_c=\mathbb E_d[J_{c,d}^{T}W_{c,d}J_{c,d}],
\qquad
q_c(e)=e^TG_ce.
$$

Whiten the sum $S=\sum_cG_c+\gamma I$ and define

$$
H_c=S^{-1/2}G_cS^{-1/2}.
$$

Whitening prevents a high-amplitude consumer from defining the coordinates merely by
scale.  Now study the commutant

$$
\mathcal C=\{X:XH_c=H_cX\ \text{for every }c\}.
$$

Equivalently, inspect the small singular values of

$$
\mathcal K(X)=([H_1,X],\ldots,[H_m,X]).
$$

The identity is always in the commutant.  Extra self-adjoint elements yield projectors
onto common reducing subspaces.  The finest simultaneous block diagonalization can be
constructed from the commutant algebra, with numerical error explicitly controlled;
this is the object developed by Maehara and Murota in
[Algorithm for Error-Controlled Simultaneous Block-Diagonalization of Matrices](https://epubs.siam.org/doi/10.1137/090779966).
Exact simultaneous diagonalization is the special commuting case described by Cardoso
and Souloumiac's
[Jacobi Angles for Simultaneous Diagonalization](https://epubs.siam.org/doi/10.1137/S0895479893259546).

### Why this is new and what it would mean

Previous rank and HOSVD tests asked whether one map has a small global basis.  This asks
whether a **family of downstream readers shares a block structure**, while allowing
each block to remain high-rank internally.  It directly formalizes the proposed joint
optimization in which early components are simple and downstream components are sparse
relative to the same decomposition.

If $e_a$ and $e_b$ lie in different exact blocks, then every quadratic consumer has

$$
q_c(e_a+e_b)-q_c(e_a)-q_c(e_b)=2e_a^TG_ce_b=0.
$$

Thus the algebra predicts more than reconstruction: independently editing two blocks
should have negligible interaction across all registered consumers.  Approximate blocks
predict a quantitative bound from their off-block energy.  Stable blocks can also be
priced, extracted, and selectively removed separately.

### Assumptions that may fail

- Document-averaged pullbacks may appear block diagonal even if individual contexts use
  incompatible rotations.  Split-half and per-context residuals are mandatory.
- The finite MLP2 failure may lie outside the tangent/Fisher regime.  The blocks must
  predict finite Möbius interactions, not only $e^TG_ce$.
- A small randomized sketch can create or erase apparent invariant subspaces.  Results
  must replicate as sketch dimension grows and against a covariance-matched null.
- Several consumers may be redundant.  A one-consumer family is trivially diagonalizable
  and carries no evidence of common circuit structure.
- RMSNorm and attention make the pullbacks state-dependent.  One global family may have
  only the identity commutant even when context-routed families have useful blocks.

### Cheapest falsifying real-model experiment

Use 64 fit and 64 evaluation documents and 64 frozen edit directions drawn from the
existing MLP2 folded/rank-512 error span plus orthogonal randomized controls.  Collect
signed finite responses for final logits, attention 5, attention 6, and at least one
independently verified late circuit.  Fit no neural network.  Estimate each sketched
$G_c$, whiten, and freeze the commutant spectrum/block threshold on fit documents.

Fail the move if any of the following holds on evaluation documents:

1. there is no reproducible commutant mode beyond the identity;
2. block projectors have split-half principal-angle instability;
3. held-out cross-block finite interaction is not lower than equal-size random blocks;
4. a block removal is no more consumer-selective than matched SVD coordinates; or
5. the storage and response-collection price exceeds a flat routed dictionary at the
   same finite CE fidelity.

### Toy gate executed in this review

`toy_consumer_commutant_blocks.py` plants PSD consumer forms with hidden block sizes
`2,3,2` behind a random orthogonal gauge.  It recovers exactly three commutant modes and
the correct block sizes; recovered off-block energy is `1.52e-30`; regauging changes the
spectrum by `5.55e-15`; and cross-block edit interactions vanish to about `1e-15`.
A generic dense family has only the identity commutant and nonzero edit interactions.
Adding small cross-block noise turns the two structural zero modes into singular values
`0.00573` and `0.00686`, well below the next mode `0.18736` but correctly nonzero.

This validates the algebra and implementation, not the existence of bilin18 blocks.

## Ranked new move 2 — sparse routed interaction tensors, selected by causal MDL

### Exact bilin18 object

Start with the already positive MLP1 weight-action dictionary.  Fold each encoder atom
through the native bilinear gate:

$$
e_a^T[(Lx)\odot(Rx)]=x^TQ_ax,
$$

and represent a local program as

$$
\widehat y(x)=b+\sum_{a\in S(x)}d_a\,x^TQ_ax.
$$

The bank may be globally large while $S(x)$ is small.  A tree or DAG router permits
shared coarse blocks and overlapping meanings.  For adjacent components, first fit the
Möbius interaction tensor

$$
T^{\mathrm{int}}_{A,B}=T_{A,B}-T_{A,0}-T_{0,B}+T_{0,0},
$$

because the joint computation may be substantially sparser than either full component.
Optimize normalized tensor functional distance, exact bias and scale, CE, support, and
full stored/executed price.  Hard routing makes the whole function piecewise polynomial;
tensor similarity is exact only within a fixed support and cannot validate the router.

Mixtures of separable tensor dictionaries have established local-identifiability results
under sparse generative assumptions; see Ghassemi et al.,
[Learning Mixtures of Separable Dictionaries for Tensor Data](https://doi.org/10.1109/TSP.2019.2952046).
Those assumptions are hypotheses here, not guarantees.

Among equal-fidelity flat/tree/DAG candidates, use **prequential description length**
of held-out routes and downstream consequences as a validation metric: how many bits are
needed to teach the router/reader as examples accumulate?  This rewards reusable
structure and sample efficiency rather than a post-hoc semantic label.  Online MDL's
operational meaning and stability are demonstrated by Voita and Titov's
[Information-Theoretic Probing with Minimum Description Length](https://aclanthology.org/2020.emnlp-main.14/).
MDL selects among candidate coordinates; it does not discover a causal basis by itself.

### Assumptions that may fail

- Existing sparse codes may depend on computing all 4,608 native products, making them
  representationally sparse but not executable.
- Sparse-dictionary identifiability conditions such as generative sparsity and adequate
  diversity/incoherence may fail.
- A post-hoc tree over independently trained atoms is not canonical; the previous
  geometric nesting test failed.
- Local weight-path interactions can be destroyed or rotated by RMSNorm and attention.
- A short MDL code can encode token identity or dataset regularity without supporting
  interventions.  Causal held-out tasks remain mandatory.

### Measurable consequence and cheapest falsifier

First compute an **oracle routing bound** on the frozen real MLP1 tensor atoms.  For each
held-out position, choose the best $k$ blocks after charging the complete decoder and
atom bank.  If oracle $k=8,16,32$ cannot meet the desired CE/interface fidelity at a
price below the current dictionary, stop before training a router.

If it passes, compare flat, tree, and DAG routers with identical bytes and executed
products.  Freeze the route-description protocol before evaluation.  The hierarchical
candidate must beat the flat candidate in prequential bits **and** predict an unseen
composition or selective edit with no additional collateral.  Merely lowering local MSE
or producing attractive atom labels fails.

The algebra, scale counterexample, Möbius cancellation, wrong-router control, and
three-seed Adam recovery already pass in
`toy_sparse_routed_interaction_tensor.py`.  The next toy before any tree/DAG fit must
plant overlapping parentage and verify that a DAG wins only when the ground truth truly
has overlapping support; a planted flat null must prevent a false hierarchy win.

## Ranked new move 3 — projected causal abstraction at the live-state interface

### Exact bilin18 object

Let $z$ be the residual state at a chosen boundary, initially just before attention 5 or
attention 6, and let $\tau(z)$ be a proposed compressed code.  Let $\mathcal I$ contain
the interventions we actually require the program to support: native/compiled MLP0--2
writes, sparse block edits, attention-presence versus content substitutions, and their
registered compositions.  A high-level program needs an intervention map $\omega$ such
that compressing and intervening approximately commute:

$$
d_{\mathrm{out}}\!\left(
\tau(F_i(z)),\widehat F_{\omega(i)}(\tau(z))
\right)\leq\epsilon
\quad\text{for }i\in\mathcal I.
$$

The output distance must include logits/CE and selected consumer responses, not residual
MSE alone.  Multiple low-level interventions may map to the same high-level edit while
retaining different effects—a central problem for our lossy replacement programs.
Xia and Bareinboim's
[Causal Abstraction Inference under Lossy Representations](https://proceedings.mlr.press/v267/xia25a.html)
introduces projected abstractions specifically to accommodate this failure of abstract
invariance.  For successive abstractions, Rischel and Weichwald's
[Compositional Abstraction Error and a Category of Causal Models](https://proceedings.mlr.press/v161/rischel21a.html)
formalizes the desideratum that composite abstraction error be bounded by the component
errors.

### Why this matters beyond reconstruction

This makes “simpler” earn its name operationally.  A code is useful only if it supports:

- predicting an unseen allowed intervention;
- composing two edits with bounded additional error;
- removing one abstract variable without unrelated consumer damage; and
- transporting the same intervention map to held-out documents or consumer circuits.

It also explains the attention-interface anomaly cleanly: a context-free early write and
a live attention suffix may fail to belong to the same abstraction family even when each
component separately has low error.

### Assumptions that may fail and cheapest falsifier

The guarantee is only relative to the allowed intervention set and chosen output metric;
an incomplete set can certify the wrong quotient.  Exact causal-abstraction theorems do
not automatically apply to our continuous, finite-sample, approximate transformer.
Worst-case distances may be dominated by numerical or rare-token outliers, while means
can hide catastrophic failures, so report mean, high quantile, and maximum separately.

For any candidate $\tau$, freeze pairs of low-level states that map to the same abstract
code.  On held-out documents, apply every registered intervention and measure the
within-code spread of suffix logits and consumer responses.  Add one intervention and
one consumer not used to form the code.  If within-code spread is no smaller than a
matched-rank PCA/SVD code, or if the intervention map fails on the unseen consumer, the
abstraction is not doing useful causal work.  This is cheaper and more decisive than
training another whole MLP replacement.

A dedicated toy is required before real use: plant a lossy low-level system where two
micro-interventions share one projected high-level effect, verify exact commutation and
the compositional error bound, then include a hidden intervention that must break the
abstraction.  That toy is the gate; no real-model abstraction claim is licensed yet.

## Reconsidered mathematics that is pruned or deferred

- **Raw tensor/arithmetic-circuit rank and ordinary HOSVD:** pruned as an explanation of
  the layer-10 knee by the full-rank, smooth native polarization profile.  Algebraic rank
  remains a lower bound inside a consumer-validated block, not the discovery metric.
- **Norm minimization before HOSVD:** useful for choosing balanced factors and numerical
  conditioning, but it cannot change the contracted functional tensor or create a common
  downstream block.  It needs its own planted gauge toy before reuse and is subordinate
  to Moves 1--2.
- **Another weight SAE or post-hoc hierarchy:** previous independent dictionaries show
  activation containment without clean geometric nesting; shared/private rank-512
  hierarchy lost at matched price.  Repeating the same objective is redundant.
- **Polynomial/Volterra extrapolation:** degree 1--3 local fits all failed at the physical
  suffix replacement.  Do not infer finite composition from another local polynomial.
- **Hankel/automata minimal realization:** still mathematically relevant after a broader,
  intervention-complete consumer bank.  The earlier final-output tangent panel was full
  rank and split-unstable; running another narrow Hankel SVD now would rediscover probe
  choice rather than a minimal causal state.
- **Information bottleneck alone:** mutual-information compression can preserve token or
  document identity while failing interventions.  Use MDL to compare already causal
  candidates and projected abstraction to validate them.
- **Invariant theory/gauge canonicalization:** tensor similarity and the commutant spectrum
  already quotient the relevant factor/orthogonal gauges for these pilots.  A more ornate
  invariant ring has no current downstream prediction.
- **Approximation certificates:** Davis--Kahan-type subspace stability and commutator-tail
  bounds become valuable if Move 1 finds a spectral gap.  Before such a gap they certify
  nothing useful.
- **Sparse program synthesis:** premature at whole-model scale.  It becomes appropriate
  inside verified consumer-common blocks or a causally validated routed dictionary.

## Execution and immediate plan

1. **Completed CPU toy for Move 1.**  One pytest passed in `2.16 s`; the numerical
   receipt runner took `0.31 s`.  All positive, gauge, perturbation, null, and edit
   controls pass.  This is the executed highest-priority new mathematical action.
2. **Rayleigh DESIGN scorer lifecycle repaired, not run.**  Metadata-only hashing and
   JSON joins now precede authority; scorer authority is published and exactly re-read
   before the first DESIGN tensor deserialization.  Protected or authority drift remains
   publishable as an exact-byte-bound failure instead of requiring the input to become
   healthy again.  New toys cover success ordering, pre-authority drift, post-authority
   drift, authority mutation, rival terminals, lock replacement, and receipt-last
   publication.  The complete relevant transitive suite passes `140/140` in `16.65 s`.
   A fresh independent exact-source audit is still required.  Do not run the scorer or
   HELDOUT before exact GO.
3. **If Rayleigh HELDOUT passes,** use its validated consumers to seed Move 1's pullback
   family.  If it fails, do not train a Fisher-weighted MLP2; broaden the signed consumer
   bank and test common blocks directly.
4. **In parallel after source hygiene,** run Move 2's frozen MLP1 oracle-routing bound.
   It is the cheapest real-model test of whether sparse tensor blocks can become a truly
   cheaper executor rather than a sparse description of an expensive computation.

No strict ledger quantity moves from this review or its toy.

## 01:01Z execution addendum: second scorer audit and exact repair

The fresh independent audit of commit `af7be129` was outcome-blind and returned
**NO-GO** even though `136/136` closure tests passed.  The remaining issue was a
transaction boundary, not the Rayleigh mathematics: after one exact replay of the
published scorer authority, the code could lose its lock or acquire a rival terminal
before opening the DESIGN tensor ledger.  It would suppress the final receipt, but the
protected values could already have been opened.  DESIGN scoring therefore remained
unlicensed and no audit artifact was created.

The repair adds a dedicated guard immediately before the first possible DESIGN tensor
load, in this exact order:

1. exact replay of the already-published scorer authority;
2. proof that bundle, receipt, and failure terminals are absent; and
3. proof that this process still owns the lock.

There is no protected operation between that guard and the semantic snapshot.  The
failure reporter also no longer opens protected tensors merely to diagnose a failure
that occurred before the open boundary.  Adversarial tests now replace the authority,
insert a rival terminal, and replace the lock at that boundary; all three yield **zero**
DESIGN tensor loads.  The six focused Rayleigh suites pass `65/65` in `5.89 s`.
Predictor features, ridge grid, family selection, nulls, thresholds, and preregistered
HELDOUT decisions are unchanged.  A new exact-source independent GO is still mandatory
before the scorer runs.

Separately, the compiled-program end-to-end confirmation finished in `91.2 s`.  At
5,419 table rows, the heterogeneous 202.6M-value program improves CE over the old
230.087M-value deployed program by `0.072302` nat (`SE=0.001598`, `t=45.25`,
`n=92,160`) while using about 12.0% fewer stored values.  It improves over the previous
189.5M-value build by `0.003064` nat (`SE=0.000297`, `t=10.31`).  At 16,110 rows the
corresponding gains are `0.032619` and `0.007486` nat.  This is strong executable
compression evidence, but it remains a compiled-program result: it does not move the
strict native-model causal ledger or explain why the learned native circuit works.

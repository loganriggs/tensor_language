# Hourly strategic circuit review — 2026-09-08 20:22 UTC

## High-level result

This hour materially improved circuit quality rather than adding another fitted
subspace.  The exact M11 task-functional tensor was split into its writer state
`H` and downstream reader state `R`; two prospective reporter-disjoint CPU tests
then determined what is reusable.  The reader is stable across reporters, while
the writer requires six coupled construction-by-direction states.  A separate
global scalar-gain hypothesis failed.  A fresh v21 corpus passed native capability,
and an exhaustive causal atlas patched all 24 eligible upstream input/modules and
all 108 attention heads.

The atlas also revealed a preregistration error in my own design.  Its exhaustive
measurements are useful discovery evidence, but its `invalid_instrument` terminal
cannot support a null claim.  The next confirmation will use a fresh v22 bank and
the correct reader-contracted task-functional writer endpoint rather than raw `H`
energy.

## Exact H/R computation and localization

For MLP11 input `x`, the exact factor writer state is

\[
H_n(x)=(L_nx)(R_nx),
\]

and the downstream tangent reader is `R_n=<g,D_:n>`.  Their positionwise product
and sum reconstruct the frozen task-functional factor tensor exactly.  The managed
artifact has shape `4 x 16 x 16 x 4608`; its maximum reconstruction error is zero.

The A/P cross-contractions resolve the context-sign question:

- A writer × A reader: `+0.11810`
- A writer × P reader: `+0.11666`
- P writer × A reader: `-0.07051`
- P writer × P reader: `-0.06439`

Changing the reader leaves sign and scale nearly unchanged; changing the writer
flips the sign.  This is writer control at the present factor resolution.  It is
not a claim that raw writer vectors are globally semantic—the pooled raw A/P writer
cosine is only `-0.250`—but it is a causal localization through exact cross-
contraction.

The frozen H/R state crossfit sharpened that interpretation.  Independently
averaged cell-level `Hbar` and `Rbar` contract to held-out `Q` with cosine `0.95243`,
signed recovery `0.91772`, and `48/48` positive directions.  The additive H/R
program collapses to cosine `0.42966`, so the output-level additive Q hierarchy
does not factor into additive writer and reader hierarchies.  Reader states have
held-out cell cosine `0.95370` and corresponding-cell cross-fold stability
`0.97844`; writer values are only `0.64567` and `0.78731`.  The retained abstraction
is therefore a stable reader plus six coupled construction×direction writer states.

The scalar-gain crossfit closed the tempting simpler explanation.  The constant
six-state baseline has raw-Q residual `0.30891`.  Writer, reader, product, and joint
gain models all have negative held-out R² (`-0.393`, `-0.023`, `-0.219`, `-0.890`)
and worsen raw residual (`0.409`, `0.315`, `0.394`, `0.486`).  Continuous amplitude
must come from factor-/position-/cell-specific covariance, not one global H/R
strength number.

## Full v21 module/head patching

The v21 capability screen passed every A1/A2 construction×direction×side cell;
all 16 A1 and 16 A2 rows are jointly capable.  With no causal outcomes open, I
registered and ran an exhaustive writer atlas over all v21 rows:

- input embedding response;
- complete attention responses at layers 0–11;
- complete MLP outputs at layers 0–10;
- every one of nine heads at every attention layer 0–11.

The atlas executed exactly 136 forwards / 8,704 row evaluations.  The all-site
base self-patch is exactly inert, and the all-site donor union gives exactly `1.0`
target behavior and raw-H recovery.  Behaviorally, the strongest non-input modules
are MLP1 `0.817`, attention9 `0.551`, MLP3 `0.546`, MLP2 `0.537`, MLP4 `0.435`, and
MLP6 `0.385`.  The strongest heads are L9H1 `0.222`, L9H4 `0.220`, L8H1 `0.194`,
L11H3 `0.189`, and L0H3 `0.103`.

The first four heads are exactly the four independently identified on v15.  That
cross-corpus replication is valuable narrowing evidence: the behavioral head
frontier is stable across two different constructions and lexicons.  It is still
discovery evidence here because the registered atlas terminal is invalid.

## Red-team of the atlas instrument

Three problems invalidate its registered selectivity decision:

1. The donor union was required both to reconstruct the full donor trajectory and
   to remain quiet on donor C.  Exact closure necessarily reproduces C's unrelated
   donor change, so those requirements conflict.
2. A raw absolute `1e-4` threshold was applied to quadratic H values.  Tiny M11-
   input roundoff is amplified by unnormalized factor gauges even while target-H
   cosine and recovery are exactly `1.0`.  H closure needs a scale-aware relative
   error; residual state and margin can retain absolute closure.
3. Raw H gives equal importance to all 4,608 available factors.  It penalizes a
   head that changes a small subset strongly read downstream.  For example L9H4
   causes `0.220` behavioral recovery but only `0.027` raw-H projection.  The
   correct causal-use endpoint is H contracted with frozen R, yielding Q; raw H
   belongs under capability/occupancy diagnostics.

Accordingly, no “union nonselective” or “no head” claim is accepted.  The invalid
result and a separate audit are retained rather than repaired silently.

## Incorporating weight-tensor decomposition

The user-proposed distinction is now operational:

1. **Capability:** restrict literal checkpoint writer maps and the M11 bilinear
   tensor to the causally nominated subspace; decompose exact unfoldings and native
   hidden-factor atoms.  This describes computations the weights can implement on
   any input.
2. **Occupancy:** project sealed v20/v21/v22 activations into those fixed coordinates
   and record frequency, sign, magnitude, and co-occurrence.  PCA/SAE/hierarchical
   SAE may summarize occupancy but may not rotate or define the weight basis.
3. **Causal use:** contract writer changes with stable reader R and require fresh
   component reset/rescue effects on Q and behavior.  This identifies which available,
   occupied modes are actually read.

After fresh confirmation, the restricted operator for each live module/head set
will receive exact SVD/HOSVD and literal-atom analysis first.  Sparse or hierarchical
coding follows only if it reconstructs held-out interfaces and is stable under
component bootstraps and admissible factor gauges.  Weight-supported but unoccupied
modes remain latent capability; occupied but intervention-null modes remain
correlated state.

## Throughput, system state, and next experiment

Since 19:22, the circuit lane landed the exact H/R artifact, two CPU crossfits, v21
capability, one exhaustive causal atlas, a prospective atlas prior, and an explicit
instrument audit.  One CPU receipt initially failed only at JSON serialization
after computation; the partial file was preserved and a NumPy-boolean conversion
was rerun without changing any method or bar.  Both managed runners are healthy.
The GPU queue is empty because the 136-forward atlas completed in 18 seconds; the
next GPU job is not queued until its prospective fresh-bank design is sealed.

`CIRCUIT_FOCUS: PASS` — exact H/R localization and exhaustive head/module causal
patching narrowed the reusable mechanism and exposed an invalid endpoint.

`CEREMONY_BUDGET: PASS` — the H/R artifact was reused for two zero-forward tests;
the only full GPU atlas was exhaustive and completed cheaply.

`NOVELTY_LESSON_GATE: PASS` — earlier v15 module/head atlases, unconditional greedy
failure, weight-only capability red-team, H/R crossfits, and Shapley effect-game
notes were checked.  The next run is not licensed as a v21 repair or a repeated
greedy fit.

Ranked continuation:

1. Build and capability-gate a history-disjoint v22 bank before any intervention.
2. Freeze the replicated four-head set plus the complete-module frontier from v21.
   On v22 measure exact reader-contracted Q, behavior, P/C component leakage, exact
   self-patches, and donor closure as separate—not conflicting—controls.
3. If the complete selected union is selective but diffuse, run frozen forward
   greedy addition and backward necessity.  Enumerate exact source-effect Shapley
   interactions only for a final shortlist of at most 12 routes.
4. Translate the confirmed set into its literal restricted weight tensor, compute
   exact spectra/atoms, and overlay activation occupancy without refitting the basis.
5. Compose the confirmed writer state with the stable reader and require joint
   replacement/removal on a further held-out corpus before calling it a compiled
   circuit.

No current result satisfies the predictive, composable, manipulable tensor-program
stopping condition.

# Rung 529: split equality transitions into shared consensus and private remainder

**Registered:** 2026-09-03 11:22 UTC

**Status:** prospective design; no rung-529 state insertion or CE outcome has been computed

**Claim level:** held-out shared/private causal-state decomposition, not internal component minimality or compression

## Question and opposing hypotheses

Rung 528 found that the four complete post-MLP12 state changes are not proportional. The two sign-corrected actions
were nevertheless close: `N:Z7` circuit cosines were `.914/.949` and `N:Z8` were `.919/.931`, while all task
cosines exceeded `.995`. A post-result calculation using only those already-open response fingerprints found that the
mean of the other three aligned actions predicts `Z7` with `.313/.314` relative error, better than every individual
donor, and that the remaining private response does not reproduce across document halves.

Rung 529 tests two opposing explanations:

1. **Shared-plus-private computation:** averaging independent implementations cancels implementation-specific
   residue and recovers a state that is sufficient for the common equality computation. Replacing a target state by
   the other three actions' consensus should outperform every single donor on new physical runs, new documents, and
   held-out circuits. The private remainder alone should lose the equality effect.
2. **Response-space averaging artifact:** the apparent improvement exists only after aggregating CE fingerprints.
   Averaging the actual 1,152-dimensional boundary states through the nonlinear suffix will not improve on the best
   single donor or will damage unrelated behavior.

This changes the candidate object. It does not weaken rung 528's `.35` error bar, refit a failed pair, choose only
one favorable continuation, reduce rank, or optimize reconstruction.

## Fixed actions, scales, data, and continuations

Reuse rung 528's checkpoint, score-absent action, raw post-MLP12 boundary, four correct-gauge actions, four suffix
continuations, task masks, and 62 circuit families without alteration. The alignment scales into native-action units
are frozen from rung 528 discovery:

- `beta_N = 1`;
- `beta_P = 0.595594568993135`;
- `beta_Z7 = 0.8070768390655048`;
- `beta_Z8 = 0.7212548186259912`.

The uncorrected actions `W7/W8` remain wrong-sign controls. Data phases are:

- discovery: documents `0:248`, split `0:124` and `124:248`, with the same 32 discovery circuits;
- confirmation: documents `248:496`, split `248:372` and `372:496`, with the same 32 circuits;
- documents `496:500` unused;
- validation: documents `500:1000`, split `500:750` and `750:1000`, with the 30 circuit families excluded from
  discovery and confirmation.

No target action may be dropped. All four `N/P/Z7/Z8` are tested.

## Exact shared/private state construction

For action `a`, let `delta_a` be its raw post-MLP12 residual change relative to the score-absent boundary, exactly as
in rung 528. Put every transition in native-action units:

`s_a = beta_a * delta_a`.

For target action `a`, construct the leave-one-action-out consensus without using its state:

`consensus_a = (1 / beta_a) * mean_{b != a}(s_b)`.

Its private remainder is defined exactly:

`private_a = delta_a - consensus_a`.

Thus `delta_a = consensus_a + private_a` by construction. This algebraic identity is not the claim. The claim
requires finite downstream sufficiency of `consensus_a` and finite loss of the equality effect when only
`private_a` remains.

Every state is inserted into the common score-absent post-MLP12 boundary and the real layers13--17 suffix recomputes.
Run all four continuations from rung 528: native, attention14 held absent, MLP17 held absent, and both held absent.
The four arms and their two-component factorial interaction are never averaged away for a gate.

## Single-donor and wrong-sign controls

For every target `a`, run all three leave-one-source-in replacements

`single_{a<-b} = (beta_b / beta_a) * delta_b`, for every `b != a`.

These are physical state insertions, not response-vector estimates. Rung 529 can claim a consensus advantage only if
it beats every single donor, so no favorable donor is selected after seeing the result.

Wrong-sign consensus controls replace each eligible `Z7` or `Z8` source in the leave-one-out mean by `W7` or `W8`
without changing its positive alignment scale. If the target itself is `Z7`, its consensus contains no Z7 source,
so only the W8 replacement is applicable; symmetrically Z8 has only W7. N and P have both controls. Every applicable
wrong-sign control is run in discovery. Sixteen fixed circuit-coordinate permutations with seeds
`529300..529315` give a separate chance-alignment control.

## Effects and discovery gates

For every inserted state, continuation, document half, and circuit tag, compute the same member-minus-matched-control
CE effect as rung 528 relative to the score-absent run. Task coordinates remain near match, far match, one earlier
match, and multiple earlier matches.

A target becomes a discovery candidate only if all clauses hold:

1. The native target and consensus are live in every continuation: circuit RMS at least `.0005` nat and concatenated
   task norm at least `.00025` nat in both document halves.
2. Consensus versus native target has concatenated circuit cosine at least `.90` and relative error at most `.35`
   on D0; on D1, without refitting, cosine is at least `.80` and error at most `.50`.
3. Task cosine is at least `.70` and task error at most `.65` in both halves. Every continuation separately has
   circuit cosine at least `.65` on D0 and `.55` on D1.
4. On D0, consensus relative error is at least `.05` lower than **every** single-donor error. On D1 it is no more
   than `.02` worse than the best frozen single donor.
5. Consensus D0 cosine exceeds every applicable wrong-sign control and the permutation 95th percentile by at least
   `.10`.
6. Reconstructing `consensus_a + private_a` before the one BF16 boundary rounding reproduces the target boundary
   exactly; every consensus, private, single, and wrong-sign edit is nonzero and all calls reconcile.

Retain every passing target. One through four opens confirmation; zero is a registered scientific null. There is no
ranking or best-target selection.

## Confirmation and held-out prediction

For each candidate, freeze its identity, alignment scales, its lowest-error D0 single donor, and its strongest D0
wrong-sign control. Recompute native, consensus, private, the frozen single donor, and the frozen wrong control on
documents `248:496`.

In each half and pooled, consensus must predict the native target with circuit cosine at least `.75`, relative error
at most `.55`, task cosine at least `.70`, and task error at most `.65`; every continuation cosine must be positive.
Its error must be at least `.03` below the frozen single donor, and its cosine must exceed the frozen wrong control by
`.10`. At least one candidate must pass without reselection.

Only confirmation passers open documents `500:1000` and the 30 held-out circuits. The same rules apply with circuit
cosine at least `.70` and error at most `.60`. The consensus must again beat the frozen single donor by `.03` error
and the wrong control by `.10` cosine in both document halves.

## Sufficiency, selective removal, and interpretation

For every validated candidate under the native continuation and in both confirmation and validation halves:

- the full native action's all-copy effect has absolute magnitude at least `.002` nat;
- replacing the full transition by `consensus_a` meets the held-out circuit/task prediction bars above;
- inserting only `private_a` retains at most25% of the full action's absolute all-copy effect; and
- the absolute off-target difference between private-only and full-action runs is at most `.001` nat.

These clauses make consensus a sufficient shared computation and private-only insertion a selective removal of that
shared computation. A pass identifies a distributed shared/private state at one boundary. It does not yet tell us
which Q/K/value factors, attention heads, or bilinear MLP terms implement it, and it saves zero deployed values. The
next step after a pass must split the validated consensus internally while preserving its full continuation
fingerprint.

## Predictions and frozen routes

- **A — exact live instrument:** all boundary identities, reconstruction, liveness, supports, and call counts pass.
- **B — consensus beats every singleton:** at least one all-target discovery candidate passes clauses1--5.
- **C — new-document physical consensus:** at least one B candidate confirms with the frozen singleton/control.
- **D — held-out circuits and documents:** at least one confirmed candidate validates on all 30 unopened circuits.
- **E — sufficient and selectively removable shared state:** at least one D candidate passes the private-only task
  and off-target clauses.

`strong_null = not (A and B and C and D and E)`.

- A false: repair only the instrument.
- A true/B false: averaging action states does not expose a shared component at this boundary; close the consensus
  route without changing bars or using a learned rank.
- B true/C false: the consensus advantage is discovery-specific.
- B/C true/D false: it does not identify a state across circuit families.
- B/C/D true/E false: it is predictive but not a selectively removable equality computation.
- A--E true: identify each passer as a shared/private transition decomposition and split its shared state internally
  using exact attention/MLP interactions.

No route licenses threshold tuning, target dropping, response-only evidence, quantization, or a compression claim.

## Literal price

Discovery runs 124 forwards per four-document batch: one direct native, one score-absent, four action-boundary runs,
two wrong-sign boundary runs, 12 non-native target-continuation runs,
16 consensus continuations, 48 single-donor continuations, 16 private-only continuations, and 24 applicable
wrong-sign consensus continuations. Its unconditional price is

`62 * 124 = 7,688 forwards`.

If `k` targets pass, confirmation runs the base six forwards, the `u<=2` distinct wrong-sign boundary runs needed
by their frozen controls, plus three non-native target-continuation runs and consensus, private, frozen singleton,
and frozen wrong control under four continuations for each target: `62 * (6 + u + 19k)`. If `q` confirm,
validation costs `125 * (6 + v + 19q)`, where `v<=2`. The maximum at `k=q=4` is `23,396` forwards, zero backwards, zero fitted values beyond the three
already frozen R528 scales, zero deployed values added or removed, and no compression claim.

**Pre-outcome accounting correction, 2026-09-03 11:25 UTC:** the first draft counted the 24 wrong-sign consensus
insertions but omitted the two `W7/W8` model forwards needed to obtain their physical boundary states. No R529 GPU
outcome existed when this was corrected. Scientific arms, gates, data, and predictions are unchanged; only the
literal execution price increased from `6,820/19,910` to `6,944/20,408` unconditional/maximum forwards.

**Second pre-outcome accounting correction, 2026-09-03 11:27 UTC:** the first correction still omitted the three
non-native target-continuation runs per target. The target's ordinary continuation comes from its already-counted
action-boundary run, but comparisons after holding out A14, M17, or both need three additional physical suffix
runs. The final audited price is therefore `7,688` unconditional and `23,396` maximum forwards. Again, no GPU
outcome existed, and no scientific arm, threshold, data split, or prediction changed.

## Frozen authorities

- R528 result: `f931e5fb6f618b002203ce1e870a8ad4442ed3a38a7475809754ab2de91554b6`
- R528 sufficient statistics: `c17db82832a76daba23f74e57e75abc258093c6820c79c93a62d8d29b6143d38`
- R528 terminal audit: `a3843265eb15a1fe6771c848843dfafe5703d100933331aab114dbf0e2286f71`
- leave-one-action-out response diagnosis:
  `207ff8cfdac919ac4a817564450a6339a2b420d6e91a6f545a9723ed6aded67c`

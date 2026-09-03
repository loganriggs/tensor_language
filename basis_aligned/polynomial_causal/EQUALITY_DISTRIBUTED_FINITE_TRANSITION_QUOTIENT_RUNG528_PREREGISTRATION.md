# Rung 528: do four equality-score implementations create the same distributed state transition?

**Registered:** 2026-09-03 10:52 UTC

**Status:** prospective CPU-first design; no rung-528 model outcome has been computed

**Claim level:** held-out, continuation-defined causal-state equivalence; not component minimality or compression

## Question

The native equality score `N` and three replacement implementations `P`, `Z7`, and `Z8` have already been shown to
produce the same useful attention-level action when the two negative-family replacements use the correct sign. The
unresolved question is whether that common computation persists through several modules as one interchangeable
state, or whether each implementation causes a different downstream trajectory that happens to solve the same task.

For each implementation, rung 528 takes the complete residual-stream change between score-present and score-absent
runs immediately after MLP12. It asks whether those changes are interchangeable when the actual layers13--17 suffix
runs, including when attention14, MLP17, or both are independently held to their score-absent writes.

“Same state” therefore means that a positive scale learned on one document half predicts finite downstream effects on
the other half, survives new documents and the 30 circuit families excluded from selection, and works when the
scaled residual change is physically inserted into the model. Activation cosine, rank, reconstruction error, SAE
loss, and parameter count are not selection metrics.

## Duplicate-work boundary

- Rungs 498 and 501 compared equality-score actions at the attention interface and at the MLP9 reader. They did not
  test a complete later residual-state transition.
- Rung 505 tested one code-derived five-site set and found that its sign pattern did not transfer to natural text.
  Attention14 and MLP17 are used here only as fixed continuation interventions, not asserted to be a natural-text
  group.
- Rung 506 compared pairs among 19 complete later attention/MLP writes and found no pairwise whole-write relation.
- Rungs 507--515 compared individual exact bilinear terms, branches, or consumer terms and found no reusable
  pairwise relation at those grains.
- Rung 528 instead treats the cumulative state change across the entire attention-score-to-post-MLP12 interval as
  one candidate transition. It can include several public writes and all their interactions. It neither selects a
  native component pair nor repeats a source-term screen.

## Fixed model, data, actions, and circuit partitions

Use the same pinned bilin18 checkpoint, 1,000 natural documents, score scales, copy-task masks, and 62 named circuit
families as rungs 499--506. The score actions are fixed:

- `N`: native L8H4 score;
- `P`: the L5H5 score supplied to L8H4 with its frozen positive scale;
- `Z7`: the L7H3 score supplied to L8H4 with its frozen scale and a minus sign;
- `Z8`: the L8H3 score supplied to L8H4 with its frozen scale and a minus sign.

The common score-absent action removes L8H4's equality-score term. The unnegated `W7` and `W8` actions are fixed
wrong-sign controls and cannot become candidates.

Data phases are fixed before outcomes:

- discovery: documents `0:248`, reported separately as `0:124` and `124:248`;
- confirmation: documents `248:496`, split as `248:372` and `372:496`;
- documents `496:500` remain unused;
- final validation: documents `500:1000`, split as `500:750` and `750:1000`.

The 32 circuit tags with top-level roots `{0,2,4,6,8,18}` are discovery/confirmation coordinates. The other 30 tags
with roots `{1,3,5,7,11,13,23}` remain unopened until final validation. Every circuit coordinate is the mean CE
change on member positions minus the mean change on matched non-member positions from the same task slice.

## Exact boundary state

Let `x_a^12` be the unnormalized residual stream immediately after MLP12 under score action `a`, before block13's
residual mixing and RMS normalization. Let `x_0^12` be the corresponding score-absent residual. Define the complete
finite transition

`delta_a = x_a^12 - x_0^12`.

This is not the MLP12 write alone. It contains every action-dependent change accumulated through attention8,
MLPs8--12, intervening attention/MLP blocks, normalization effects, and their interactions.

The runner must expose this raw boundary inside a source-closed copy of the observed-model recurrence. For every
action it then starts from the score-absent trajectory and replaces the post-MLP12 residual by

`x_0^12 + gamma * delta_a`,

where `gamma=1` for the native transition and is a discovery-frozen signed scale for a cross-action substitution.
The carried first-value attention state and embedding skip state must match the score-absent/action trajectories;
the instrument must prove they are identical at this boundary or fail A. With `gamma=1`, inserting `delta_a` must
reproduce the original action's layers13--17 logits to the registered numerical tolerance. This self-replay check is
what makes the boundary state sufficient for the tested suffix rather than an informal activation patch.

## Four fixed downstream continuations

For each inserted transition, run four complete suffixes:

1. `native`: no later write is changed;
2. `without_A14`: replace attention14's write by its same-document score-absent value;
3. `without_M17`: replace MLP17's write by its same-document score-absent value;
4. `without_both`: make both replacements.

All later layers recompute normally except the named replacement writes. These four arms are a two-component
factorial. Their interaction is computed as

`I = effect(without_both) - effect(without_A14) - effect(without_M17) + effect(native)`.

No small-interaction assumption is made. The four continuations are retained separately in every equivalence test;
the interaction is reported to show how much a usual single-mediator result would miss.

## Downstream fingerprints and discovery relation

For action `a`, continuation `c`, document half `h`, and circuit tag `j`, define

`F[a,c,h,j] = (member-minus-control CE after inserting delta_a) - (member-minus-control CE with delta=0)`.

Define `T[a,c,h]` analogously on the four fixed copy-task cells `(near, far, one earlier match, multiple earlier
matches)`. Concatenate the four continuations when fitting, but retain every continuation for the gates below.

Only the three registered pairs `(N,P)`, `(N,Z7)`, and `(N,Z8)` are candidates. For each pair `(N,b)`, fit one signed
scale on discovery half0:

`beta(N <- b) = dot(F[b], F[N]) / dot(F[b], F[b])`,

using the concatenated four-continuation by 32-circuit vector. A pair is a discovery candidate only if:

1. every action/continuation circuit RMS is at least `.0005` nat and every concatenated task norm is at least
   `.00025` nat;
2. `0.25 <= beta <= 4`; the correct sign gauge is already built into `Z7/Z8`, so allowing another sign flip here
   would make the wrong-sign controls vacuous;
3. on half0, the concatenated circuit cosine is at least `.90` and the scaled relative residual at most `.35`;
4. without refitting on half1, circuit cosine is at least `.80` and residual at most `.50`;
5. on every individual continuation, circuit cosine is at least `.65` on half0 and `.55` on half1, with the scale
   sign preserved;
6. the concatenated task cosine is at least `.70` and scaled residual at most `.65` in both halves; and
7. the candidate's half0 cosine exceeds the 95th percentile of 16 independently permuted circuit-coordinate
   controls and both wrong-sign controls by at least `.10`.

Keep every passing pair. One through three opens scaled substitution; zero is a registered null. There is no ranking,
nearest-neighbour choice, or threshold change.

## Physical scaled substitution on the second discovery half

Response proportionality alone is insufficient because the suffix is nonlinear. For each discovery candidate,
perform both actual state substitutions on documents `124:248` under all four continuations:

- substitute `beta * delta_b` for `delta_N`;
- substitute `(1/beta) * delta_N` for `delta_b`.

Compare each substituted run with the corresponding native-transition effect. In each direction, the concatenated
circuit cosine must be at least `.80`, scaled-response residual at most `.50`, task cosine at least `.70`, and task
residual at most `.65`. Every continuation must keep positive circuit cosine. Only bidirectional passers proceed.

This is a real intervention on the complete boundary state. It does not assume that multiplying the transition by
`beta` multiplies CE by `beta`.

## Confirmation and held-out circuit validation

On documents `248:496`, recompute the native transition fingerprints and both scaled substitutions without changing
the pair or scale. A pair confirms only if, in each half and pooled:

- native-transition circuit cosine is at least `.75` and residual at most `.55`;
- both physical substitutions have circuit cosine at least `.75` and residual at most `.55`;
- task cosine is at least `.70` and task residual at most `.65` for the native relation and both substitutions; and
- every continuation has positive circuit cosine in both substitution directions.

Only confirmed pairs open documents `500:1000` and the 30 held-out circuit families. The same rules apply there,
except the minimum circuit cosine is `.70`, maximum residual `.60`, and every continuation must again remain
positive in both document halves. No pair may be added on confirmation or validation.

If two or three `N`-centered edges validate, compute the implied scale between every non-native pair. A shared
four-action state is reported only if direct finite substitutions between those pairs meet the validation bars and
the product of scales around every available cycle differs from one by at most25%. Otherwise report only the
validated `N`-centered pairs.

## Selectivity and interpretation

A validated pair counts as a causal-state relation only if, under the native continuation in both confirmation and
validation, each action's all-copy effect has absolute magnitude at least `.002` nat and is at least three times its
absolute off-target effect. This prevents a generic loss perturbation from being named a task circuit.

A pass establishes that two score implementations create an interchangeable distributed transition at one exact
boundary, as judged by multiple downstream continuations, new documents, held-out circuit families, and physical
state exchange. It does not show that the whole transition is minimal, explain its internal MLP/head decomposition,
or save deployed parameters. A passing transition becomes the target to split internally while preserving the
validated continuation behavior.

## Predictions and frozen routes

- **A — exact live instrument:** model/action hashes match; raw boundary capture and insertion are exact; all
  self-replays meet tolerance; every transition and continuation edit is live; supports and calls are exact.
- **B — discovery relation:** at least one of the three action pairs passes every discovery and wrong-sign control.
- **C — physical and new-document prediction:** at least one B pair passes bidirectional substitution on discovery
  half1 and repeats on confirmation without refitting.
- **D — held-out circuits and documents:** at least one fixed pair passes both validation halves on the 30 unopened
  circuit families.
- **E — selective distributed state:** at least one D pair meets the task-versus-off-target rule. Any claimed
  multi-action quotient also satisfies direct pair substitutions and scale-cycle consistency.

`strong_null = not (A and B and C and D and E)`.

- A false: repair only the boundary instrument; no state result is interpretable.
- A true/B false: the four score actions do not share one post-MLP12 continuation-defined state at this scale. Close
  this boundary; do not tune rank, scales, or circuit bars.
- B true/C false: downstream-response similarity is not physically interchangeable or does not repeat on new
  documents. Do not call it a state relation.
- B/C true/D false: the relation depends on the selected circuit families and is not identified.
- B/C/D true/E false: retain it only as a broad downstream equivalence, not an equality circuit.
- A--E true: split the validated distributed transition using exact attention/MLP interactions while preserving its
  complete continuation fingerprint and physical interchange.

No route licenses a rank sweep, quantization, reconstruction-only compression, weaker bars, favorable continuation
selection, or a claim that native head/MLP boundaries are canonical.

## Literal price and stopping points

Batch size is four. A standard batch uses one direct native check, one score-absent boundary/capture run, four
score-action boundary runs, and `4 actions * 4 continuations = 16` unit-transition suffix runs: 22 full forwards.
Discovery additionally measures two wrong-sign controls under four continuations, costing ten more forwards per
batch. Thus the unconditional discovery price is

`62 batches * 32 = 1,984 forwards`.

If `k` pairs pass discovery, bidirectional scaled substitutions on the second discovery half cost
`31 * 8k = 248k` forwards. Confirmation costs `62 * (22 + 8k)` forwards. If `q` pairs confirm, validation costs
`125 * (22 + 8q)` forwards. With the maximum `k=q=3`, the ceiling is `11,330` forwards, zero backwards, at most three
fitted positive scalars, zero deployed values added or removed, and no compression claim. The runner must reconcile direct,
boundary-capture, continuation-patch, and substitution call counts separately and report peak GPU memory.

## Frozen dependency hashes

- rung505 source: `0c5f6679ec40cb02bd6af1e28b0b41ca2ad7967fd4b6c9d73a4f388153f3e4de`
- rung505 result: `3720a2feb24fc5ec4554d858a00a576a1fcd44f0e789d2b728e66483d7d8d1a1`
- rung506 source: `9a17e28312a0e7214e5fc587123e3267e2650b382f3a40daf12ad1a380b1d004`
- rung506 result: `f86e5f0303ab0616ea14e3141fd09886ca54d326e8d83ea6c8c13a62f66db75e`
- rung501 source: `97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077`
- rung501 result: `b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f`
- rung510 source: `7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a`
- rung510 result: `16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177`

# Rung 506: group later writes by their finite downstream effects on natural text

Status: prospectively frozen after rung 505 and before any rung-506 write-removal outcome is computed. Rung 505's
five-site code-derived program failed its registered natural-text sign pattern, so no rung-505 outcome may select a
site, pair, threshold, or circuit family here.

## Question

Rung 505 established two facts that must not be conflated. The four correctly oriented implementations of the
equality score remain causally interchangeable at the attention interface, but the old code-selected group
`{MLP8, MLP9, MLP12}` does not have the same causal context pattern on natural text. The next object must therefore be
defined from natural-text computation rather than inherited from code or from native head/MLP boundaries.

This rung asks whether any two of the 19 attention/MLP writes after attention head L8H4 represent the same
downstream causal variable. “Same variable” has an operational meaning:

1. removing either write changes the same fixed downstream circuit examples in the same direction;
2. that agreement holds for the native equality score, the positive L5H5 replacement, and the correctly negated
   L7H3 and L8H3 replacements;
3. it repeats on new documents and on circuit families excluded from discovery; and
4. removing the two writes together follows one simple, discovery-frozen composition rule.

This directly targets cross-module grouping, held-out prediction, and selective manipulation. It is not a rank,
activation-similarity, sparse-autoencoder, or compression experiment. A pass identifies a causal state relation at
the whole-write boundary; it does not yet prove that either complete native module is minimal. A null routes to
splitting attention or MLP writes into their exact query/key/value or bilinear source-pair terms.

## Fixed data and downstream coordinates

Use the same hash-bound 1,000 natural documents and the same 62 already named circuit tags as rungs 475--499. Each
circuit tag supplies two token masks:

- `member`: tokens belonging to that circuit's examples;
- `slice_control`: matched comparison tokens from the same task slice that are not circuit members.

Top-level circuit families with roots `{0,2,4,6,8,18}` give 32 discovery tags. Roots `{1,3,5,7,11,13,23}` give 30
held-out tags. These partitions were defined before this rung.

The document phases are fixed:

- discovery: documents `0:248`, with internal repeats `0:124` and `124:248`;
- confirmation: documents `248:496`, with internal repeats `248:372` and `372:496`;
- documents `496:500` are unused so no batch crosses a boundary;
- held-out circuit-family validation: documents `500:1000`, reported separately as `500:750` and `750:1000`.

Before outcomes, the smallest `member` support per tag is 27 in each discovery repeat, 16 and 23 in the confirmation
repeats, and 32 and 46 in validation. The corresponding minimum `slice_control` supports are 193, 208, and 343.
The six fixed copy-task masks are inherited from rung 505: all, near, far, one earlier match, multiple earlier
matches, and off-target.

## Fixed score actions and write sites

The recipient remains attention head L8H4. The four correctly oriented score actions and all scales are frozen from
the validated sign-gauge receipts:

- `N`: native L8H4 score;
- `P`: L5H5 score supplied to L8H4 with its positive scale;
- `Z7`: L7H3 score supplied with the frozen scale multiplied by `-1`;
- `Z8`: L8H3 score supplied with the frozen scale multiplied by `-1`.

No scale is fit here. Rung 505 already showed that the wrong-sign controls reverse the downstream effect, so they are
not rerun.

The 19 candidate write sites are fixed and complete from MLP8 through MLP17:

`m8, a9, m9, a10, m10, a11, m11, a12, m12, a13, m13, a14, m14, a15, m15, a16, m16, a17, m17`.

For each batch, first run the score-absent L8H4 action and capture all 19 resulting writes. Removing site `s` under
score action `a` means replacing its current write with that same-document score-absent write, then recomputing all
later layers normally. For a pair `{s,t}`, patch both writes to their score-absent values. This is a real finite
intervention; later nonlinear interactions are retained.

## Computation

Let `L_a(x)` be next-token cross-entropy at token `x` with score action `a` intact. Let `L_{a,s}(x)` be the loss after
removing write `s`, and `L_{a,st}(x)` after removing both `s` and `t`.

For any token mask `M`, the finite effect of a removal is

`e_a(s; M) = mean_{x in M}[L_{a,s}(x) - L_a(x)]`.

Positive means the write helped those predictions; negative means the write hurt them. No first-order gradient is
used.

For circuit tag `j`, define the downstream circuit fingerprint coordinate

`F_a(s)[j] = e_a(s; member_j) - e_a(s; slice_control_j)`.

Subtracting the matched control asks whether removal affects that circuit more than comparable non-member tokens.
`F_a(s)` has 32 coordinates in discovery/confirmation and 30 different coordinates in held-out validation. The task
fingerprint `U_a(s)` is the four-vector `(near, far, one earlier match, multiple earlier matches)`.

Cosine is the dot product divided by the two Euclidean norms. A norm ratio is the larger norm divided by the smaller.
These compare causal effect patterns, not activation directions.

For a selected pair, the exact finite interaction is

`I_a(s,t) = F_a({s,t}) - F_a(s) - F_a(t)`.

This is the part of the joint causal effect missed by adding the two single-write effects. It is the finite analogue
of the hidden mediator interactions described by Vaidyanathan et al.; it includes all recomputation downstream of
the two patches.

## Discovery rule: retain every qualifying pair

A site is eligible only if, for every score action:

- its circuit-fingerprint root-mean-square is at least `.0005` nat;
- its two discovery-repeat fingerprints have cosine at least `.50` and norm ratio at most `3`; and
- its pooled fingerprint under `P`, `Z7`, and `Z8` has cosine at least `.70` with `N` and norm ratio at most `3`.

An unordered pair of eligible sites is a discovery edge only if, for every score action:

- pooled circuit-fingerprint cosine between the sites is at least `.85`, with norm ratio at most `3`;
- the pair cosine is at least `.60` in each discovery repeat; and
- pooled task-fingerprint cosine is at least `.60`.

Retain every passing edge; do not rank them. The discovery instrument is identifying only if it returns between one
and eight edges. Zero edges is a scientific null at the whole-write grain. More than eight is a non-identifying
observation basis, not permission to take the best eight.

## Confirmation, held-out circuit families, and composition

Only the discovery edges are evaluated as pairs. A discovery edge confirms on documents `248:496` only if, for every
score action:

- pooled circuit-fingerprint cosine is at least `.75`, each repeat cosine is at least `.50`, and norm ratio is at
  most `3`;
- each site's score-source fingerprint remains at least `.60` cosine with its native-score fingerprint; and
- task-fingerprint cosine between the two sites is at least `.50`.

No new edge may be added on confirmation. At least one discovery edge must confirm; validation opens only then.

On the 30 held-out circuit families and documents `500:1000`, a confirmed edge validates only if every score action
has pooled pair cosine at least `.70`, positive pair cosine in both document halves, norm ratio at most `3`, and task
cosine at least `.50`. Again, no edge is added.

The pair's finite joint effect must also obey a simple composition rule chosen without confirmation or validation.
Concatenate the four score-source discovery fingerprints. In this order, assign the first applicable rule:

1. `additive`: `||I|| / ||F({s,t})|| <= .25`, predicting `F({s,t}) = F(s)+F(t)`;
2. `left redundant`: `||F({s,t})-F(s)|| / ||F({s,t})|| <= .25`, predicting the left singleton;
3. `right redundant`: the corresponding right-singleton rule;
4. `one-scalar interaction`: fit the one scalar
   `beta = dot(I, F(s)+F(t)) / ||F(s)+F(t)||^2`; require `|beta| >= .25`, `-.8 <= beta <= 2`, and residual
   `||I-beta(F(s)+F(t))||/||I|| <= .50`; predict `(1+beta)(F(s)+F(t))`.

The ordered site names decide left versus right, so this rule is deterministic. An edge with no discovery rule does
not compose. On confirmation and validation, the frozen prediction must have cosine at least `.70` and relative
residual at most `.65` for every source's circuit fingerprint; its task-vector prediction must have cosine at least
`.60` and relative residual at most `.75`.

Finally, selective manipulation requires, for every score source on both confirmation and validation, absolute
all-copy pair-removal effect at least `.002` nat and at least three times the absolute off-target effect. This prevents
a generic loss perturbation from being called a circuit.

## Literal execution price

Batch size is four. Singleton discovery costs `62*(3 + 4*20) = 5,146` full forwards: native, exact analytical replay,
one score-absent capture, then intact plus 19 singleton removals for each of four sources.

If discovery returns `k` edges with `1 <= k <= 8`, discovery pair effects cost `62*(1+4k)` more, and confirmation
costs `62*(3+4*(20+k))`. If `q` of those edges confirm, validation costs `125*(3+4*(20+q))`. Thus a run reaching
validation executes exactly

`20,729 + 496k + 500q` full forwards,

at most `28,697` when `k=q=8`, with zero backward passes, zero fitted vectors, at most one fitted scalar per edge,
zero deployed parameters added, and zero deployed parameters saved. The receipt must verify calls, captures, patches,
all supports, output finiteness, exact native replay, attention-factor reconstruction, and a nonzero edit at every
patched site/source that is interpreted.

## Registered predictions and nulls

### A. Exact, live instrument

All input hashes and fixed partitions hold. Analytical native replay matches direct native logits exactly up to the
existing `1e-12` relative-squared bound; attention-factor reconstruction is at most `1e-10`; calls/captures/patches
equal the appropriate conditional formula; all masks meet the support floors above; and every interpreted score and
write edit is finite and nonzero.

### B. The four score actions remain calibrated on the new discovery and confirmation documents

For `P`, `Z7`, and `Z8` versus `N`, in every reported document repeat, all-copy recovery lies in `[.65,1.40]`,
per-document all-copy effect cosine is at least `.85`, and off-target change is at most `.01` nat. B must pass before
an edge is interpreted.

### C. At least one downstream causal-state edge confirms

The exact no-ranking discovery and confirmation rules above leave at least one edge. This is the cross-module
grouping claim on known circuit families and new documents.

### D. At least one edge validates on held-out circuit families and documents

At least one confirmed edge passes every held-out circuit-family and task rule above. This is identification rather
than an in-sample screen.

### E. At least one validated edge has predictable composition and selective removal

At least one validated edge obeys one discovery-frozen finite composition rule on confirmation and validation and
passes the all-copy versus off-target selective-removal test. This licenses a causal state relation at the whole-write
boundary, not an adopted replacement or compression.

The strong null is A false, B false, zero discovery edges, more than eight discovery edges, zero confirmation edges,
zero validation edges, or no validated edge with predictable finite composition and selective removal.

## Frozen result routes

- A false: repair only the instrument; no scientific result is interpretable.
- A true/B false: the score-action calibration is corpus- or partition-dependent; preserve the sign gauge at its
  validated boundary and stop this downstream assay.
- B true/C false with zero edges: no whole attention/MLP write pair is the same downstream state under these actions;
  next split fixed candidate writes into exact attention factors or bilinear source-pair terms.
- B true/C false with more than eight edges: the 62-circuit observation basis is too coarse; enrich the downstream
  actions before grouping, without selecting eight edges.
- C true/D false: discovery similarity does not generalize to held-out circuit families; preserve the screen and
  change the downstream coordinates.
- C/D true/E false: the relation is predictive but not yet compositional or selectively manipulable; preserve it as
  an observation-level equivalence, not a circuit.
- A--E true: identify every passing edge as a whole-write causal-state relation. The next rung must split those writes
  internally and build an executable joint replacement before any adoption or simplicity claim.

No result permits lowering a bar, ranking an overlarge edge set, selecting from rung-505 effects, treating native
module boundaries as minimal, or substituting rank/reconstruction/quantization for causal grouping.

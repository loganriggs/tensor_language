# Rung 505: does one sign-gauged equality score drive one downstream program?

Status: prospective design frozen after rung 504 and before any rung-505 subset outcome is opened. This is the
registered `change_downstream_observation` route from rung 504. It reuses a fixed causal program identified earlier;
it does not select sites, product terms, ranks, or thresholds from the new outcomes.

## Question and duplicate-work boundary

Rung 504 showed that the equality-score response at MLP9 is not itself a causal copy mediator at one- or two-source
removal grain. The complete pair table implies that MLP8 carries only `.0064/.0141` of the actual finite copy benefit
despite carrying `.254/.268` of the local MLP9 write response. Therefore a float32 local decomposition could explain
arithmetic but cannot repair the deployed causal failure.

Rungs 465--466 already identified a larger program on 192 code documents:

- task-shaped group `T = {MLP8, MLP9, MLP12}`;
- broad-suppression group `G = {attention14, MLP17}`; and
- their finite interaction `I = v(T union G) - v(T) - v(G)`.

Rung 466 used native L8H4 and an L5H5 score transplant. It did not test natural text or the subsequently validated
sign gauge. Rung 505 therefore asks a genuinely new question: on natural text, do the native score, the positive
L5H5 supply, and the correctly negated L7H3/L8H3 supplies all drive the same fixed downstream program? A pass groups
the upstream score implementations with their downstream consumers. A failure shows that score-level interchange
does not imply an interchangeable downstream realization.

## Frozen data, actions, sites, and controls

Use only natural documents `500:1000` from the exact 1,000-document row authority used by rungs 499--501 and the
sign-gauge validation. These documents are already open for whole-action copy and MLP9 outcomes, but their five-site
subset outcomes have never been run. Report fixed halves `500:750` and `750:1000` separately.

The recipient is fixed as L8H4. The five fixed sites and groups are inherited without change from rung 466:

`T = {m8, m9, m12}`

`G = {a14, m17}`.

Run the complete `2^5 = 32` removal factorial for each of four score sources in the ordinary early-present
background:

1. `N`: native L8H4;
2. `P`: L5H5 score supplied to L8H4 with its frozen positive scale;
3. `Z7`: L7H3 score supplied to L8H4 with the frozen scale multiplied by `-1`;
4. `Z8`: L8H3 score supplied to L8H4 with the frozen scale multiplied by `-1`.

The two negated actions and their scales are fixed by the validated sign-gauge receipts; no scale is fit on rung 505.
The score-absent L8H4 trajectory is common to all four sources in this background. Capture its 19 later attention/MLP
writes from MLP8 through MLP17. Removing a site means replacing that site's write by the same-document score-absent
write and then letting all later layers recompute normally. This is the exact intervention used by rung 466.

Also run the unnegated L7H3 and L8H3 supplies as wrong-sign controls, but only for the empty, `T`, `G`, and `T union G`
subsets. They are not candidates and cannot be selected. Their whole-action effects were anti-aligned in the prior
sign-gauge experiment; they test whether the downstream program is specific to the correct gauge orientation.

The fixed task cells are `all copy`, `near copy`, `far copy`, `one earlier match`, `multiple earlier matches`, and
`off target`, computed from token identity exactly as in rung 499. The four-vector order used for program comparisons
is `(near, far, one earlier match, multiple earlier matches)`.

## Computation

Let `E_a(S,c)` be the copy benefit in context cell `c` when source action `a` is present and the sites in subset `S`
are replaced by their score-absent writes. The benefit is score-absent cross-entropy minus intervened cross-entropy,
so positive means the score action improves prediction.

Define the causal contribution of removed subset `S` as

`v_a(S,c) = E_a(empty,c) - E_a(S,c)`.

Positive `v` means those sites were helping the action; negative `v` means their presence suppressed an over-strong
effect. For every nonempty subset, also report the exact Möbius/Harsanyi finite interaction

`d_a(S,c) = sum_{B subseteq S} (-1)^(|S|-|B|) v_a(B,c)`.

The group interaction is

`I_a(c) = v_a(T union G,c) - v_a(T,c) - v_a(G,c)`.

To measure the entire later correction, also replace all 19 later writes by their score-absent values and define

`K_a(c) = E_a(empty,c) - E_a(all 19,c)`.

Cosines compare four-context directions. Projection magnitude is `dot(v,K)/dot(K,K)`. Norm ratios always mean the
larger Euclidean norm divided by the smaller and are therefore at least one.

## Literal price

Batch size is four, so documents `500:1000` give 125 batches. Each batch runs:

- one native model forward and one analytical no-edit replay;
- one score-absent forward that captures all 19 later writes;
- `32 + 1` forwards for each of four score sources: every five-site subset plus the all-19 direct control; and
- four forwards for each of two wrong-sign controls: empty, `T`, `G`, and `T union G`.

The exact price is therefore

`125 * (2 + 1 + 4*(32+1) + 2*4) = 17,875` full-model forwards,

zero backward passes, zero fitted parameters, zero deployed parameters added, and zero deployed parameters saved.
The receipt must report exact call, capture, and patch counts and peak GPU memory.

## Registered predictions

### A. Exact and live instrument

All frozen source/model/row hashes hold. The native and analytical no-edit forwards agree with maximum logit error
zero and relative-squared error at most `1e-12`. Equality-term reconstruction error is at most `1e-10`. The empty
subset reproduces each unpatched action exactly. Every declared patch fires once at the fixed site, has matching
shape/dtype/device, and at least one nonempty group patch changes a live write for every source. Task supports are
positive in both halves; all 17,875 forwards, capture calls, and patch calls match the formula exactly. No document
outside `500:1000`, no fitted scale, no rank, and no 62-circuit validation outcome is opened.

### B. The score actions remain calibrated inside this intervention harness

For `P`, `Z7`, and `Z8`, in both halves:

- all-copy recovery relative to `N` lies in `[.65, 1.40]`;
- the per-document all-copy effect cosine with `N` is at least `.85`;
- off-target cross-entropy change relative to `N` has absolute value at most `.01 nat`.

For both wrong-sign controls, the per-document all-copy effect cosine with `N` is at most `-.50` in both halves.
Failure of B invalidates downstream gauge comparisons even if a subset looks favorable.

### C. The old fixed program transfers from code to natural text

For both `N` and `P`, pooled and in both halves:

- `v(T)` has signs `near-/far+/one+/multiple-`; pooled norm is at least `.015 nat` and cosine with `K` at least `.80`;
- every entry of `v(G)` is negative, and its pooled cosine with `K` is below `.70`;
- `I` has pooled norm at least `.005 nat`;
- `v(T union G)` has pooled cosine at least `.80` with `K` and projection magnitude in `[.40, 1.60]`.

Between `N` and `P`, pooled cosines are at least `.85` for `T`, `G`, and `T union G`, and at least `.75` for `I`;
each corresponding norm ratio is at most `2.5`, and every half cosine is positive.

### D. The complete program is invariant across the validated sign gauge

For each of `Z7` and `Z8`, the same sign, norm, correction-alignment, and union-projection clauses in C hold. Against
both `N` and `P`, every pooled cosine is at least `.80` for `T`, `G`, and `T union G`, and at least `.70` for `I`;
each norm ratio is at most `3.0`, and every corresponding half cosine is positive. This is the main cross-boundary
grouping claim.

### E. Correct gauge orientation matters to the downstream program

For each negative-family donor, compare its correctly negated and wrong-sign action. In both halves the correct-sign
action must beat the wrong-sign action by at least `.25` in cosine to `N` for both `v(T)` and `v(T union G)`. The
wrong-sign control may be anti-aligned or otherwise different; no particular nonlinear response is assumed beyond
this registered separation.

The strong null is A false; B false; both `N` and `P` task-group norms below `.005 nat`; nonpositive `N`--`P` task
group cosine; all five-site effects numerically inert; or neither wrong-sign control is distinguishable from its
correct-sign action.

## Frozen interpretation and result routes

- A false: repair only the instrument. No subset result is evidence.
- A true/B false: the score-action calibration did not survive the new patch harness; repair or abandon the assay,
  not the thresholds.
- B true/C false: the rung-466 five-site program is code-specific. Abandon it as a cross-corpus unit and move to a
  natural-text action-conditioned state description; do not select replacement sites from these outcomes.
- B/C true/D false: the score computation is interchangeable at the attention interface, but its downstream
  realization depends on which head supplied it. Preserve the score gauge only; do not merge the downstream programs.
- B/C/D true/E false: the fixed program is stable, but this experiment has not identified gauge orientation as the
  cause. Register a new shifted-position or sign control before claiming one signed program.
- A--E true: identify one fixed downstream program across corpus and score implementations at the five-site boundary.
  The next rung may split MLP8/9/12 using downstream finite-effect equivalence across all four sources, followed by
  exact held-out term-group removal. This still is not an adopted compression: no internal MLP units are yet extracted
  and no deployed values are saved.

No result permits changing the site set, weakening bars, selecting a favorable subset, trying ranks, or calling a
whole attention head equivalent to another. The payload/output sides remain separate.

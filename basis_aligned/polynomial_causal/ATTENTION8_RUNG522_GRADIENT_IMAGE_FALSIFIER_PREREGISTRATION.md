# Conditional post-rung-522 gradient-image falsifier

Date: 2026-09-03  
Status: prospective and **not authorized to run unless rung 522 passes A--D**  
Model calls made by this document: zero

## Why this is not another rank experiment

Rung 522 asks whether one four-dimensional part of attention8 predicts the effects of omitted circuits, transfers
to a fourth circuit, and can be removed selectively. Even if all of that passes, it does not prove that later layers
literally read only those four directions. A finite set of donor swaps could agree for accidental reasons: the
tested donor differences may avoid important directions, or nonlinear curvature may make a locally broad computation
look narrow on those swaps.

This follow-up tests a stronger computational claim: on new documents, do the downstream derivatives that implement
the four circuit effects actually use the frozen projector? It targets stable identification, downstream-read
specification, OOD prediction, and selective manipulation. Fewer parameters or lower rank is not a success measure.

## Exact object

Let `Y` be the complete attention8 write for a batch, with shape `[rows, positions, 1152]`. For a circuit `c`, define
separate member and matched-control scalar losses

`S_c^M(Y) = mean CE at its member positions` and
`S_c^C(Y) = mean CE at its frozen matched-control positions`,

and let `S_c=S_c^M-S_c^C`.

Backward passes give `G_c^M=dS_c^M/dY`, `G_c^C=dS_c^C/dY`, and therefore

`G_c = dS_c/dY = G_c^M-G_c^C`, all shaped `[rows, positions, 1152]`.

For a natural donor swap let `Z=Y_donor-Y_recipient`. For the frozen rung-522 frame `Q` and projector `P=QQ^T`, the
complete first-order change and the part carried by the proposed circuit variable are

`b_c(Z) = <G_c,Z>` and `a_c(Z) = <G_c,ZP> = <G_c Q, Z Q>`.

The inner products cover every row, position, and channel affected by the batch loss; they do not assume that only
the same token position matters. Replacing `Q` by `QR` for any orthogonal `R` changes none of these values.

Two questions must be kept separate:

1. **Full gradient containment:** `||G_c Q||_F^2 / ||G_c||_F^2`. This asks whether all downstream sensitivity for
   the paired circuit loss lies in the projector. It is a deliberately strong statement.
2. **Natural-swap transport:** compare the vectors `a_c(Z)` and `b_c(Z)` across held-out donor maps and directions.
   This asks whether the projector carries the first-order effects that the natural swaps can actually excite. It
   may pass even when `G_c` also contains unrelated directions that the observed `Z` never uses.

The CPU implementation of these quantities is
`ops/attention8_rung522_gradient_image_falsifier_math.py`. Its planted checks require exact recovery for a true ridge
function, distinguish low full-gradient containment from exact natural-swap transport, verify projector-gauge
invariance, and expose a rank-four frame whose natural donors excite only one direction.

## Data boundary

- Freeze the exact rung-522 selected `Q`; no refitting or threshold changes.
- Use fresh documents absent from rung 522, split by document into discovery-free evaluation halves. Report natural
  text and code-like text separately if both contain enough examples of all four circuits.
- Rebuild circuit member/control pairs using the already-frozen matching rules. A circuit with too few fresh pairs
  is reported as underpowered; it is not pooled with another circuit to force a result.
- Use the same D0/D1 donor construction and forward/reverse directions, derived only inside each fresh document half.
- Compare the selected frame with the same 20 Haar frames and the 16 label-randomized frame families archived before
  rung-522 TEST. No new control is chosen after seeing gradients.

## Registered predictions

### A. The proposed variable carries held-out first-order circuit effects

For each of the four circuits and both fresh-document halves, across the eight natural donor maps and both
directions, `a_c` versus `b_c` must have signed cosine at least `.75`, best-scaled relative residual at most `.55`,
and positive aligned recovery. At least four of five document bootstraps must pass every cell. This is the same kind
of signed-effect requirement as rung 522, now applied to local downstream computation rather than finite CE changes.

### B. The result is specific to the learned circuit variable

For every circuit, compute projected tangent responses separately from `G_c^M` and `G_c^C`. The minimum over cells
of

`bounded selectivity(RMS(a_c^M), RMS(a_c^C)) * aligned recovery(a_c,b_c)`

must strictly exceed the maximum of the 20 frozen Haar frames and the higher 95th percentile of the 16 frozen
label-randomized families. The selected frame must also put a larger fraction of the paired circuit gradient inside
its projector than it puts inside the matched-control gradient, with a document bootstrap lower bound above zero.

### C. Natural swaps identify every claimed coordinate

Stack `ZQ` over all fresh rows, positions, maps, and directions in each document half. Let
`s1 >= ... >= s4` be its singular values after division by the square root of the number of observations. Require
`s4/s1 >= .05` in both halves. If this fails, rung 522 may still identify a lower-dimensional projector image, but
the data do not identify all four coordinates and no four-coordinate semantic account is allowed.

### D. Local and finite computations agree

Without additional fitting, compare `a_c(Z)` with the finite projected-swap effect and compare
`<G_c,Z(I-P)>=b_c(Z)-a_c(Z)` with the finite whole-swap effect minus the finite projected-swap effect. Both
comparisons must pass the rung-522 signed-cosine/residual bars, and the signs and circuit ordering must agree. If
projected first-order effects pass A but fail to predict the finite projected swap, curvature—not a simple linear
downstream reader—explains the finite intervention. If the orthogonal tangent term does not predict the difference
between whole and projected swaps, the proposed decomposition is not locally complete along the observed donors.

All four predictions must pass. Prediction A alone is only a local-response screen.

## Strong null and decisions

The strong null is that rung 522 found a projector useful for its particular finite donor census, but not a stable
downstream variable: gradient transport does not transfer to fresh documents, Haar or label-randomized frames do as
well, natural swaps excite fewer than four directions, or local effects do not predict finite effects.

- A--D pass: the frame gains a stronger downstream-reader interpretation; next infer a small named state/operation
  and test a commuting causal abstraction.
- A/B/D pass but C fails: retain only the naturally excited dimension and do not name the unexcited coordinates.
- A fails while rung 522 passed: describe rung 522 as a finite intervention equivalence class, not a ridge-like
  downstream variable.
- Rung 522 fails A--D: do not run this follow-up. Use its saved multiway responses for the already-proposed held-out
  response-factor screen, or collect more independent documents if the failure is a power limit.

## Literal price

The CPU summaries store one frozen projector, gradients or sufficient gradient contractions, `ZQ`, and scalar
responses. GPU cost is dominated by the fresh-document native forwards and one backward per circuit loss/batch,
plus finite fresh donor checks for D. The executable runner must state exact batch counts and a hard ceiling after a
managed no-outcome memory smoke; this preregistration deliberately does not guess that price before the gradient
instrument is implemented.

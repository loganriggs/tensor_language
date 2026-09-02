# Rung 491 preregistration — exact named-source decomposition of MLP1's live-state reader

## Question

Rung490 validated that `B(delta_b,z_N)` predicts the complete finite effect well for MLP0 branches T and I but not
C. This is an operational response law, not yet a semantic explanation: `z_N` is the whole native MLP1 state. The
next question is which actual residual-stream sources inside `z_N` make that reader work.

Immediately before MLP1, the unnormalized residual is an exact sum of the embedding/skip path, attention0, MLP0,
and attention1. MLP0 itself already has exact T/C/I/S branches. After applying the one scalar RMS-normalization gain
per token and retaining a numerical remainder, define nine sources:

1. `E`: embedding plus skip path;
2. `A0`: attention0 write;
3. `M0_OTHER`: MLP0's retained constant, bias, and numerical remainder;
4. `M0_T`;
5. `M0_C`;
6. `M0_I`;
7. `M0_S`;
8. `A1`: attention1 write; and
9. `NUMERICAL`: the exact residual needed after BF16 sequencing and normalization rounding.

They must sum exactly to the deployed normalized state `z_N`. Because `B` is linear in its second argument,

`B(delta_b,z_N) = sum_s B(delta_b,z_s)`.

This decomposition is fixed by named residual sources, not learned coordinates or rank.

## Physical arms

For each target branch `b` in T/C/I, use its branch-absent MLP0 direct write and naturally recomputed branch-absent
attention1 write as the fixed background at MLP1. Construct:

- `OWN_b`, the exact full midpoint secant;
- `FULL_b=B(delta_b,z_N)`, the validated native-state term;
- nine `SINGLE_b,s=B(delta_b,z_s)` writes; and
- nine `LEAVE_b,s=FULL_b-SINGLE_b,s` writes.

Inject each write into the branch-absent MLP1 output and let layers2--17 recompute. Per-token benefit is
`CE(branch absent)-CE(arm)`. For every singleton source, compare its MLP1 write with sixteen controls obtained by
shifting only that source state to another token position. C's results are descriptive: rung490 established that
FULL_C does not meet the complete-effect prediction bar, so no source inside an insufficient parent may be selected
as an explanation of C.

## Data split

Use the same hash-bound 1,000 documents. Discovery is0:500, split0:250/250:500. Validation500:1000 opens only after
the source set is frozen. These documents have earlier intervention outcomes but no source-decomposed MLP1 arms; the
claim is held out by intervention outcome, not globally virgin or new-corpus OOD. Final and sealed roles stay closed.

## Frozen predictions

### A — exact and lawful source instrument

- All rung490/model/row/source/result/preregistration hashes match and rung490 has A--D true.
- The nine normalized state sources sum to native `z_N` at relative squared error at most`1e-12`; NUMERICAL is
  explicitly retained rather than discarded.
- For every branch, the nine bilinear source terms sum to FULL at float32 relative squared error at most`1e-8`.
  `FULL+CURVATURE=OWN` retains the BF16 `8u^2` and OWN-write `4u^2` bounds.
- Prefixes, calls, and injections are exact; every non-NUMERICAL physical arm is live. The explicitly retained
  numerical remainder may vanish after deployed BF16 injection and is measured rather than required to be live.
  Validation remains closed before selection.

### B — validated parent response remains live

In both discovery halves, FULL predicts OWN for T and I at cosine at least`.90` and adjusted error at most`.45`,
with same-position FULL write beating shifted-native-state controls by`.15`. C is reported against the same bars and
is not required to pass.

### C — at least one named source jointly modulates T and I

For a named non-NUMERICAL source `s` to be necessary for target `b`, in each discovery half:

- leaving `s` out of FULL must reduce FULL-versus-OWN effect cosine by at least`.03` **and** increase adjusted error
  by at least`.05`;
- the singleton physical effect RMS must be at least10% of FULL's effect RMS; and
- the same-position singleton write cosine with FULL must beat the95th percentile of its16 shifted-source controls
  by at least`.15`.

At least one source must meet every clause for both T and I. This identifies a shared named modulator without assuming
that it is sufficient alone. Report singleton sufficiency (`.80/.50`) separately but do not require it.

### D — stable source set and numerical control

Compute necessary source sets separately in documents0:250 and250:500. Their T/I intersection must be identical and
nonempty in both halves. `NUMERICAL` must never meet the necessary-source clauses, and its MLP1-write RMS must remain
below2% of FULL for T and I. Freeze the complete shared set; do not keep only the best source.

### E — held-out intervention outcomes

Open documents500:1000 only if A--D hold. In both validation quarters, every frozen shared source must again meet all
necessity, liveness, and shifted-position clauses for both T and I. No new source may be added; NUMERICAL must again
stay below2% and unselected.

The strong null fires if A, B, C, or D fails. If E fails, the discovery source attribution remains a screen. If no
shared source is found, retain FULL as an unsplit live-state reader and move to the two-dimensional integrated
NATIVE/CURVATURE response for each branch. A validated shared source licenses a selective source-removal/composition
test, not immediate compression.

## Interpretability and price

This advances computational specification, grouping, held-out causal prediction, and stable identification by using
named residual sources. It does not choose a low-rank coordinate system or optimize storage.

Per phase, run125 native,375 branch-absent, and7,500 physical full-model forwards at batch size4 (`60` physical arms
per batch), totaling8,000. Validation repeats only if licensed. Store contracted effects, source/write statistics,
hashes, and audits. Add and remove zero deployed parameters.

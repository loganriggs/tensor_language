# Rung 489 preregistration — native-state dominance versus T/I-specific MLP1 midpoint reader

## Question

Rung488 validated exactly one bidirectional midpoint-interchange graph edge: T--I. Its graph definition required both
directions of a branch pair to pass. A within-target audit exposes a confound: for target T, C's midpoint is slightly
better than I's, and for target I, C's midpoint is nearly as good as T's. T--C and C--I fail as graph edges because
the reverse C-target directions miss, not because their midpoint donors are uniformly bad.

The exact algebra supplies a stronger alternative. For branch `b`, let native and branch-absent MLP1 inputs be `z_N`
and `z_b`, and define `delta_b=z_N-z_b`. With MLP1's symmetric bilinear form `B`,

`OWN_b = B(delta_b,(z_N+z_b)/2)`

`       = B(delta_b,z_N) - 0.5 B(delta_b,delta_b)`.

Call the first term `NATIVE_b`: the native-state linear reader applied to branch change. Call the second `CURVATURE_b`:
the exact self-quadratic correction. Every branch midpoint shares the large `z_N` term. Rung489 asks whether that
common term explains the apparent interchange, or whether T and I's midpoint corrections are specifically better
than C and native-state controls.

## Arms and physical computation

Use rung488's hash-bound 1,000 documents and the same exact T/C/I removals. For each target branch `b`, hold its MLP0
direct write and recomputed attention1 write at the branch-absent trajectory. At MLP1 inject the branch-absent write
plus each of five exact candidate writes:

1. `NATIVE = B(delta_b,z_N)`;
2. `MID_T = B(delta_b,(z_N+z_T)/2)`;
3. `MID_C = B(delta_b,(z_N+z_C)/2)`;
4. `MID_I = B(delta_b,(z_N+z_I)/2)`; and
5. `CURVATURE = -0.5 B(delta_b,delta_b)`.

`MID_b` is the exact OWN secant. Let layers2--17 recompute physically. The absent trajectory is the empty baseline.
Report per-token CE benefit `CE(absent)-CE(arm)`, exact MLP1-write cosine, and sixteen controls that cyclically shift
only the candidate's state factor to another position. Also report the two-part downstream decomposition:

`own effect = native-state effect + curvature effect + their nonlinear CE interaction`.

That last equation is an exact two-arm inclusion--exclusion accounting of the physical outcomes; it does not assume
that CE effects add.

## Data and scope

Discovery uses documents0:500, split0:250/250:500. Documents500:1000 have been used for rung488's different arms,
but the NATIVE and CURVATURE physical outcomes have never been computed. Treat them as intervention-outcome held out,
not globally untouched data. Open those outcomes only after discovery classifies the mechanism. Split validation into
500:750/750:1000. Final and sealed roles remain unopened.

## Frozen predictions and classification

### A — exact instrument

- All rung488, model, row, source, result, and preregistration hashes match; rung488 has A--E true and exactly the
  T--I context edge.
- Native and branch-absent prefixes replay exactly; call and injection counts are exact.
- In float32, `NATIVE_b+CURVATURE_b=OWN_b` has relative squared error at most `1e-8` for every branch. The deployed
  BF16 sum is at most `8u^2=1.220703125e-4` from the native-minus-absent MLP1 write, and OWN injection is at most
  `4u^2=6.103515625e-5` from the native MLP1 write, with `u=2^-8`.
- Every physical arm is live. Validation stays closed before the discovery decision.

### B — common native-state reader

For T, C, and I separately, in each discovery half, NATIVE predicts OWN's complete per-token physical effect with
cosine at least`.90` and best scalar-adjusted relative error at most`.45`. Its same-position MLP1 write cosine must
beat the95th percentile of its16 shifted-native-state controls by at least`.15`.

### C — T/I-specific midpoint donor

For target T, MID_I must meet the original `.80/.50` effect bars and beat both MID_C and NATIVE by at least`.03`
cosine and `.05` lower adjusted error. For target I, MID_T must meet the same clauses against MID_C and NATIVE. Every
clause must hold in both discovery halves. This is deliberately target-conditional: reverse-direction behavior on C
cannot create specificity for a T or I target.

B and C describe competing explanations. The frozen expected classification is `COMMON_NATIVE`: B true and C false.
If C is true, classify `TI_SPECIFIC`; if neither is true, classify `NEITHER`. Report CURVATURE and the nonlinear CE
interaction for all branches without selecting or thresholding them.

### D — stable discovery classification

Compute B and C separately in documents0:250 and250:500. The same non-NEITHER class must occur in both halves. No
threshold, donor, branch, or class may be changed after seeing either half.

### E — intervention-outcome validation

Open the new physical outcomes on documents500:1000 only if A and D hold. Freeze the discovery class. Each validation
quarter must independently reproduce every clause defining that class. No alternative class may be substituted.

The scientific strong null fires if A, D, or E fails. A COMMON_NATIVE result replaces the T/I-specific interpretation
with a three-branch state-dependent linear reader plus measured self-curvature. A TI_SPECIFIC result preserves the
T/I grouping and licenses extraction. NEITHER routes to separate within-branch integrated finite-response readers.

## Interpretability and price

This tests computational specification, cross-branch grouping, stable identification, held-out causal prediction,
and exact composition. It does not select a rank or claim compression.

Per phase: 125 native,375 absent, and1,875 physical full-model forwards at batch size4, totaling2,375 discovery
forwards and the same conditionally for validation. Store only contracted effects, write-cosine sums, audits, and
hashes. Add and remove zero deployed parameters.

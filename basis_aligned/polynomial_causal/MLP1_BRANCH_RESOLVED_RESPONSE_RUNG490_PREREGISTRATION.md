# Rung 490 preregistration — held-out branch-resolved MLP1 finite-response signature

## Why this follows the rung489 null

Rung489 correctly classified neither of its two proposed global mechanisms. T/I-specific midpoint donors failed, and
one native-state reader for all T/C/I failed because C missed the `.90/.45` effect bars. Its pre-registered per-branch
reports nevertheless expose a sharper split in both discovery halves:

- T native-state prediction: cosine `.972/.970`, adjusted error `.234/.242`;
- I native-state prediction: `.981/.982`, `.194/.191`;
- C native-state prediction: `.871/.870`, `.492/.493`.

The exact physical NATIVE/CURVATURE decomposition also has the same branch ordering in both halves. Relative to each
branch's OWN response, curvature-only RMS is T `2.71/2.70`, I `.92/.89`, C `.68/.69`; the downstream nonlinear
interaction RMS is T `2.74/2.74`, I `.94/.91`, C `.75/.76`.

This suggests a branch-resolved law rather than a global reader: T and I share a strong native-state response
direction, while C requires more of its full finite state, and the nonlinear corrections remain quantitatively
different across all three. This is a discovery screen only. Rung489 kept the corresponding physical outcomes on
documents500:1000 closed, so rung490 freezes the split before opening them.

## Computation

Reuse rung489's exact arms and code without changing their definitions. For branch `b`,

`OWN_b = B(delta_b,z_N) - 0.5 B(delta_b,delta_b)`.

`NATIVE_b=B(delta_b,z_N)` is physically injected into the branch-absent MLP1 trajectory. `CURVATURE_b` is injected
separately. Let layers2--17 recompute. Per-token effects are `CE(absent)-CE(arm)`. The nonlinear interaction is the
exact outcome residual

`OWN effect - NATIVE effect - CURVATURE effect`.

No response is fitted. The validation calculation runs the same five physical modes for instrument parity, but the
scientific tests below use only the frozen NATIVE, each branch's exact OWN, CURVATURE, and their physical interaction.

## Data scope

Load the hash-frozen rung489 result and require A true, B/C/D/E false, both halves classified NEITHER, and validation
closed. Run documents500:1000 once, split into quarters500:750 and750:1000. These documents are not globally unseen
and rung488 computed different midpoint arms on them, but rung490's NATIVE and CURVATURE physical outcomes have never
been computed. The claim is therefore prospective intervention-outcome validation, not new-corpus OOD evidence.
Final and sealed roles stay closed.

## Frozen predictions

### A — exact instrument

All model, row, parent-source, parent-result, and preregistration hashes match. Calls, prefixes, and injections are
exact. `NATIVE+CURVATURE=OWN` passes the same float32 `1e-8`, BF16 `8u^2`, and OWN-write `4u^2` bounds independently
on validation. Every physical arm is live.

### B — T/I pass while C fails

In each validation quarter, T and I NATIVE effects each predict their OWN effects with cosine at least`.90` and best
scalar-adjusted relative error at most`.45`. In the same quarter, C must fail at least one of those two clauses. All
three same-position writes must beat their16 shifted-native-state controls by at least`.15` cosine.

### C — branch contrast is material

In each quarter, the smaller T/I effect cosine minus C's cosine is at least`.05`, and C's adjusted error minus the
larger T/I error is at least`.10`. This prevents a result driven by C barely crossing an arbitrary threshold.

### D — finite-correction ordering

In each quarter, both the CURVATURE/OWN RMS ratio and nonlinear-interaction/OWN RMS ratio must have strict ordering

`T > I > C`.

Report all ratios and signed means. This ordering is a branch signature, not an additive decomposition: the
interaction is explicitly retained.

All A--D must hold. Otherwise the strong null fires and the discovery screen is not promoted. A pass identifies a
held-out operational split at MLP1: T/I share a native-state response approximation that does not cover C, while the
three finite corrections remain distinct. It does not restore the T/I-specific midpoint-donor claim and does not yet
extract a complete circuit. The next step in either case is a branch-wise integrated response model, with a pass
allowing T/I to share only the native-state term and requiring separate finite corrections.

## Relevance and price

This advances cross-branch grouping, within-computation splitting, held-out causal prediction, and stable
identification. It does not optimize rank or compression.

Run 2,375 full-model forwards at batch size4 on validation only. Store contracted effects, response ratios, write
controls, hashes, and call audits. Add and remove zero deployed parameters.

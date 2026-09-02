# Rung 468: frozen code-selected MLP product terms on natural text

Status: prospective cross-register confirmation, frozen after rung 467 and before applying any selected MLP product
term to the natural-text role. Rung 467 selected 1,358 exact terms using code documents `0:96` and tested them on
code documents `96:192`. This rung does no fitting or selection: it transfers those exact term indices and their
matched-count controls to the already-frozen, repository-disjoint 192-document `final_natural` role.

The natural role has been opened by earlier equality experiments, so this is not a globally untouched final role.
However, no MLP8/9/12 product-term intervention has been run on it. It is an outcome-unopened distribution-shift test
of the fixed code-derived split.

## Frozen objects and computation

Reuse without change:

- native matcher `N`, transplanted matcher `H`, and the natural-fit matcher scale;
- proposed product-term sets: MLP8=450, MLP9=426, MLP12=482 exact indices from rung 467;
- amplitude and SHA256-random matched-count indices from rung 467;
- context order `(near, far, one predecessor, multiple predecessors)`;
- the exact intervention `z_live[j] <- z_absent[j]` at selected terms, with all later layers recomputing.

On all 192 natural documents and fixed reporting halves `0:96` and `96:192`, run the full source trajectories, all
eight subsets of the three proposed groups, all eight subsets of the three complete MLP writes, and each individual
plus union arm for both matched controls, under both sources. No gradient, threshold, index, scale, role, or module
may be refit or reselected.

For any arm, its four-context causal vector is the complete source effect minus the intervened source effect. Parent
vectors are the causal effects of replacing every product activation in the same MLP set by its absent value. This
is the same operational definition as rung 467.

## Registered predictions

### A. Exact instrument

All rung-467 result/source, model, natural-row, receipt, selected-count, selected-index, control-index, and source-
scale hashes hold. Native replay relative squared error is at most `1e-12`; attention-factor reconstruction error is
at most `1e-10`; empty groups are exact no-ops; every registered arm executes once per batch; all term baselines have
shape `[batch,256,4608]`; and SEALED remains closed.

### B. The code-selected union transfers to natural text

For both sources, the proposed union has the `(-,+,+,-)` sign pattern, cosine at least `.75` with the natural complete-
MLP parent vector, projection magnitude in `[.15,1.50]`, context-vector norm at least `.01 nat`, and norm at least
twice its absolute off-target effect. Native/hybrid proposed vectors have cosine at least `.75`. Parent alignment and
source agreement are positive in both 96-document halves.

### C. It still beats matched-count controls

For both sources, the proposed union's parent cosine is at least `.10` above both amplitude and random controls, or
its positive parent projection is at least twice both controls while its cosine is at least `.70`. The selected group
must beat both controls in the corresponding metric in each natural-text half.

### D. The split remains cross-module

At least two individual proposed MLP groups have cosine at least `.50` with their corresponding complete-module
parent vector under both sources, native/hybrid group cosine at least `.60`, and positive parent alignment in both
natural-text halves.

### E. The interactive composition law transfers

For both sources, the proposed union minus the sum of its three singleton causal vectors has norm at least `.005 nat`.
The native/hybrid natural interaction vectors have cosine at least `.70`. For each source, the natural interaction
has cosine at least `.50` with the corresponding frozen code interaction vector from rung 467. This asks whether the
same non-additive cross-MLP correction, not only the same marginal direction, survives the register shift.

The strong null is an invalid instrument; proposed-union norm below `.005 nat` for both sources; native/hybrid union
cosine at most zero; failure to beat both controls under both sources; or no individual MLP group with positive parent
alignment under both sources.

## Decision and price

- A--E pass: identify the fixed product-term groups as a natural/code, source-invariant, cross-module component. The
  next step is to explain or compile what the 1,358 terms read, not to tune their count.
- B--D pass but E fails: the marginal component transfers but its interaction law is register-specific; preserve an
  explicit background/register interface.
- B or C fails: retain the held-out code split only and stop term-threshold tuning. Move to the full class-projected
  bilinear form or a state-level causal quotient.

This confirmation saves and adds zero deployed parameters. Only causal transfer, control separation, source
interchange, and interaction transfer count; term count is a literal description cost, not evidence by itself.


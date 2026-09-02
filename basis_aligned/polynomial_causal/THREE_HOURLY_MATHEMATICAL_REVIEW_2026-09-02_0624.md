# Three-hour mathematical review — 2026-09-02 06:24 UTC

## Goal and current boundary

The goal is an executable decomposition whose parts generalize to held-out and shifted inputs, can be extracted or
selectively removed without damaging unrelated computations, and are reusable/compositional where the model itself
reuses them. Fewer parameters or lower rank is not sufficient. It is useful only when it helps recover parts with
those causal and predictive properties.

The current equality circuit has three exact query-position interventions in MLP8, MLP9, and MLP12. Their individual
effects survive two reasonable intervention coordinates. Pair/triple interaction allocations do not, and complete
MLP writes do not form stable groups in the existing 62-circuit downstream basis. Rung476 is now testing whether the
narrower, frozen product-coordinate groups from rung467 supply the missing downstream grouping.

## Exact tensor being tested

For MLP `m`, product coordinate `j`, token state `x`, and output coordinate `d`, the MLP write is

`write_m(x)[d] = sum_j Down_m[d,j] * (Left_m(x)[j] * Right_m(x)[j]) + bias_m[d]`.

Thus each product coordinate is an exact multiplicative feature followed by a fixed 1,152-number write direction.
For circuit `c`, source `s`, and a set of coordinates `G`, rung476 measures

`R[m,G,s,c] = mean CE_change after replacing G at equality-positive query positions in circuit c`.

The result is a response tensor indexed by MLP, proposed group/control, matcher source, circuit, and document half.
Two stored pieces count as the same downstream variable only if their response vectors agree across the independent
source and document views and survive exact intervention. This is a quotient by downstream distinguishability: two
writes are treated as equivalent only when later computation cannot reliably tell their causal effects apart on the
registered tests.

This quotient is better aligned with the research goal than Euclidean distance between weights or a small matrix
rank. It can directly group pieces stored in different native MLPs when later circuits use them in the same way.

## Why rung476 can fail even if a useful decomposition exists

Rung467 selected coordinates using first-order effects on four code equality contexts. Its strong held-out code result
establishes a real code-specific component, but rung468 already showed that its magnitude and matched-control advantage
did not transfer to natural text. Rung476 does not refit that selection. A null would therefore reject this particular
old coordinate group as a general downstream grouping; it would not imply that MLP8/9/12 have no finer circuit parts.

There is also a mathematical mismatch between the selection and the new target. Rung467 compresses a four-context
response, whereas the new object has 62 behavioral coordinates and two source views. A group can be optimal for the
four-context direction while mixing several distinct columns of the larger downstream-response tensor.

## Higher-information successor if the frozen split fails

Construct a product-coordinate × downstream-circuit response tensor directly. At first order, an individual product
coordinate has response

`F[m,j,s,c] = mean over circuit-c positions of -<downstream CE gradient, Down_m[:,j] * equality product change_j>`.

This keeps together the feature's input-dependent activation change, its exact write direction, and how later layers
read that direction. It is not a weight-only SAE. Use disjoint top-level circuit families as discovery and validation
views:

1. Compute `F` on discovery circuit families, both sources, and two document halves.
2. Find sparse groups of coordinates with similar signed response profiles across these independent views. Compare a
   sparse shared-factor model, a parts-based nonnegative/signed model, and an archetypal model whose atoms are anchored
   to observed response columns.
3. Freeze the groups and their names before looking at validation circuit families.
4. Test prediction on held-out circuit families and shifted data.
5. Replace each complete proposed group in the live model. Require selective circuit removal, preservation of unrelated
   circuits, and physical interchange when two MLP groups are claimed equivalent.

First-order responses are only a proposal mechanism. Exact interventions decide whether a group is a circuit and
whether interactions make the linear grouping invalid.

## Identifiability conditions worth exploiting

This is a multi-view latent-factor problem, not an arbitrary one-matrix decomposition. The same latent computation is
observed through matcher source, document half, circuit family, and eventually data register. Agreement across those
views can remove many accidental rotations. Stronger anchors can make the remaining ambiguity explicit:

- **separability or anchor circuits:** each latent computation has at least one downstream circuit where it contributes
  without the other proposed computations;
- **independent interventions:** selected removals or swaps change one latent part while leaving the others fixed;
- **shared factors across views:** the grouping is fixed while response strengths may change by source or register;
- **sparsity in circuit use:** each downstream circuit reads only a small subset of latent parts; and
- **simple exact input rule:** a grouped product feature has a compact token/context rule that predicts its activation.

An archetypal/convex-hull constraint is principled only if the scientific hypothesis is that atoms should correspond
to observed extreme response profiles. It prevents arbitrary atoms outside the data cloud and can improve stability,
but it does not by itself make the atoms causal or semantic. It should be retained only if it improves held-out circuit
prediction, exact selective removal, and reuse relative to unconstrained sparse alternatives.

## Simplicity objective without rank drift

The selection objective should be a held-out scorecard, not one scalar compression proxy. Candidate decompositions
should be compared on:

- prediction of unseen examples and unseen circuit families;
- selective removal of the named circuit with small effect on unrelated circuits;
- extraction or interchange of a proposed shared component;
- reuse of one component in multiple computations;
- stability across sources, halves, and appropriate register shifts; and
- description length only after charging for group memberships, activation rules, exceptions, and interactions.

A low-rank response tensor can be useful as a proposal because it shares statistical strength across views. It is not
the result: if the resulting factors cannot be selectively manipulated or fail held-out circuit families, the low rank
has not found the desired computation.

## Decision

Let rung476 finish unchanged. If the frozen product groups pass, proceed to circuit-family-heldout physical
interchange. If they fail, do not tune their term count or reduce rank. Build the multi-view per-product downstream
response tensor on discovery circuit families, compare sparse/archetypal grouping hypotheses, and let held-out circuit
prediction plus exact selective interventions choose among them. This directly connects tensor decomposition to the
three core interpretability tests rather than treating decomposition simplicity as the endpoint.

# Equality pre-MLP9 three-edge factorial discovery V1

## Question

The post-MLP9 boundary factorial localized essentially the complete equality
score-removal effect to the pre-MLP9 residual.  On the same opened documents
564--627, factor that residual into the three native module-write edges between
the established L8H4 score edit and MLP9:

1. `A8`: the complete attention8 write, including the directly edited L8H4
   equality term;
2. `M8`: the complete MLP8 write induced by the post-attention8 state;
3. `A9`: the complete attention9 write induced by the post-MLP8 state.

Capture native and recipient-absent values for these three writes.  Under the
absent trajectory, run all eight fixed native/absent combinations.  A bit value
of one installs that edge's native write and zero installs its absent write;
all other computation is live.  Thus `000` and `111` must reproduce the absent
and native trajectories exactly.  This is a causal oracle-edge factorial: no
component is fitted, projected, rescaled, or selected before the outcomes.

## Selection rule

Among the six non-corner supports of size one or two, a sparse support is
eligible only if it:

- recovers at least `.90` of the copy-positive NLL effect;
- changes all-noncopy mean NLL by at most `.01` nat from native;
- has recovery above `.70` in both document halves and in the near, far,
  one-predecessor, and multiple-predecessor cells.

If multiple supports qualify, select the smallest, then the one whose aggregate
copy recovery is closest to one, then lexical arm name.  This selected support
is only an opened-panel candidate for a later frozen fresh/OOD test.

## Predictions

- **A — instrument:** custom native/absent logits and MLP9 writes reproduce the
  authoritative forwards; all installed edge writes and both corner pre-MLP9
  states have relative L2 error at most `2e-6`.
- **B — corner sufficiency:** `000` has zero recovery within `.01`; `111`
  recovers `[.99, 1.01]` and changes all-noncopy mean NLL by at most `.001` nat.
- **C — sparse support:** at least one registered support of size at most two
  passes the fixed selection rule.
- **D — single-edge concentration:** at least one singleton recovers `.50` or
  more of the copy-positive effect.
- **E — low-order graph:** every baseline-anchored pair interaction has
  absolute recovery at most `.25`, and the third-order interaction has
  absolute recovery at most `.15`.
- **F — stability:** the selected sparse support passes every registered half
  and cell threshold without reselection.

A/B are required for interpretation.  Passing C/F yields a sparse oracle-edge
candidate.  D/E diagnose whether the graph is concentrated and approximately
low-order; their failure does not invalidate a mechanically sound factorial.
No result here is an extracted component, fresh prediction, or complete
circuit.

## Price

One checkpoint load; 64 opened documents; two authoritative forwards, two
custom capture forwards, and all eight independently constructed factorial
forwards per four-document batch: exactly 192 forwards.  No new text, fits, gradients, or
parameter updates.

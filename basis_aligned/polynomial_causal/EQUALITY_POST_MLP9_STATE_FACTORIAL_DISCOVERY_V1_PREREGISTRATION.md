# Equality post-MLP9 state factorial discovery V1

## Question

The complete native MLP9 write restores only `.0393` of the equality-score
removal effect.  Decompose the complete post-MLP9 state change into:

`delta_post = delta_pre_residual + delta_mlp9_write`.

On opened documents 564--627, execute the established L5H5 score removal and
freeze native/absent pre-MLP9 residuals and MLP9 writes per row.  Under the
absent trajectory install four post-MLP9 boundary states:

- **pre-fixed:** native pre-MLP residual + absent MLP9 write;
- **write-only:** absent pre-MLP residual + native MLP9 write;
- **both-fixed:** native pre-MLP residual + native MLP9 write;
- **pre-recompute:** native pre-MLP residual + MLP9 recomputed from that state.

No state is fitted or rescaled.  `both-fixed` and `pre-recompute` are independent
construction routes to the native post-MLP9 boundary and must agree.

## Predictions

- **A — instrument:** custom native and absent trajectories reproduce the
  authoritative forwards exactly; component closure and installed boundary
  tensors are within `2e-6` relative L2.
- **B — boundary sufficiency:** both-fixed and pre-recompute each recover at
  least `.95` of the copy-token NLL effect, differ by at most `.01` recovery,
  and change all-noncopy NLL by at most `.01` nat from native.
- **C — write-only null repeats:** write-only recovery is at most `.10`.
- **D — upstream residual dominance:** pre-fixed recovery is at least `.70` and
  exceeds write-only recovery by `.50`.
- **E — near-additive causal graph:** absolute interaction in recovery,
  `both - pre_fixed - write_only`, is at most `.20`.
- **F — cell stability:** pre-fixed and both-fixed recovery are positive in
  both document halves and on near, far, single-predecessor, and
  multiple-predecessor copy cells.

Passing identifies the pre-MLP9 residual as the dominant state edge and MLP9's
write as a small correction.  It is a boundary factorial with oracle native
states, not an extracted upstream program or fresh/OOD circuit.

## Price

One checkpoint load; 64 opened documents; authoritative native/absent forwards
plus six custom trajectories per four-document batch; no new text, fitting,
gradients, or parameter updates.

# Early-MLP suffix transport v1: live-consumer norm amendment

Status: prospective source contract.  This amendment records no final-row,
checkpoint-derived, or outcome value and does not authorize opening the final role.

## Physical observable

The consumers are the outputs of `attn.c_proj` in model layers 0 through 17,
inclusive.  The hook is a forward-output hook, after the projection's native bias;
it is not a hook on the projection input or on the later attention return.  This is
the same physical write measured by the earlier `layer_norms` instrument.

For each four-row final batch, action, layer, and row, let `write_action[p]` be that
output at sequence position `p`.  Let `write_native_same_background[p]` be the
corresponding output of the native baseline on exactly the same row and background:
`O/O/N` for an N-background action and `O/O/E` for an E-background action.  On the
common scored support `p = 64:256`, compute

```
numerator   = mean_p ||write_action[p]||_2
denominator = mean_p ||write_native_same_background[p]||_2
ratio       = numerator / denominator.
```

The implementation converts each captured output to float32 before the norm and
mean, matching the earlier instrument, then publishes the ratio as a float64 scalar.
It emits one `RowReduction` per layer after the 48 canonical batches are joined:
`row_sum[row] = ratio` and `row_count[row] = 1`.  A denominator at or below `1e-12`
is an integrity failure.  The native baseline ratio must equal one within absolute
and relative tolerance `2e-6`.

## Identity and lifecycle

Every action capture and its native denominator capture must share the exact batch,
row support, background, model authority, and ordered layer-0-through-layer-17
component identity.  The action and model identities are checked both before and
after the observed forward.  Every output hook must fire exactly once, must be
removed in a `finally` path, and must be inert after removal.  Missing, duplicate,
reordered, malformed, nonfinite, or identity-drifting captures poison the
transaction.

Only four row scalars per layer are retained by the capture boundary.  Raw consumer
writes, model/module handles, role rows, residual states, and CUDA tensors cannot
cross it.  Captured action magnitudes are private, one-use inputs to the paired
reducer.  A captured O/O magnitude is instead an immutable, hash-revalidated
denominator that may be reused by all actions on that exact background, batch,
model, and support; this permits 96 denominator forwards rather than one redundant
native forward per action.  Only typed CPU reductions and tensor-free receipts may
escape.

These ratios are integrity diagnostics.  They cannot select or fit a program,
amplitude, grammar, route, action, or scientific claim.  This amendment adds no
final-role opening authority and does not modify the frozen preregistration.

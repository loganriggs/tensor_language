# Full-value even-key interaction, 14 September

The existing exact key-coordinate rewrite now supplies all 128 value channels of
head9.8. For its fixed projection, let G be the product of both native normalized
QK score arrays, and G_ref the product with reflected key numerators and unchanged
native denominators. The extracted component uses the causal gate

\[
G_{even}=(G+G_{ref})/2,\qquad
Y=G_{even}[(1-\lambda)XV_9^T+\lambda X_0V_0^T]O_9^T.
\]

X and X0 are supplied normalized current and initial inputs. Both QK factors,
rounded rotary constants, native signed mixture, and full residual output map
are retained. This is a conditional interaction decomposition; reflection is not
a new native forward pass under a key edit. The chosen key projection is fixed
from the earlier scalar circuit. Extending its value channels does not identify
128 new circuits or establish new semantic roles.

[Four-context replay](EVEN_KEY_VALUE_BANK_V1_RESULT.json) agrees with independent
old scalar executions within 5.2e-16. The program stores 1,048,577 scalars in
4,263,970 serialized bytes, including QK, key coordinates, both value maps, output
map and mixture. Native input generation and downstream consumers remain external.

**Timing correction:** the first receipt's `pred_b` used 256 separate current and
inherited scalar evaluations, so it does not satisfy the registered 128-call
comparison. Its 50.1x figure is not the accepted benchmark. The
[matched correction](EVEN_KEY_VALUE_BANK_MATCHED_V1_RESULT.json) mixes values
first and evaluates exactly 128 scalar channels: 73.38 ms versus 2.81 ms, or
26.11x, on the first fixed 18-token context. Both arms include value projections,
mixture and output map. This measures reuse versus repeated scalar evaluation,
not speedup over already-vectorized native attention or the full model.

The separate [standalone response extraction](COMPLETE_PORTS_EXTRACTION_V2_RESULT.json)
passes 432 cases with no repository imports. It retains its existing rank-64
approximation and supplied pristine states. The
[whole-tensor](INTERACTION_PACKAGE_SHARED_BANK_V1_RESULT.json) and
[row-sharing](INTERACTION_PACKAGE_ROW_SHARING_V1_RESULT.json) screens across five
distinct conditional programs both fail the 1% saving gate. Stop literal
deduplication for that portfolio; those nulls do not rule out algebraic sharing.

[Native-module recomposition](EVEN_KEY_NATIVE_RECOMPOSITION_V1_RESULT.json) now
passes on four cached contexts: even plus independently computed dense-basis odd
matches the actual FP32 attention module within 3.16e-7 relative error, and direct
FP64 algebra within 8.96e-16. Omitting odd causes 10.37–15.27% full-head output
error; the even component is not a faithful whole-head replacement. Both branches
retain the native signed current/first value mixture and output map.

Next: test selective native removal and consumer reuse. Current results establish
conditional algebraic extraction and computation reuse, not downstream behavior.

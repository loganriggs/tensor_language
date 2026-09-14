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

## Native selective-removal result

[The registered full-value test](EVEN_KEY_FULL_VALUE_NATIVE_V1_RESULT.json)
completed 768 body forwards in 9.82 seconds on the reused 72 regional and 32
natural prefixes. Instrument and historical replay A/D pass: live even+odd versus
native head write <=3.99e-7, self-donor exactly zero, native capability present.
Regional selectivity B and unrelated preservation C both fail.

| Regional family | Full-even removal coverage | Earlier scalar coverage | Even donor transfer |
|---|---:|---:|---:|
| 0 | 43.02% | 42.28% | 47.46% |
| 1 | 41.84% | 41.96% | 46.75% |
| 2 | 44.51% | 43.42% | 47.48% |

All removal pairs and donor rows have the directed sign, but all families miss
the frozen 50% removal/transfer bars. Family0 also fails the unrelated/target
removal ratio: 0.527 versus <=0.5. Mean absolute newline CE changes are
0.03448/0.02522 versus <=0.02; the first half's maximum 0.11365 also misses <=0.1.
The earlier scalar's unrelated-readout changes are much smaller while its cue
coverage is similar. This does not promote the expanded component as a selective
circuit or refute the earlier scalar result.

The strongest explanation is that added value directions carry other computations.
The [physical remainder test](EVEN_KEY_REMAINDER_NATIVE_V1_RESULT.json) removes
full-even minus the known scalar, rather than merely subtracting measured effects.
Its 208 forwards finish in 3.33 seconds with bit-identical native anchors.
Paired cue-change norm is only 4.89/3.91/6.36% of the scalar's; mean absolute
unrelated-readout changes are 6.26/4.23/7.09 times the scalar's. Both directional
predictions pass. This localizes the cue-dependent part to the old scalar while
its complement carries substantial other effects. The complement still changes
individual cue-token logits; small paired cue changes do not mean zero effect.

Separate scalar and remainder effects predict full-even removal within 1.54–4.48%
on both regional readouts in all three families. However, overall composition C
fails: the first natural half's newline CE error is 5.458%, above 5%, while the
second half is 1.733%. Remainder mean absolute newline CE is 2.12/5.76 times
the scalar's. These are conditional development-panel results, not a newly
named semantic circuit, universal additivity or OOD identification.

The [CE-boundary control](EVEN_REMAINDER_CE_BOUNDARY_V1_RESULT.json) completed 128
forwards in 2.85 seconds, with all four native CE anchors bit-identical. The
analytic decomposition is exact, but the hypothesis that metric curvature mostly
explains the failure is rejected. Raw-score/final-RMS interaction has L2 norm
0.726/1.045 times the total CE interaction; softcap curvature 0.026/0.054 and
logsumexp curvature 0.355/0.359. These signed terms can cancel; they are not
nonnegative shares. Native downstream nonlinearity remains a material limit.

## One shared graph for the three components

The [shared graph](EVEN_VALUE_SHARED_GRAPH_V1_RESULT.json) expresses the old
scalar reader in the native current-value row space and its writer in the native
output column space. Both weight-derived spans replay below 9.72e-16. The scalar
branch needs two 128-coordinate vectors instead of two 1152-coordinate vectors.
S, R=even−S and O share routing, value reads and the output map. Their requested
writes combine in 128-dimensional head space before the unchanged nonlinear
suffix; no additive-logit approximation is assumed.

All 36 tested binary/signed gain cases pass <=1e-10 write replay on four old
contexts. The [storage audit](EVEN_VALUE_SHARED_GRAPH_PACKED_V1_RESULT.json)
caught padded least-squares buffers in the first serialization. Explicit owned
clones preserve every tensor value and produce a 4,265,602-byte packed program
versus 4,281,986 bytes for a matched private-reader/writer reference: 0.383% saving
for this conditional program. The scalar branch itself drops 18,432→2,048 bytes.
The whole native model, external contexts, source code and suffix are not included
in that program-file saving. This is exact reuse, not quantization.

The [factorial identifiability control](EVEN_VALUE_FACTORIAL_V1_IDENTIFIABILITY.json)
shows that the six measured S/R/O removal combinations leave two independent
ambiguities among pair and triple interactions. The missing S+O and R+O removals
are now [registered](EVEN_VALUE_FACTORIAL_NATIVE_V1_PREREGISTRATION.md); neither
is imputed from an assumed linear response. This is the next native experiment.

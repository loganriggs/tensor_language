# One-bit context-gated source basis for the head17.2 port

Fixed rank-eight response and source-PCA coordinates transfer the target effect
but do not preserve collateral readers across all environments. Open the context
slot explicitly without using behavior scores: retain the exported ODD-panel
source-PCA basis `P_odd`; derive a second rank-eight source-PCA basis `P_comp`
from the already-open competing-cues panel; and, for each sequence, choose the
basis retaining more energy of the frozen eight-edge source addition.

The gate is exactly
`argmax_b ||delta_source P_b||_F^2`, over `b in {odd,comp}`. It consumes the same
source-addition port as the projection and one discrete bit; it uses no logits,
readers, labels, fitted gains, or fresh rows. Also form the orthonormal rank-at-
most-16 union span by QR of `[P_odd,P_comp]`.

Freeze and test on all 48 `REGIONAL_VALUE_FRESH_GEO_V1_ROWS.json` rows, which
have zero text overlap with both basis panels and use new city cues, templates,
and cue positions. Evaluate nine arms: the gated rank-eight port, rank-16 union,
each fixed rank-eight basis, four seeded random rank-eight bases, and the
unprojected source program.

Predictions for the gated port: exact instrument and artifact identity; response
replay; bidirectional causal fidelity; preservation; <=75% retained source norm;
minimax score at least 10% below both fixed bases; and score plus `.10` no larger
than the random median. All must pass for promotion. The union arm is a declared
comparator, not a fallback selection.

Price: 48 competing-panel rows plus 48 fresh rows, 30 native/edited executions
over 192 full-model sequences, nine fresh candidates, 864 candidate suffix
sequences plus 96 complete-corner suffix sequences, zero backwards, fits, or
parameter updates.

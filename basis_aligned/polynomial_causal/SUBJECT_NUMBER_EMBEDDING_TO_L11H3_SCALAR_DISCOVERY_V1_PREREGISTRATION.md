# Subject-number embedding-to-L11H3 scalar discovery V1

Registered after the fresh small-mediation confirmation. This assay reuses the
opened 64-row removal panel solely to discover whether the native L11H3 rank-one
change can be computed without its base-head donor value. It makes no OOD or
behavioral claim.

For each row, repeat the frozen same-norm removal of the embedding-number
coordinate and measure

`alpha = axis^T (L11H3_base - L11H3_removed)`,

where `axis` is the already-frozen rank-one L11H3 output direction. Candidate
inputs are only the frozen embedding decoder score `s` and the same-norm removal
distance `d`; number labels, templates, answer logits, and downstream behavior
are forbidden as features.

Predeclared candidate forms, in increasing complexity, are:

1. `score`: `[1, s]`;
2. `score_abs`: `[1, s, |s|]`;
3. `score_signed_quadratic`: `[1, s, |s|, s|s|]`;
4. `score_distance`: `[1, s, |s|, d, sd]`.

Evaluate each by leave-one-noun-pair-out and leave-one-template-out OLS. Select
the simplest form whose worst relative L2 is within `.01` of the best; then fit
its frozen coefficients on all opened rows. Sixty-four pair-block target
permutations provide a matched null.

Instrumentation requires exact artifact hashes, two partial model passes, finite
states, repeated geometry at `1e-5`, and mean `|alpha|` matching the independently
recorded mediation rescue norm within `1e-4`.

A scalar candidate is licensed only if both cross-validation schemes have
relative L2 at most `.25`, cosine at least `.95`, and sign agreement at least
`.90`, and its worst relative L2 improves by at least `.20` over the permutation
median. Passing only licenses a frozen fifth-vocabulary causal test. Failure
means the L11H3 edge retains a native context/head-state port; no richer form may
be selected post hoc from these outcomes.

Price: two partial forwards over 128 sequences, four fixed OLS families, 64
CPU-only permutation controls, no behavior logits, gradients, or updates.

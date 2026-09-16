# Subject-number embedding-to-L11H3 context Möbius V1

Registered after the donor-free score-only scalar null. This assay localizes the
missing transform by crossing the 32 already-opened subject forms from the
removal authority with four prompt contexts: `near`, `behind`, `under`, and
`above`. The same opposite-number attractor is held fixed within each subject.
The `near` and `behind` cells are exact replay cells; `under` and `above`
combinations are unopened. This is factorial localization, not an OOD or
behavioral claim.

For every cell, repeat the frozen same-norm removal of the embedding-number
coordinate and measure

`alpha(subject, context) = axis^T (L11H3_base - L11H3_removed)`.

Apply the unique zero-sum two-factor decomposition

`alpha = grand + subject_main + context_main + interaction`.

The interaction is equivalently the family of Möbius double differences. Report
RMS and sum-of-squares fractions for every component, additive reconstruction
error relative to both raw and centered alpha, all singular values of the
zero-row/column-sum interaction matrix, and rank-1/2 energy capture. Context and
subject labels are used only to localize terms; they are not licensed circuit
inputs.

Instrumentation must reproduce all 64 prior `near`/`behind` alpha values within
`1e-4`, preserve the frozen removal geometry within `1e-5`, use exactly two
partial forwards over 256 sequences, and pass two positive red-team fixtures:

1. a synthetic additive matrix must yield interaction error at most `1e-12`;
2. a synthetic rank-one, zero-sum interaction must be recovered within `1e-12`.

The additive subject-plus-context hypothesis is sufficient only when its raw
relative L2 error is at most `.15`. Otherwise an explicit interaction term is
material only when interaction RMS is at least `5.0` and at least `1e5` times
the replay mismatch RMS. Rank one is provisionally reusable only if it captures
at least `.70` of interaction energy. These descriptive gates license a later
native-feature extraction assay, never a direct circuit claim.

As a specificity control, independently permute the four context columns within
each subject 256 times and recompute additive error. Report the observed error's
percentile and difference from the null median. This tests whether context names
align a shared main effect; it does not null away genuine subject-specific
interaction.

Price: two partial forwards over 256 sequences, 256 CPU-only permutations, no
behavior logits, gradients, fitting, or parameter updates.

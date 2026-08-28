# Suffix-transport v1 implementation amendment: support and moment semantics

Status: **prospective and outcome-blind.** This amendment resolves two reduction
details left implicit in the frozen preregistration. It is part of the scientific
source closure and must be committed, pushed, and audited before any fresh role is
harvested or loaded. It changes no candidate family, rank, optimizer, row, selector,
gate, or interpretation.

## Common scored-token support

Every fit, validation, and final row is truncated to its first 257 tokens. Model
inputs are token positions 0 through 255 and next-token targets are positions 1
through 256. The common scored-token support is model/input positions 64 through 255
inclusive, exactly 192 positions per row. Thus fit has `384*192` scored positions and
validation/final each have `192*192`. Coordinate losses, suffix KL, CE, copy,
frequency bins, intervention-response logits, and their raw sufficient statistics
all use this same integer support unless the frozen intervention definition names one
selected position within it. There is no mean-of-batch-means reduction.

## Centered second moment

For one site, let `Y` be the initialized-Q current-state label matrix on all fit
support, with shape `[N,64]`, accumulated in float64. Freeze the per-coordinate mean

$$
\mu={1\over N}\sum_{i=1}^N Y_i
$$

and the scalar denominator

$$
D={1\over 64N}\sum_{i=1}^N\lVert Y_i-\mu\rVert_2^2.
$$

The receipt stores integer `N`, float64 coordinate sums, float64 coordinate squared
sums, `mu`, total centered sum of squares, `D`, and the ordered support identity.
The authoritative centered sum and `D` use a mergeable float64 Chan/Welford
accumulator; this avoids cancellation in `sum(y^2)-sum(y)^2/N`. Direct and differently
chunked Chan/Welford calculations must agree within
`atol=2e-11, rtol=2e-13`. The algebraic value reconstructed from the separately stored
sums and squared sums is reported as a precision diagnostic, not substituted for the
stable accumulator. `D<=0`, nonfinite values, failure of the stable replay, or a
support mismatch is an integrity failure. The denominator is frozen before the first
optimizer step and never updated.

For a batch/support matrix, the site's local term is

$$
{1\over 64nD}\sum_{i=1}^n\lVert\widehat p_i-Y_i\rVert_2^2,
$$

and the two site terms are summed without an additional factor of one half.

## Separate teacher capabilities

Coordinate labels are exact native projected coordinates evaluated at the captured
current student state for that step. They are recomputed after the pre-step student
trace, detached immediately, and cannot be cached across steps or routes.

OON logits for suffix KL are an autonomous exact-OON forward on the same token rows,
not a spliced current-state coordinate-label forward. They too are detached before
the student loss. Student L/R/S/T scopes have zero original MLP0/1/2 calls; coordinate
label and OON teacher scopes have separate named capabilities and exact call ledgers.

## Gauge scope

The eight registered gauge variants remain post-fit physical replay checks. Rewriting
bases, programs, codes, labels, edits, and `A` must preserve physical outputs and
scores to the registered tolerance. The protocol does not claim that AdamW's
elementwise second-moment trajectory is invariant under a general Haar rotation, and
no refit is performed in a rotated gauge.

# Rung 539: make pending-opener invariance controls causally testable

**Frozen:** 2026-09-03 14:55 UTC, before any R539 model forward

## Why this precedes DAS

Rung 538 found that a complete donor residual at the final position entering
block 8 causally changes the answer for both target-changing prompt families.
Before learning a subspace, we must establish whether the two answer-preserving
families have any causal effect available at that same site. Otherwise a learned
projector producing zero effect on them could merely reflect a zero full-state
ceiling, not semantic selectivity.

## Frozen families and intervention

Use only FIT and SELECT rows from:

- `pending_state_preserved_surface_edit`: wording, nouns, and opener distance
  change while the pending parenthesis and correct answer `)` remain fixed;
- `nonopener_punctuation_substitution`: comma changes to colon before an
  unchanged pending parenthesis, at equal length and with the answer fixed.

For each base/donor direction, replace the complete 1,152-dimensional residual
at that prompt's final position entering block 8 with the other prompt's final
residual. Different-length surface pairs use each prompt's own final position;
this deliberately includes position/distance as a nuisance and is not treated as
an answer-changing interchange.

## Measurements

Let

$$
M(x)=\ell_x(\texttt{)})-\ell_x(\texttt{"})
$$

be the pending-parenthesis answer margin. For a base-to-donor full swap, report

$$
E^{b\to d}=M(b\leftarrow d)-M(b),
$$

and analogously for the reverse direction. For every family, split, and
direction, save every row's $E$ and report its signed mean, mean absolute value,
and a 2,000-resample group-bootstrap 95% lower bound on the mean absolute value.
Also report the root-mean-square change over all 50,304 output logits and the
source-target activation difference.

A family is **causally testable as an invariance control** only if, in all four
split × direction cells:

- the bootstrap lower bound on mean $|E|$ exceeds 0.05 logit units; and
- mean full-vocabulary logit RMS exceeds 0.01 logit units.

Failure does not refute the semantic invariance. It means that zero learned-
projector effect cannot count as strong causal selectivity for that family. The
later projector test must then use an absolute leakage bar and other live
controls rather than divide by a near-zero ceiling.

## Instrument and price

The verified R538 result and terminal audit hashes are inputs. Exactly 128 pairs
(64 per family) are opened, giving 16 combined native forwards plus 32 patched
forwards: 48 forwards, zero backwards. FINAL_TEST, OOD, answer-changing outcomes,
projector ranks, and optimization remain unopened. The source-closed local loader
must verify the pinned checkpoint bytes before constructing the model.

This rung measures whether our negative controls are informative. It does not
search for a circuit or compress an activation.

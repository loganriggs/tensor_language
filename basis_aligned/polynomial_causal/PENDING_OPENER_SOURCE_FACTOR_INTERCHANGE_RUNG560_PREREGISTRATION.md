# Rung 560 preregistration: pending-opener score × payload source-factor interchange

**Frozen:** 2026-09-03 17:35 UTC, before any R560 model call

## Why this experiment

R546 established that the complete 128-dimensional output of L13H8 can causally move the predicted closer. R556 then
showed that linear subspaces with dimensions 2 through 16 recover the requested answer changes but also move all three
answer-preserving controls. Increasing the dimension is therefore not licensed.

R560 changes the unit of analysis. Bilinear attention already factorizes the head's contribution from source position
$k$ to final query position $q$:

$$
c_{qk}=p_{qk}u_k,
$$

where

$$
p_{qk}=\frac{\langle q_q,k_k\rangle}{128}
        \frac{\langle q'_q,k'_k\rangle}{128}
$$

is the product of the two query/key scores, and

$$
u_k=W^O_{13,8}v_{13,8,k}\in\mathbb{R}^{1152}
$$

is head 8's value from source position $k$ after its output projection. The head output is the sum of these terms over
earlier positions.

Each R545 endpoint has a registered correct closer. The source position is defined before outcomes by mapping that
closer back to its opener—token 8 `)` to token 357 `(`, token 60 `]` to token 685 `[`, and token 1 `"` to opening-quote
token 366—and taking the final occurrence of that opener. A CPU construction check must find exactly the intended
source in all FIT/SELECT rows. This answer-bound rule is necessary because the audit found that all 108 FIT/SELECT
`completed_then_reopened` rows repeat the base value in `proposed_variable_donor`; their donor prompt and registered
donor answer are consistent, but that metadata label is not. This semantic
mapping makes the unequal-length distance-extension pairs valid: base and donor source positions are located
independently and their scalar/vector factors can still be interchanged.

## Interventions

At only the final query position, subtract the native semantic-source term and add one of:

$$
\begin{aligned}
c_\text{score} &= p_\text{donor}u_\text{base},\\
c_\text{payload} &= p_\text{base}u_\text{donor},\\
c_\text{joint} &= p_\text{donor}u_\text{donor}.
\end{aligned}
$$

All other source-position terms, heads, layers, and downstream computations remain live. Both base-to-donor and
donor-to-base directions are evaluated.

An adjacent-wrong-source control repeats each intervention with the donor factors from the token immediately before
the registered opener. It must not reproduce the intended answer change.

## Data and measurements

Use only the frozen R545 FIT rows for intervention choice and SELECT rows for one held validation. FINAL_TEST and OOD
remain unopened. All five independently constructed families are required:

- answer-changing: direct three-value type substitution; completed-then-reopened order;
- answer-preserving: surface rewrite; distance extension; non-opener punctuation edit.

For answer-changing rows, recovery is the factor intervention's closer-logit movement divided by the row-matched
complete-L13H8 movement saved by R546. For answer-preserving rows, report:

- absolute change in the correct closer's margin over the other two closers;
- that change divided by the row-matched complete-head change;
- full-vocabulary logit RMS divided by the row-matched complete-head RMS.

For each target row and direction, also report the bilinear interaction in closer-logit movement:

$$
I=\Delta_\text{joint}-\Delta_\text{score}-\Delta_\text{payload}.
$$

This directly tests whether score and payload effects add or depend on one another.

## Frozen FIT choice

Each of `score`, `payload`, and `joint` is evaluated independently. A candidate passes FIT only if:

- in every answer-changing family and direction, median recovery is at least $0.50$, the group-bootstrap 95% lower
  mean recovery is above zero, and at least 75% of rows have positive recovery;
- in every answer-preserving family and direction, mean absolute closer-margin change is at most $0.10$ logit, at most
  25% of the row-matched complete-head closer change, and mean full-vocabulary RMS is at most 25% of the row-matched
  complete-head RMS;
- in every target family and direction, adjacent-wrong-source absolute mean recovery is at most $0.25$.

Choose among passing candidates by fewest transplanted factors (`score` and `payload` cost one; `joint` costs two),
then by the largest worst target bootstrap lower bound, then alphabetically. If none passes, stop with a source-factor
null and do not open SELECT.

## Frozen SELECT decision

Evaluate only the FIT-chosen intervention, its adjacent-wrong-source control, and the native replay. Apply the identical
target, invariance, and wrong-source bars. Any failed cell is a null. Do not change the source definition, add factors,
increase a dimension, or open FINAL_TEST/OOD after seeing SELECT.

## Instrument checks

- Exact checkpoint and input hashes are required.
- Manual attention replay must match the native model before intervention.
- The projected factor product must reconstruct the directly projected semantic-source contribution.
- Exact native, replay, intervention, split, and model-call counts are recorded.
- A no-model dry run must exercise source-position mapping, deterministic candidate choice, and interaction scoring.

A held result identifies whether the pending-opener state travels through the position-selection score, the source
payload, or their product. It does not yet compile that factor into weights or establish OOD generalization.

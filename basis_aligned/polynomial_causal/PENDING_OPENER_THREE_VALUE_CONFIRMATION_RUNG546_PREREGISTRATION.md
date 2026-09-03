# Rung 546 preregistration: fresh three-value capability and L13H8 confirmation

**Frozen:** 2026-09-03 15:41 UTC, before loading the model or evaluating any R545 outcome

## Question

Does the post-R544 candidate domain—parenthesis, square bracket, and quote—survive entirely fresh prompts, and does
the complete output of attention layer 13 head 8 causally transfer both answer-changing constructions while giving
the three answer-preserving controls enough effect to detect a selective learned intervention later?

This is a capability and complete-state confirmation only. It fits no subspace, chooses no rank, and cannot inspect
FINAL_TEST or OOD.

## Frozen data and site

The run is bound to R545's row file and receipt. It evaluates only FIT and SELECT: 108 semantic groups, five paired
families per group, hence 540 prompt pairs. The sole site is the complete 128-dimensional L13H8 pre-output-projection
state at each prompt's actual final position. The site was specified before R545 outcomes because it passed all
four-value full-state target/control gates in R544.

## Native capability bar

For the correct closer $a$ among the three candidate closer tokens, define

$$
m_a = \operatorname{logit}(a)-\frac{1}{2}\sum_{b\ne a}\operatorname{logit}(b).
$$

For each answer-changing family, split, and ordered delimiter pair, at least 75% of base prompts and 75% of donor
prompts must have $m_a>0$. The pooled mean over both directions must be positive with a 95% bootstrap lower bound
above zero. For each answer-preserving family and split, at least 75% of both sides must natively prefer the unchanged
correct closer.

## Complete-state target bar

For both answer-changing families, both patch directions, and both splits, replacing the complete L13H8 state must
move the donor closer above the base closer in the intended direction. Each pooled cell requires positive mean,
95% bootstrap lower mean above zero, and at least 70% positive rows. Every ordered delimiter pair also requires
positive mean and at least 50% positive rows.

## Complete-state control-liveness bar

For every answer-preserving family, direction, and split, the complete-state swap must have a 95% bootstrap lower
mean absolute correct-closer margin change above 0.03 logits and mean full-vocabulary logit root-mean-square change
above 0.01 logits. This does not require the change to have either sign; it only proves that a later near-zero learned
projection effect would be informative.

## Budget and decision

With batch size eight, 540 pairs require 68 batches. Each batch has one native forward and two patched forwards, for
exactly 204 model forwards, zero backwards, and no weight update. A pass requires the native capability bar, every
target full-state bar, and every control-liveness bar. Any failure blocks subspace fitting and is recorded without
opening FINAL_TEST/OOD.

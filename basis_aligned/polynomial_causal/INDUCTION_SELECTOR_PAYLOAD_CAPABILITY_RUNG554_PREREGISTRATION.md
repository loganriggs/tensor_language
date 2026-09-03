# Rung 554 preregistration: induction selector × payload native capability

**Frozen:** 2026-09-03 16:36 UTC, before opening any model output

## Question

Does the unmodified model actually implement the two-factor induction behavior in the outcome-blind R552 dataset?
This must be established before searching for a selector site, a payload site, or a lower-dimensional representation.

Each group has two source→payload pairs and four conditions:

$$
A\rightarrow B,\ C\rightarrow D,\qquad
(S,P)\in\{0,1\}^2,
$$

where $S$ chooses whether the final query is $A$ or $C$, and $P$ chooses the original or swapped payload assignment.
The correct answers are $B,D,D,B$ for $S_0P_0,S_1P_0,S_0P_1,S_1P_1$ respectively.

For a condition with correct payload $a$ and the other payload $a'$, define the native margin

$$
m=\operatorname{logit}(a)-\operatorname{logit}(a').
$$

Only FIT and SELECT may be evaluated. FINAL_TEST and OOD remain unopened.

## Frozen predictions and bars

### A. Four-cell capability

For each split and each of the four $(S,P)$ cells separately:

- at least 75% of groups must have $m>0$;
- the 95% group-bootstrap lower bound on the mean $m$ must be greater than zero.

All eight split-by-cell tests must pass. This prevents strong performance in one selector or assignment condition
from hiding a failed condition.

### B. Relation-preserving controls

For the irrelevant-source edit, filler change, and lag extension, score base and edited prompts separately. In each
split-by-family-variant-by-endpoint cell, at least 75% of groups must have $m>0$ and the 95% group-bootstrap lower mean
margin must exceed zero. These edits change real tokens while preserving the selected source→payload relation.

### C. Selected-match necessity and selectivity

For each match-breaking row, let

$$
d_{\mathrm{selected}}=m_{\mathrm{base}}-m_{\mathrm{selected\ match\ broken}}.
$$

Pair it by semantic group with the irrelevant-source row, which starts from the same factorial condition, and define

$$
g=d_{\mathrm{selected}}-
\left|m_{\mathrm{base}}-m_{\mathrm{irrelevant\ source\ edited}}\right|.
$$

In each split, at least 70% of rows must have $d_{\mathrm{selected}}>0$, its 95% group-bootstrap lower mean must exceed
zero, and the 95% group-bootstrap lower mean of $g$ must exceed zero. The last comparison rejects generic damage from
editing any earlier source token.

## Instrument and stopping rule

- Frozen inputs are the R552 rows, R552 receipt, R553 independent audit, and this file.
- All 864 unique FIT/SELECT sequences are evaluated exactly once in batches of 32: 27 model forwards, zero backward
  passes, and no weight updates.
- The checkpoint, artifact hashes, exact sequence count, forward count, split names, and nonzero token edits are live
  checks.
- No component, head, subspace, rank, or regularization strength is selected.

If A, B, and C all pass, separately preregister complete-state selector and payload/write site ceilings. If any part
fails, preserve the native-capability null and do not spend a component sweep on this synthetic task without first
changing the task construction. This rung is behavioral evidence, not yet circuit identification.

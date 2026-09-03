# Rung 537: pending-opener multi-counterfactual identification

**Frozen:** 2026-09-03 14:25 UTC, before any new model forward

## Question

Does one causally meaningful state at the final prediction position represent which
opener type remains pending, or did the earlier low-rank DAS result learn a shortcut tied
to deletion, position shift, punctuation identity, or templated wording?

This rung tests circuit identification. A small rank by itself cannot pass.

## Frozen rows

The CPU-only row authority is
`basis_aligned/bilinear_quotient/pending_opener_multifamily_rows_rung537.json`, SHA-256
`c62cdf3929231e06de6883d74f3ab2c86bd524e02474bb2259267d6976e9e7d9`.
It contains 96 shared groups and 288 pairs:

| Split | Groups | Rows |
|---|---:|---:|
| FIT | 48 | 144 |
| SELECT | 16 | 48 |
| FINAL_TEST | 16 | 48 |
| OOD | 16 | 48 |

Every group contributes the same three families and belongs to one split in all of them.
Lexical pools are disjoint across splits. FINAL_TEST and OOD may not select a site, rank,
seed, threshold, or family definition.

The families are:

1. `opener_type_substitution` (answer-changing interchange): base and donor differ by
   exactly one token, space-prefixed `(` versus `"`; the required next closer changes
   from `)` to `"`. Length and lexical-token multiset are identical.
2. `closed_then_reopened_type` (answer-changing interchange): both prompts contain a
   completed first parenthetical/quotation and then open the other type. Word content,
   token length, and lexical-token multiset are identical, but the punctuation structure
   and required closer differ. This is deliberately not the same single-token edit.
3. `pending_state_preserved_surface_edit` (invariance): wording, nouns, and distance
   change while a parenthesis remains pending and the required closer stays `)`.

The old opener-deletion family is not used to identify the shared state because deletion
also shifts every later position. It remains legacy evidence and a possible necessity
test after position-matched controls exist.

## Stage A: capability and counterfactual validity

For a prompt $x$, let $\ell_x(t)$ be the final-position logit for token $t$. For an
answer-changing pair with base answer $y_b$ and donor answer $y_d$, define

$$
m_b=\ell_b(y_b)-\ell_b(y_d),
\qquad
m_d=\ell_d(y_d)-\ell_d(y_b),
$$

and the symmetric two-endpoint separation

$$
S=\frac{m_b+m_d}{2}.
$$

On FIT and SELECT separately, each answer-changing family must have:

- at least 75% of pairs with both $m_b>0$ and $m_d>0$;
- mean $S>0.5$ logit units;
- document/group-bootstrap 95% lower confidence bound for mean $S$ above zero.

The invariance family must have the same answer by construction and at least 75% correct
next-closer preference against the quote alternative on both sides. Failure of either
answer-changing family stops before DAS. FINAL_TEST and OOD remain unopened.

These are capability gates, not evidence that a shared internal circuit exists.

## Stage B: intervention-site ceiling

Candidate sites are the final-position residual entering layers 8 through 14, the full
L13H8 output, and product activations of MLPs 8 through 14. GPU work must use the managed
runner. Sites are screened on FIT and selected once on SELECT.

For base-to-donor interchange at site $s$, let

$$
\Delta_s=
\bigl[\ell_{b\leftarrow d,s}(y_d)-\ell_{b\leftarrow d,s}(y_b)\bigr]
-\bigl[\ell_b(y_d)-\ell_b(y_b)\bigr].
$$

The reverse direction is defined analogously. A site is live for a family only if the
mean signed $\Delta_s$ is positive in both directions, its group-bootstrap 95% lower
bound is above zero, and at least 70% of individual directions move donorward. Both
answer-changing families must be live at the same site. This prevents selecting a site
that only responds to the one-token edit.

If no common site passes, the shared pending-opener claim is not fit. Report the family ×
site ceiling matrix and stop; do not optimize rank around a dead site.

## Stage C: shared projector

Only after Stages A and B pass, fit ranks $k\in\{1,2,4,8\}$ on FIT. Select rank and seed
using SELECT. The primary identification test is two-way cross-family transfer:

- fit on direct type substitution, test the structural close-and-reopen family;
- fit on the structural family, test direct substitution;
- fit a shared projector and compare it with shared-plus-family-specific projectors.

Normalize learned-subspace movement by the complete-site movement in the same direction.
A projector passes SELECT only if both cross-family recoveries are positive with 95%
lower bounds above zero, median normalized recovery is at least 0.50, and the random-rank
control is below 0.10. Because prior DAS exceeded its natural full-swap ceiling, also
report absolute donor margins, intervention norm, and dose-response at scales
$0,0.5,1,1.5$. Overshoot cannot be interpreted as stored dimensionality.

The invariance family is evaluated by applying the learned projector/removal on both
surface variants. The absolute difference between their normalized causal effects must
be at most 0.15 on SELECT. Non-opener punctuation, wrong-closer, and random-subspace
controls must be added and frozen before a claim can advance from `proposed` to
`specified` in the canonical record.

## Stage D: one-shot tests and interpretation

After site/rank/seed/bars are fixed, run FINAL_TEST once, then OOD. Promotion to
cross-family identification requires all of:

- both directions of cross-family transfer retain the SELECT sign;
- pooled normalized recovery at least 0.50 with a 95% lower bound above zero;
- the invariance-effect difference at most 0.15;
- random and punctuation controls remain below their frozen bounds;
- OOD retains at least 50% of FINAL_TEST recovery and the same sign;
- unrelated registered circuit endpoints remain within the project's collateral bounds.

Passing identifies a causal pending-opener state at the selected activation site. It does
not yet identify its weight implementation. Weight compilation is a later stage and must
rerun every family and control physically.

## Registered nulls and next actions

- If only the direct edit works, treat the result as punctuation-token-specific.
- If only the structural edit works, test whether ordering/closure rather than pending
  type is the variable.
- If both work separately but do not cross-transfer, fit a shared-plus-private model and
  split the causal variable if only private parts are effective.
- If interchanges work but invariance fails, expand wording/distance controls; do not
  call the subspace semantic.
- If the full-site ceiling is live but every learned rank fails, report a non-low-rank or
  optimization-limited result; do not weaken the circuit criteria.

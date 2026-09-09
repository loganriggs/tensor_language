# V26 token-role interface audit — 2026-09-09

The existing v23 token-role partition works on all 64 sealed v26 rows at padded width 17.
Every real token belongs to exactly one role, padding belongs to none, endpoints are unchanged,
changed-role tokens really differ, and prefix/suffix-role tokens match. Every row has a nonempty
causal changed-source to matched-suffix-destination cell. The row-authority hash reproduces
`805b734109a275de40f809c639f90ca2a524d82ab39467f10e2bfeb7b5ca9c15`.

This is a CPU interface result, with zero model forwards and no capability or causal results opened.
The executable is `../bilinear_quotient/ops/audit_v26_token_role_topology.py`; the adjacent JSON
contains source hashes and every row's position lists.

The roles are defined by paired token equality, not grammatical annotation. In particular, the
subordinate-clause target has 3–5 matched-suffix tokens, while its paraphrase control has 7–9.
The postnominal target has 7–9 unchanged-prefix tokens and four matched-suffix tokens. Thus matching
role names across structures does not establish that the same grammatical material is intervened on.
These are prospective interpretation constraints, not reasons to modify the sealed population.

The continuation remains the already-queued v26 capability gate. Only its complete pass licenses a
separately registered causal transfer. That transfer should preserve the existing role algorithm,
report each construction separately, and distinguish complete-head transfer from directed-cell
transfer. Failure must preserve the null; changing professions again would not test a new structure.

Even successful structural contrast transfer would leave the extraction objective open: the native
model still supplies the head inputs and the downstream computation, and the outcome is an is/was
contrast rather than full-distribution prediction. Recursive dependency closure and joint task
prediction remain the primary directions in the concluding user clarification of math_ideas.md.

# R585 iteration-6 final independent pre-execution review

Date: 2026-09-03 UTC

## Decision

**APPROVE exact commit `62680bfc78a9c119c40aca8e8a7f5c1eec30ec87` for the registered R585 execution.**

The iteration-5 cross-split blocker is closed in both manifest orderings. The
repair filters structural identities to the opened split, while missing evidence
for a required identity within that split still hard-aborts. I found no remaining
execution blocker in the exact producer, adapter, tests, or dry run.

This approval licenses execution of these exact bytes only. It does not predict
or endorse a scientific outcome.

## Outcome blindness and exact reviewed bytes

I reviewed immutable Git blobs and a detached worktree at the stated commit,
never the moving producer. At `2026-09-03T23:13:13Z`, existence-only checks found
the R585 result JSON, receipt JSON, and evidence directory all absent. I did not
open, stat, or inspect an R585 outcome. No model, CUDA/GPU, queue, board, or
registry operation was performed.

| File | SHA-256 |
|---|---|
| producer | `3963ac0e666874c4d5f35d7be79d1834d0b88b003643acd9950d504dca29e2a1` |
| owner test | `237946ac4fa7ef5a65a5c6269ad7cfd064195aef993e469df2b06b9b78600024` |
| dry run | `c56f0feee2060966fa3fd4210dac0bdc7c945c779eb8c3c5f3068aa6c3fd6a5c` |
| managed adapter | `b3f80585e5b18657ad52604c722f4cf1a492480efebed2b55d572785098ed8b4` |
| adapter test | `8df90f0ea8160ad167fa6fcb77462ddfc5e4a29068f81fa5bb44bf0fff86d931` |

The producer, adapter, and dry run also pin the iteration-5 review and test at
`6d8c82416e1c7c8eb831633feeb3905bb92a818ba0f1f71332f571c2ba945d39`
and `aaa01d044a0609aaee17371d7945d93e61e80241674c9bbc4e706880f83dab34`.

## Cross-split repair

`structural_identity_failures` now requires the current `split`. For each
registered identity, it resolves the referenced cell and skips it exactly when
`cell["split"] != split`. Within the selected split it still requires:

1. the direction's score record;
2. every non-replay arm's captured final vocabulary-logit vector;
3. the recipient replay final-logit vector for replay identities; and
4. all frozen inserted terms needed for the independently reconstructible local
   comparison.

The independent attack used a manifest containing one FIT and one SELECT
identity. Supplying exactly one split's complete rows succeeds for both FIT and
SELECT, with identity order `(FIT, SELECT)` and the reverse `(SELECT, FIT)`.
Removing a required direction from the selected split still raises
`structural identity direction missing before publication`.

The full-logit hard abort from iteration 5 remains intact:

$$
\max_v |\ell^{(a)}_v-\ell^{(b)}_v| > 10^{-5}
\quad\Longrightarrow\quad
\text{abort before publication}.
$$

Removing an arm's final logits or recipient replay logits also aborts. Equal
local inserted vectors cannot mask unequal end-to-end logits. The separately
named `structural_inserted_term_identity_checks` remains reconstructed from
saved factor arrays.

## Whole-package audit

The preceding repairs remain closed:

- exact authority parsing and semantic endpoint/direction joins;
- exact 20,736 endpoint-by-site-by-role operation census;
- frozen recipient and donor factors before intervention;
- independent remainder and live `inserted = removed + delta` reconstruction;
- full primitive-logit, margin, CE, vocabulary, `n/d/q`, activity, and scale
  identities;
- exact sorted FIT and SELECT invalid-instrument derivation;
- native-attention, replay/native, incomplete capture, live-factor, hook, and
  nonfinite hard-abort boundaries;
- realized ordered 124-cell bootstrap census for every scored split;
- score reports and terminal failure-list reconstruction;
- atomic receipt-last publication and conservative recognized-only recovery;
- FIT-first terminal precedence, 459 FIT plus 231 SELECT forwards for 690 total,
  zero backwards and updates, and FINAL/OOD closure.

The manifest-derived full-logit capture remains exact at 5,184 non-replay
`(directed_id, arm)` pairs: 3,456 FIT and 1,728 SELECT. At vocabulary width
50,257, its float32 storage is 694,752,768 bytes in FIT and 347,376,384 bytes in
SELECT, and those vectors are discarded after the per-split check. This remains
within the registered transient-memory bound.

## Tests

Exact candidate tests in the detached worktree:

- owner suite: `60 passed`;
- managed adapter: `11 passed`;
- manifest: `14 passed`;
- replacement adversarial: `13 passed`;
- next-wave adversarial: `11 passed, 3 xfailed`;
- first repair review: `4 passed, 3 xfailed`;
- second repair review: `5 passed, 4 xfailed`;
- prior final review: `8 passed, 3 xfailed`;
- iteration-4 review: `4 passed, 1 xfailed`;
- iteration-5 review: `5 passed, 1 xfailed`;
- producer and adapter static gates: pass;
- producer and adapter preflight: pass;
- managed no-argument model-free dry run: pass with zero model calls.

Historical strict xfails remain bound to the older immutable candidates whose
blockers they document; they are not failures of iteration 6.

New independent acceptance test:

`basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_iteration6_final_review_adversarial.py`

- `7 passed`;
- static gate: pass;
- preflight: pass;
- `git diff --check`: pass.

## Remaining scientific uncertainty

Execution validity does not establish circuit identification. Even a held R585
result would show that the frozen score/payload factor is sufficient for these
registered counterfactuals, controls, and SELECT data; it would not by itself
show stable identification across broader OOD conditions, selective removal of
this computation without unrelated damage, or reuse in other induction
contexts. Those remain post-outcome scientific tests, not pre-execution
blockers.

## Reusable lesson

Every validator that receives split-local rows and a global manifest should be
tested with two splits in both manifest orders. The invariant has two sides:
other-split rows must not be required, while missing current-split direction,
arm, or replay evidence must fail closed. Iteration 6 now satisfies both sides.

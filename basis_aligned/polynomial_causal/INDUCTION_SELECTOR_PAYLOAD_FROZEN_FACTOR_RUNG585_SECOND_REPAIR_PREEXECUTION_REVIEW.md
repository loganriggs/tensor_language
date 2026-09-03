# R585 second repair: independent exact-byte pre-execution review

<!-- BQLANE: cpu -->

**Review date:** 2026-09-03 UTC
**Reviewed commit:** `a19c029fd178e716a94024d573a82308e78d32be`
**Verdict:** **BLOCKED — do not execute or enqueue these exact bytes**

## Exact candidate and outcome boundary

This review loaded the candidate from immutable Git blobs at `a19c029fd`, not
from the concurrently changing working tree:

- producer:
  `8a4f20d06dd04cd81d6bb8c94377ee987b66bea4201395e61bbe23a1b5dd9a8c`;
- owner test:
  `57e52e8da53f3a6e7b194efb64f56d1ff9fb442c2c39547a6f1fed4263a10653`;
- dry run:
  `6fb41eb862c00f27673cfe694cf8670eae23f1d60a6a5dd85a35a5309b7e90f5`;
- managed adapter:
  `efaeb3ee746f1c18caa52ab4466d403a31c0b2fe509a2c118b39ebcdad10e2d9`;
- adapter test:
  `67486003c4f14208179fd8b52099951b1212570d0830632ce37e85f3abcdddbe`.

The live R585 result, receipt, and evidence namespaces were absent at the start
of this review and again at `2026-09-03T22:03:23Z`. Only existence checks were
used; no R585 outcome was opened or created. The review used no model, GPU,
queue, board, or registry access.

## What the second repair closes

The second repair closes all three literal failures in the preceding exact-byte
review:

1. **Every model-free terminal is excluded from production validation.** All 13
   registered held/null/instrument fixtures are rejected unless the explicit
   test-only `allow_model_free_fixture` switch is supplied. A real completed
   terminal must provide evidence for the phase it claims to have run.
2. **Evidence IDs now match the frozen authority.** Endpoint IDs, directed
   ID/arm pairs, and endpoint/site pairs are compared literally with orders
   rebuilt from the frozen execution authority. The old invented-ID attack is
   rejected.
3. **Managed recovery is reachable.** The adapter verifies its pinned bytes,
   calls producer recovery, and only then requires unused final namespaces. A
   recognized stale stage plus a recognized partial result is quarantined
   through the managed path. Arbitrary bytes and any complete three-path final
   namespace are preserved and refused.

The phase contracts are now explicit:

| Evaluated phase | endpoints | directions | arm rows | factor rows | operations | forwards |
|---|---:|---:|---:|---:|---:|---:|
| FIT | 1,728 | 3,744 | 11,232 | 6,912 | 13,824 | 459 |
| FIT + SELECT | 2,592 | 5,616 | 16,848 | 10,368 | 20,736 | 690 |

All seven arrays have the corresponding FIT-only or FIT+SELECT first
dimension. Their fixed trailing dimensions are four sites, two equality
factors where applicable, and hidden width 1,152. FIT and SELECT each realize
exactly 124 ordered bootstrap cells when that split reaches scoring. An FIT
instrument failure has no scored split; a SELECT instrument failure retains
only the completed FIT score; other nulls retain exactly the scored phases.
FINAL_TEST and OOD remain closed. The declared budget is 459 FIT plus 231 SELECT
for at most 690 forwards, zero backwards, and zero parameter updates.

The producer and adapter both pass the repository static gate, including the
fixed unique exception-handler aliases, and preflight. The managed model-free
dry run reports no model or CUDA use and no opened outcome.

## Remaining blocker: saved evidence is not bound to the computation

The validator now checks the right number, order, IDs, hashes, dtypes, shapes,
finiteness, and the endpoint equality-factor reconstruction. It still does not
recompute four essential meanings of the saved files.

### Endpoint IDs do not bind the saved semantic coordinates

The endpoint JSONL rows contain the actual tokens, length, final position,
source and payload positions, condition, answer token, and alternative answer
token. Validation compares only each endpoint ID with the authority. The
independent attack kept the exact endpoint ID but changed its token sequence;
the complete result passed. Therefore the file can claim the correct endpoint
name while describing a different input or causal coordinate.

### Directed rows do not inherit donor/recipient semantics from the authority

For each exact directed ID and arm, the JSONL row also names its recipient and
donor endpoint and contains the semantic condition fields that define the
counterfactual. The validator checks only the directed ID and arm. The
independent attack retained every expected ID and arm but replaced every
`recipient_endpoint_id` and `donor_endpoint_id` with invented IDs. The complete
result passed.

Thus an evidence row can be attached to the correct direction label while
describing a different or nonexistent counterfactual.

### Saved intervention vectors are not checked against endpoint factors

For site $s$, direction $d=(r\rightarrow q)$, and arm $a$, the experiment
defines a frozen inserted vector

$$
z_{d,a,s} = \sum_{k=1}^{2} e_{\operatorname{score}(a),s,k}
u_{\operatorname{payload}(a),s,k},
$$

where score-only uses donor $e$ with recipient $u$, payload-only uses recipient
$e$ with donor $u$, and joint uses donor $e$ and donor $u$. The saved arrays
must satisfy

$$
\operatorname{live\_removed}_{d,a,s}
+ \operatorname{hook\_delta}_{d,a,s} = z_{d,a,s}.
$$

The validator never evaluates this identity. The independent attack used
finite, correctly shaped, correctly ordered arrays whose two sides disagree;
the complete result passed. Consequently the evidence does not yet prove which
quadratic term was actually removed and inserted.

### Primitive logit identities are not rechecked from the saved rows

Runtime code checks the primitive intervention records before writing them,
but the result validator does not repeat that check on the JSONL bytes. The
independent attack changed `correct_margin` to contradict
`answer_logit - other_logit`, refreshed the descriptor hash and byte count, and
the complete result passed. The same gap applies to the saved CE and vocabulary
RMS identities and prevents an independent consumer from rebuilding the score
objects from the evidence.

These are one blocker class rather than four unrelated findings: the package
binds **where rows are stored**, but not yet **what computation those rows
represent**.

## Required repair

Before execution, extend completed-evidence validation as follows:

1. Rebuild an `endpoint_id -> frozen endpoint` map and require every endpoint
   JSONL row to equal its authority entry on split, tokens, length, final
   position, source/payload positions, condition, answer token, and alternative
   answer token before accepting its replay/native measurements.
2. Rebuild a `directed_id -> frozen direction` map and require every directed
   JSONL row to equal its authority entry on split, recipient/donor endpoint,
   row/group/family/variant, recipient condition, direction, control kind,
   answer-change flag, and answer-token IDs. Require exactly the three arms.
3. Join each directed row to the endpoint-indexed saved $e/u$ arrays. Recompute
   the arm-specific frozen inserted vector at all four sites and require
   `live_removed + hook_delta` to equal it within the frozen numerical
   tolerance. Recompute each per-site delta norm and the recorded insertion
   activity from `hook_delta`.
4. Run `validate_primitive_logit_identities` on the parsed saved directed rows
   and fail validation on every returned clause. Rebuild the score inputs from
   those rows, or otherwise bind each score object's primitive group values to
   an independently hashed/reconstructed projection of the JSONL rows.
5. Add the four attacks in the accompanying independent test to the owner
   suite, for both FIT-only and FIT+SELECT phase contracts.

The existing nonfinite policy is otherwise sound: nonfinite result fields,
arrays, JSON, and JSONL fail before publication; nonfinite primitive records
produce an integrity failure and abort rather than licensing a scientific null.

## Validation performed

- Exact Git-blob hashes: all five candidate hashes matched.
- Row, manifest, replacement-adversarial, and producer-owner suites:
  **77 passed**.
- Adapter suite, run after restoring the hash-pinned dry-run artifact:
  **11 passed**.
- A combined run after the producer test rewrote its own worktree dry run gave
  80 passes and eight adapter hash failures. This is an expected test-order
  artifact: the owner dry-run test writes path-dependent provenance in the
  disposable worktree. Restoring the exact committed dry-run bytes makes all
  11 adapter tests pass. No candidate or outcome bytes were changed.
- Producer gate/preflight: **PASS/PASS**.
- Adapter gate/preflight: **PASS/PASS**.
- Managed `BQLIB_DRYRUN=1` adapter execution: **PASS**, zero model/GPU work.
- Fresh independent regression:
  `basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_second_repair_review_adversarial.py`.
  Default: **5 passed, 4 strict xfailed**. With `--runxfail`, the four attacks
  fail because validation raises no error: wrong endpoint tokens, wrong
  donor/recipient semantics, false live-plus-delta identity, and inconsistent
  primitive margin.
- Fresh regression gate/preflight and `git diff --check`: **PASS**.

## Disposition

Do not execute or enqueue commit `a19c029fd` with the five hashes above. Repair
the single computation-binding blocker prospectively and request one more
exact-byte, outcome-blind review.

Reusable critic lesson: an exact census and exact row IDs establish evidence
membership, but not evidence meaning. Every saved causal row must be joined to
its frozen donor/recipient semantics, and every saved intervention vector must
be recomputed from the saved factors that define the claimed operation.

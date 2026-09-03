# R585 iteration-5 final independent pre-execution review

Date: 2026-09-03 UTC

## Decision

**BLOCK exact commit `e63fa74b70f722e7d993d6bc2d4b03372e98f7ce`. Do not execute these bytes.**

Iteration 5 correctly repairs the iteration-4 structural-control regression, but
introduces one new execution blocker: structural identities are validated once
per split using split-local records, while the validator now treats identities
from the other registered split as missing errors. The FIT pass therefore
always aborts upon reaching a SELECT identity. No valid run can reach FIT
scoring or SELECT execution.

I found no other blocker in the exact package.

## Outcome blindness and exact bytes

I reviewed immutable Git blobs and a detached worktree at the stated commit,
never the moving producer. At `2026-09-03T23:04:20Z`, existence-only checks found
the R585 result JSON, receipt JSON, and evidence directory all absent. I did not
open, stat, or inspect an R585 outcome. No model, CUDA/GPU, queue, board, or
registry operation was performed.

| File | SHA-256 |
|---|---|
| producer | `33b5cbbc26e5ba62bb60a5bf62d69a1ef7ea51d1bf64e51fd3b95049e55f4327` |
| owner test | `1a3419e3aa19abc2b03424d02ff5c474296472811e780dd10bbde4cc34f410d7` |
| dry run | `ac02054a22452911150e173792f28902351fdf1b04d04b87007a570837cf026d` |
| managed adapter | `b156bf741dfbc7dd57e669bc4a9dc981092b7308d14a1c45c05e91a2e5944f1b` |
| adapter test | `6a8ce51bc139b5a070adb72ffd2abaf7e811b427a2a0b7189259e6cc20de4bb0` |

The adapter additionally pins the iteration-4 review and attack at hashes
`302e9ba506931e8513c5f069a332cc1445342ab282344269ee00b866a9e6a9fc`
and `29d8023ddbc56c70df7097394717d70fe7a2b6289fae0bce197f0d0e8f9eafd3`.

## Iteration-4 blocker is closed

All three requested parts of the previous repair are present and work in direct
CPU attacks:

1. **Live final-logit check.** Equal local inserted tensors with final
   vocabulary logits differing by 1.0 now raise
   `structural full-vocabulary identity failed before publishable evidence`.
   Missing required final-logit vectors also hard-abort.
2. **Manifest-derived capture.** `required_structural_full_logit_pairs` derives
   all and only non-replay arms used by registered structural identities. Replay
   logits come from the already captured recipient replay. The exact authority
   contains 5,184 captured `(directed_id, arm)` pairs: 3,456 in FIT and 1,728 in
   SELECT.
3. **Separate saved local check.** The reconstructible evidence is now named
   `structural_inserted_term_identity_checks`. Its value is recomputed from
   saved `native_e.npy` and `native_u.npy`; it is not presented as the live
   end-to-end check.

The runtime now checks

$$
\max_v |\ell^{(a)}_v-\ell^{(b)}_v| \le 10^{-5}
$$

on the actual final vocabulary logits before separately recording the local
inserted-term error

$$
\max_s\left\|t_{a,s}-t_{b,s}\right\|_\infty,
\qquad
t_{a,s}=e_{i(a),s,0}u_{j(a),s,0}+e_{i(a),s,1}u_{j(a),s,1}.
$$

That is the intended distinction between an end-to-end implementation sanity
check and independently reconstructible factor evidence.

## New blocker: split scope is missing

`run_science` performs the following inside its `for split in SPLITS` loop:

```python
structural_identity_failures(
    split_records,
    split_vectors,
    execution["manifests"],
    all_replay,
    frozen_insertions=frozen_insertions,
)
```

`split_records` and `split_vectors` contain only the current split. In contrast,
`execution["manifests"]` contains structural identities for both FIT and SELECT.
The updated function iterates every identity and now executes:

```python
if (directed_id, "score") not in record_by_key:
    raise RuntimeError(
        f"structural identity direction missing before publication: {directed_id}"
    )
```

During the FIT call, every SELECT direction is absent by design. The former
`continue` happened to tolerate this; replacing it with a hard error without
adding a split filter makes the producer unconditionally abort. The owner tests
exercise one-split miniature manifests, so they do not expose the problem.

The independent two-split attack contains valid complete FIT score/joint logits
and a registered SELECT identity. It reproduces the unconditional failure:

```text
RuntimeError: structural identity direction missing before publication: select-direction
```

## Required repair

Make the current split explicit and preserve the useful missing-record hard
abort within that split. Either:

1. pass `split` to `structural_identity_failures` and skip identities whose
   referenced cell has a different `cell["split"]`; or
2. pass a manifest filtered to the current split.

Then require every structural direction and required full-logit arm for the
selected split. Add a two-split test which calls the FIT validator with only FIT
records, and repeat it with FIT/SELECT identity order reversed so behavior does
not depend on manifest order. Keep the global manifest-derived capture set; it
is correct.

## Rest of the package

The iteration-4 invalid-instrument repair remains closed: exact sorted FIT and
SELECT failure lists are derived from retained evidence; native-attention,
replay/native, incomplete capture, live-factor, hook, and nonfinite integrity
errors fail closed; semantic and computation joins are reconstructed; FIT
scales, score reports, realized 124-cell bootstrap censuses, atomic receipt-last
publication, and conservative recovery remain covered. FIT-first closure is
still 459 forwards and SELECT adds 231 for exactly 690, with zero backwards or
updates and FINAL/OOD absent.

The additional transient full-logit capture is bounded. At float32 vocabulary
width 50,257 it is 694,752,768 bytes for the 3,456 FIT pairs and 347,376,384
bytes for the 1,728 SELECT pairs. Full logits are removed from the retained
vector rows after each structural check. This does not invalidate the prior
under-4-GiB transient-memory bound. At review time the host reported
27,038,855,168 available RAM bytes and 5,878,059,008 free workspace bytes.

## Tests

Exact candidate tests in the detached worktree:

- owner suite: `59 passed`;
- managed adapter: `11 passed`;
- manifest: `14 passed`;
- replacement adversarial: `13 passed`;
- next-wave adversarial: `11 passed, 3 xfailed`;
- first repair review: `4 passed, 3 xfailed`;
- second repair review: `5 passed, 4 xfailed`;
- prior final review: `8 passed, 3 xfailed`;
- iteration-4 review: `4 passed, 1 xfailed`;
- producer and adapter static gates: pass;
- producer and adapter preflight: pass;
- managed no-argument model-free dry run: pass with zero model calls.

New independent test:

`basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_iteration5_review_adversarial.py`

- normal: `5 passed, 1 xfailed`;
- with `--runxfail`: `1 failed, 5 passed`, exactly at the cross-split missing
  direction;
- gate, preflight, and `git diff --check`: pass.

## Reusable lesson

Whenever a validator consumes split-local rows and a global manifest, its API
must carry the split explicitly. A useful invariant test should construct two
registered splits, supply complete rows for exactly one, and check both halves:
other-split rows are not required, while a missing current-split row hard-aborts.
Single-split miniatures cannot establish this phase-local completeness property.

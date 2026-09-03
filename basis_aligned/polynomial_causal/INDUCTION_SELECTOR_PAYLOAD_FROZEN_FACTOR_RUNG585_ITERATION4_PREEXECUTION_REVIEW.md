# R585 iteration-4 independent pre-execution review

Date: 2026-09-03 UTC

## Decision

**BLOCK exact commit `8e1cadec43a6b0203f10aa1a3c15cb494093b6b7`. Do not execute these bytes.**

One execution blocker remains. The replacement amendment preregisters structural
identities as equality of the **full final vocabulary-logit vectors** within
`1e-5`. Iteration 4 instead calls `structural_identity_failures` with
`frozen_insertions=frozen_insertions`, which selects equality of four local
1,152-dimensional inserted terms. The result validator reconstructs the same
local equality from saved `native_e.npy` and `native_u.npy`. This is useful new
evidence, but it is not the preregistered end-to-end sanity check.

This is a real weakening, not a naming issue. The independent planted attack has
identical local inserted vectors but final vocabulary-logit vectors differing by
1 everywhere. The current runtime reports no structural failure; the original
full-logit computation reports a failure of 1.0.

No additional blocker was found in the iteration-4 invalid-instrument repair.

## Outcome blindness and exact reviewed bytes

I reviewed immutable Git blobs, not the moving worktree producer. At
`2026-09-03T22:57:50Z`, existence-only checks found the result JSON, receipt JSON,
and evidence directory all absent. I did not open, stat, or otherwise inspect an
R585 outcome; I did not load a model, call CUDA/GPU, enqueue work, or modify the
board or registry.

| File | SHA-256 |
|---|---|
| producer | `29650364b386269267dc663154c81e8413edfe2abae2ce9b7b93524760692cb4` |
| owner test | `2842f58c54c953885c3b78263ab8bbfd1ddef3b46062fd71fd39d2ea133b289b` |
| dry run | `b17cd142bfe4c5d5516b95a06a177bc15fc1e5452b0e234ad7ad5d0c5ed76c1c` |
| managed adapter | `e98d8c2a6b562fd638690bdabf159d761836c1057114af7c0fb8e7149b7332a2` |
| adapter test | `0c13c70aab4cf3775754d5ce42cf21626394d9ab06655051daa814ad744a2d41` |
| replacement amendment | `98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf` |
| dependency lock | `908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7` |
| prior final review | `8ddbcf3037b890a3fd1ae6933a526a29c1bd767a22d7fa3af8044d7d660d9238` |
| prior final adversarial test | `693b70f70b72334affd2c8da7e5e02e8b5a41125b29e1df7f943a1856a345277` |

## What iteration 4 closes

The new code no longer trusts the saved invalid-instrument lists. It derives them
from retained endpoint, factor, vector, primitive-logit, padding, and structural
evidence and requires exact sorted equality with the terminal failure lists.
Internally complete mutations of the JSONL evidence and summary are rejected.

The following checks are present and were exercised:

1. FIT and SELECT invalid-instrument clauses are derived separately, in stable
   sorted order, with correct terminal precedence.
2. Native-attention reconstruction error and replay-versus-native full-logit
   error are unreconstructible integrity failures and abort rather than becoming
   scientific nulls.
3. Incomplete capture, live factorization failure, hook-write failure, and
   nonfinite values fail closed.
4. Endpoint semantic coordinates, recipient/donor direction metadata, factor
   rows, arm-specific inserted terms, `live + delta`, delta norms/activity,
   endpoint measurements, primitive margins/CE/vocabulary statistics, and
   directed `n/d/q` values remain joined to frozen authority and recomputed.
5. FIT scales, score reports, failure lists, and the realized ordered 124-cell
   bootstrap census per scored split remain reconstructed rather than trusted.
6. Atomic receipt-last publication and conservative recovery remain enforced.
   Recognized partial state is quarantined; complete or arbitrary occupied state
   is refused.
7. FIT-first closure remains 459 forwards. SELECT is reachable only after FIT
   instrument and scientific success and costs 231 more, for exactly 690 total.
   Backward count and weight updates remain zero; FINAL/OOD remain closed.

## The remaining mismatch, computationally

For one site and arm, the local inserted tensor is

$$
t_{a,s}=e_{i(a),s,0}u_{j(a),s,0}+e_{i(a),s,1}u_{j(a),s,1}
\in\mathbb{R}^{1152}.
$$

The iteration-4 saved check compares the appropriate $t_{a,s}$ values for two
arms across four sites. That proves the frozen factor construction says those
local writes should be equal.

The preregistered check instead compares

$$
\max_v\left|\ell^{(a)}_v-\ell^{(b)}_v\right|\le 10^{-5},
$$

where $\ell^{(a)}\in\mathbb{R}^{|V|}$ is the model's final vocabulary-logit
vector after actually applying arm $a$. This additionally checks the hook,
batching, attention replacement, and downstream forward path. Local equality
predicts the full-logit identity but does not test that the implementation
realized it.

The precise runtime change is the `run_science` call:

```python
structural_identity_failures(
    split_records, split_vectors, execution["manifests"], all_replay,
    frozen_insertions=frozen_insertions,
)
```

The optional argument sends execution down the local-vector branch and bypasses
the already implemented `full_logits` branch.

## Required repair

Retain both checks and keep their meanings separate:

1. During the live run, compare final vocabulary-logit vectors for every exact
   structural identity in the frozen manifest. Abort before publication on any
   error above `1e-5`. Do not convert this integrity error to a scientific null.
2. Continue saving and independently reconstructing the local inserted-vector
   identity from `native_e.npy` and `native_u.npy`. Give it a distinct field or
   unambiguous schema name so it cannot be mistaken for the end-to-end check.
3. Add a planted test in which local inserted tensors are equal but final logits
   differ; it must trigger the live hard abort.

The result does not need to save full vocabulary logits merely to reconstruct
this sanity check later. The prospective contract requires the live full-logit
hard abort, while the local equality remains the independently reconstructible
explanation of why the identity ought to hold.

## Tests

Exact candidate suites, run CPU-only from a detached worktree:

- manifest: `14 passed`;
- replacement adversarial: `13 passed`;
- prior final-review adversarial: `8 passed, 3 xfailed`;
- owner suite: `54 passed`;
- adapter suite: `11 passed` after restoring the exact committed dry-run bytes;
- producer and adapter gate/preflight: pass;
- adapter model-free no-argument dry run: pass, zero model forwards/backwards and
  no weight updates.

The historical first-implementation suite is path-bound to old producer bytes;
against iteration 4 it correctly shows its old exact-hash assertion failing and
its repaired strict blockers XPASS. It is not an acceptance suite for these new
bytes. The immutable later review suites were used instead.

New independent test:

`basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_iteration4_review_adversarial.py`

- normal run: `4 passed, 1 xfailed`;
- `--runxfail`: `1 failed, 4 passed`, with the failure exactly
  `runtime_failures == []` versus the required full-logit structural failure;
- static gate: pass;
- preflight: pass;
- `git diff --check`: pass.

## Reusable five-part knowledge packet

1. **Audit pattern.** Freeze exact Git blobs and rebuild identities from the raw
   endpoint/factor/vector records. Expected manifest hashes are insufficient
   unless realized rows and computed summaries are joined back to authority.
2. **Row/arm/site mapping.** Endpoint arrays use frozen endpoint order; live and
   delta arrays use sorted `(directed_id, arm)` order. Each directed row maps its
   recipient and donor endpoint, then each arm chooses recipient/donor score and
   payload indices before crossing the four named sites.
3. **Smallest audited term.** The basic term is the 1,152-vector
   $e_0u_0+e_1u_1$ at one site. An intervention replaces recipient indices with
   donor indices according to score, payload, or joint arm and must satisfy
   `inserted = live_removed + hook_delta`.
4. **Active controls.** Activity, delta norms, neutral-source, neutral-payload,
   filler, lag, structural no-op, target/control typed scales, coverage by at
   least two active control families, and broad-damage statistics all remain
   necessary. Structural no-ops must additionally survive the complete forward.
5. **Failure class and residual scientific risk.** The present blocker is an
   integrity-contract regression, not a scientific null. After repair, the main
   unresolved scientific risk is that even clean selector/payload interchange
   may identify a dataset-specific sufficient factor rather than a stable,
   reusable circuit; that requires held-out/OOD prediction, selective removal,
   and reuse across related induction conditions.

For the next auditor prompt, explicitly require two distinct structural-control
tests: `(a)` independently reconstructible equality of the inserted tensors and
`(b)` live end-to-end equality of final logits, with a planted case that makes
only `(a)` pass. This prevents a stronger runtime sanity check from being
silently replaced by a more auditable but weaker local identity.

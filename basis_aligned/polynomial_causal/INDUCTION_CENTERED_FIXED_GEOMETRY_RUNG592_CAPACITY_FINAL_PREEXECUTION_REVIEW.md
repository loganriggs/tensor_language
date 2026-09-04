# Final independent pre-execution review: R592 phase-relative capacity repair

**Reviewed:** 2026-09-04 02:52 UTC, before any R592 model execution or outcome

**Candidate commit:** `7c6be867fcca7a64b3e6dffbff4540e645a32c4e`

**Capacity amendment commit:** `835401e63604190d9010ecc13e5e9d92e4f89338`

**Prior streaming block:** `a3492a2edec8c5d5d49d6b5cfe48e8bbdfb477bf`

**Review mode:** immutable Git blobs and model-free Python/NumPy tests only

**Verdict:** **APPROVED FOR EXACT-BYTE MANAGED EXECUTION**

This approval is for the exact bytes below. The managed adapter must still observe at least
9,000,000,000 available bytes at dispatch; that runtime condition is not waived by this review.

## Exact reviewed bytes

| artifact | SHA-256 |
|---|---|
| producer | `e625a94216659f4cafb91114b3f253b42844f7e54cb8531b17e0f47614dc5431` |
| model runtime | `09309b1299b85f2c57689913547fef01f2a9e7b538b2768ac62ff3e48e0f039c` |
| managed adapter | `de8b6e2977551dc19cd00449a1de5c698dbc5978c8d9c23d1ad0d21576e025c5` |
| managed-adapter test | `f81b5c3df85c5c7bd8def93136ac2bbbc3d826970c2571c0626dffbad6f1a4e3` |
| owner test | `59764d300fdbe3f2024ee40b32b23fb2bcc56ccd79b48e7b1abbe5c0083eb2fc` |
| fake-runtime test | `52d3d22e7d1eeaaa31bed66a01d28aef296974bff94e96ab7707af6fa4219e85` |
| repaired-implementation test | `dceb2416d20e7e795f8d3d0dd59bac18c123e3ed7705d3660fe6187abfc73844` |
| streaming-storage test | `c927f6828e651589089217fec7a92118563aa893cae1b529651ac7a5a7e77a9e` |
| dry run | `5aa8ee4ce3d4d40d00c74c64d12af7431fdbac090b74c7dabd5ae8ed4cb83e38` |
| phase-relative capacity amendment | `da634dd10da654739d761a6c8f8ce9c1434d8946a7477ba6d9c005c873386458` |

The adapter binds these executable inputs plus the previously approved preregistration, semantic amendments,
independent reviews, R585/R591 authorities, model facade, checkpoint metadata, and shared handoff dependencies. The
producer independently verifies its executable dependency closure before loading the runtime. Static comparison
confirmed the worktree files used by focused tests were byte-identical to commit `7c6be867f`; the independent test
loads the candidate from immutable Git blobs.

## Capacity repair

The original streaming candidate asked for 9 GB both before model construction and again after retaining all FIT
evidence. The amendment correctly leaves the initial gate at 9,000,000,000 bytes and changes only the pre-SELECT gate
to the phase-relative amount still needed after FIT:

$$
2{,}599{,}441{,}920\ \text{SELECT canonical bytes}
+41{,}671{,}168\ \text{largest five-call chunk}
+1{,}160{,}003{,}072\ \text{safety margin}
=3{,}801{,}116{,}160.
$$

This is also exactly the free space left after retaining the FIT arrays on a filesystem that passed at the initial
threshold:

$$
9{,}000{,}000{,}000-5{,}198{,}883{,}840=3{,}801{,}116{,}160.
$$

Both producer and dry run report these constants. The producer uses a strict less-than failure condition, so equality
passes and one byte below either boundary fails. A planted test proves the protected SELECT operation is never called
below the threshold and is called exactly once at equality. The first gate precedes runtime import, model construction,
Torch, checkpoint, and CUDA. The second gate follows completed FIT scoring/evidence and precedes the first SELECT call;
an exception removes the private stage and publishes no scientific or invalid terminal.

At 02:52 UTC, `statvfs` reported 9,378,344,960 available bytes. Thus the initial gate was reachable at review time and
all seven adapter owner tests passed. Free space is mutable, so the adapter must recheck it immediately before managed
dispatch. Earlier in this review an unrelated temporary copy reduced free space to about 5.38 GB; the exact same adapter
correctly failed closed, and all six R592 public namespaces remained absent. The unrelated copy was later removed by
its owner.

## Scientific and evidence invariants

The capacity-only repair does not change the runtime, registered rows, interventions, scoring, thresholds, or claims.
The focused and independent tests re-established:

1. Full-vocabulary evidence uses `VOCAB = 50,304`; no logit slicing was introduced.
2. FIT has 639 calls and SELECT has 322 conditional calls, for 961 maximum model forwards, zero backwards, and zero
   weight updates.
3. Complete canonical evidence is 7,798,325,760 bytes. Adding the largest live five-call chunk gives the exact
   registered data peak of 7,839,996,928 bytes.
4. Calls remain physical width 30. Replay is the literal-zero centered baseline. Score, payload, and joint are frozen
   centered output additions, never live subtraction. The same-layer L8 transaction, component/all-head hybrid
   transport, independent nine-head reconstruction, full-logit identities, finite checks, and FIT-first terminal rules
   remain unchanged.
5. Canonical arrays retain the registered schema and authority order. A slice is copied, flushed, file-fsynced, read
   back, and hashed before its raw call directory is deleted. Partial-prefix invalid evidence binds every completed
   canonical slice and surviving raw call/mask byte. Finalization rejects missing tails, nonfinite files, bad offsets,
   and ledger mismatch.
6. Publication remains atomic in evidence-result-receipt order. Any capacity, missing-observation, hook, identity,
   nonfinite, or incomplete-capture exception before a registered invalid terminal removes the private stage and leaves
   every public R592 namespace absent.
7. The frozen dry run recomputes exactly and opens no SELECT, FINAL, or OOD data.

## Tests and gates

- Focused candidate and new independent suites: **53 passed**.
- Managed-adapter owner suite on the final live capacity state: **7 passed**.
- Broad historical R592 collection: **129 passed, 8 expected failures**, plus three intentionally stale-review
  mismatches (two strict expected-failure tests now unexpectedly pass because the old gate defect was repaired; one old
  exact-worktree-hash assertion expects a superseded producer). These are historical-test assumptions, not failures of
  the exact candidate.
- Authoritative static gate: **PASS** for producer and adapter from the immutable snapshot.
- Advisory preflight: **no findings** for producer and adapter from the immutable snapshot.
- Public R592 result, receipt, evidence, invalid diagnostic, invalid receipt, and invalid evidence namespaces: **absent**.

No model, Torch, checkpoint tensor, CUDA/GPU, queue operation, R592 outcome, SELECT data, FINAL data, or OOD data was
opened by this review.

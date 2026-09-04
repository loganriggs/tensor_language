# R591 independent exact-byte pre-execution review

Date: 2026-09-03 UTC

Verdict: **BLOCKED** for exact commit
`1396747c09a8d46b9b72c8b33926ca96b939a3db`.

This review used the committed Git blobs and performed only static/model-free
checks. It did not load a model, touch CUDA/GPU or a queue, or inspect any R591
diagnostic output. I also did not execute the candidate dry run because its
transitive authority builder parses earlier result artifacts, as detailed
below.

## Exact packet reviewed

| Artifact | SHA-256 |
|---|---|
| producer | `b2b266529f0f842211fea46856064133df5e3f4a8a7758c9095e7d29a94b6c49` |
| producer owner test | `e756ba3d17d3ebee2f81e97e573dd216090555de1fd3f1cfc926268f902d9ce7` |
| dry run | `161193de5d90da69aafcd681e375993fa91d32e99100f0ed02fb586d5a629d8b` |
| amended preregistration | `e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593` |
| builder handoff | `61f8fb407dc026a7a2b126f2dce02b60266d040ffcce7159c5dc6a0d2517cc4f` |
| managed adapter | `5fe0a0d3bb4c149881a1d6d76f5adf7e661df35af39cc37e1cd9893b93cc33cd` |
| adapter owner test | `b20ea468089c90629191f71c6e5f97d4caec180fce64bf0d1ce17f3f9565d7b6` |
| managed-adapter handoff | `3f0041fa05e3f5b309f0b31059d61d2a0c99630fdeb1877fcf19113587cfc49c` |
| equality-support audit | `2ffc5a250cb51cd29bffaea7102326e499f191e1cd12c82962d1f803b7fb3e60` |
| shared v5 handoff | `810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80` |

## Blocking findings

1. **Padding and membership classification does not implement the frozen
   interpretation.** The preregistration defines padding-dominated from
   `N(L_30)-N(L_native)` and membership/GEMM-dominated from
   `N(M_30)-N(L_30)`. The producer instead marks each source active when *any*
   of N, F, or R exceeds the threshold. An R-only padding change or F-only
   membership change is therefore mislabeled as a native padding/membership
   cause. Those F/R contrasts are useful descriptive measurements, but they
   cannot enter the native causal-class booleans. The planted test exhibits
   both failures.
2. **The controlled panel does not satisfy the v5 serialized support receipt.**
   Its FIT length support and exact 256-member hash are checked, and all three
   schedules do have set equality. However, the dry run emits only endpoint
   count, length counts, and an ordered-membership hash. V5 requires the exact
   selected IDs *and* their ordered hash to be emitted and bound to the opened
   split, so an independent audit can reject borrowing, replacement, and
   silent shrink from retained evidence rather than rerunning builder logic.
3. **The managed exact-byte check has a time-of-check/time-of-use gap.** The
   adapter hashes `PRODUCER`, returns from preflight, and later asks Python to
   execute the same mutable pathname. It neither passes immutable checked bytes
   nor has the producer verify an immutable parent-provided digest. A concurrent
   replacement after preflight therefore executes unchecked bytes. The planted
   test swaps the file in precisely that interval.

## Boundary disclosure

The advertised model-free dry run is not outcome-free. `build_dryrun()` calls
`load_authority()`, which imports R585 and calls
`build_execution_authority()`. That function calls R585
`verify_authorities()` with its default `parse_dependency=True`, then parses
the R586 result and receipt and the R587 audit. This may be a legitimate frozen
upstream dependency, but it contradicts the builder/board claim that dry-run
verification accessed no outcome and prevents an outcome-closed reviewer from
running the owner suite. The replacement must either expose a pure authority
builder that validates only immutable semantic inputs, or explicitly relabel
and authorize this transitive access. This review did not follow that path.

## Checks that did hold statically

- The amended panel construction uses only FIT lengths 19, 20, 27, and 28,
  chooses 64 distinct lexical endpoints per length, and creates eight exact
  batches for each controlled schedule.
- The manifest contains 234 batch-32 forwards with dispatcher counts N=132,
  F=24, R=78; the runtime loop order matches the registered full-FIT and panel
  cells.
- F returns the native attention-write object, while R clones it and adds the
  exact R585 `term - canonical` vector at the four registered sites. No
  rank/head-basis substitution was introduced.
- The saved comparison identities use endpoint-key joins and the full 50,304
  logits. Strict-finite checking and the stdout-only/no-scientific-terminal
  boundary are present.
- The adapter pins every listed direct producer dependency and rejects occupied
  conventional R585/R591 result, receipt, and evidence namespaces at preflight.

## Required prospective correction

Use only the N cells for the registered padding and membership cause booleans;
retain F/R versions as descriptive values. Emit the ordered 256 endpoint IDs,
their hash, the FIT split, and per-length support census, then validate those
fields against the authority before execution. Close the adapter race by
executing immutable verified bytes (or an equivalently atomic hash-addressed
artifact) rather than a later pathname lookup. Finally, make the dry-run
authority dependency boundary honest and explicit. A new exact-byte packet
needs a different-agent review; this review cannot approve its own repair.


# Task 21 capability FIT final pre-execution review

**Reviewed:** 2026-09-04 UTC. **Verdict: APPROVE exact commit
`fc3f5c16deb7d3bcb035e3def6fcf53bf75ac9c4` for one hash-bound managed enqueue of its exact adapter.**
This document is a final independent pre-execution review, not an enqueue or execution receipt. It does not authorize
an alternate adapter, direct producer invocation, mutable-path execution, retry, localization, or any later phase.

The review was CPU/model-free and read-only with respect to the repository implementation and task-21 namespace. No
real-mode call was made. Tests of the enabled branch used monkeypatched capture/module/producer objects and temporary
publication namespaces only. No task-21 result/evidence/outcome bytes or checkpoint were opened.

## Exact reviewed chain and hashes

`git rev-parse` resolves the target to `fc3f5c16deb7d3bcb035e3def6fcf53bf75ac9c4`. The following are all ancestors
of that exact commit:

- authority/compiler build `9ebab94615eade27b1eb63e4f2c6239337b71dc9`;
- authority/compiler review `ca088ce0906160958a2586cff50b707699b7eb88`;
- blocked producer/adapter build `000a113eed35c7e8fac0d2ceed126925963cd0d7`; and
- producer review `6b8fe576594bb82a5a2093f2338603040739c9af`.

The authorization successor changes only the board, adapter, adapter tests, checked dryrun, and new authorization
amendment. The producer, compiler, task source, FIT authority, and producer-review document are byte-unchanged from
their approved commits. The target's changed files have these recomputed raw SHA-256 digests:

| File | SHA-256 |
|---|---|
| authorization-enabled adapter | `43564464637c7c0fa7a609ec55bc05377c1d872ad0d0cdf1ef80e957e5026779` |
| authorization amendment | `a31cf24ec79d86f084c29bdc18a909e1ff0457b4e0921fd6249f722adf2b08d1` |
| adapter tests | `6e8db645bc515de5010ede312027c489dc87bbc2841c8ce4bad1d836b7c86b13` |
| checked model-free dryrun | `4ee59eb5313e337e6343ee1e77bed0df5829cff0bc1a894d80c9d66814c7f309` |
| target board blob | `13d3c9ca230729146bc6ff94b02c90a7f55d6eb2e0f64a60ea6604e87fa44de1` |

The amendment and adapter bind the complete reviewed chain:

| Bound object | SHA-256 or exact identity |
|---|---|
| unchanged producer | `395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf` |
| producer-review document | `8763602a753345a19312613160d32b3ffe537a7ebfcb4bcf4c83905a25b7ed29` |
| producer implementation preregistration | `3009aff99543e34e8a7d33a486035e5168c136f168c18ebb8e3fd8a3ad290882` |
| compiler-review document | `3f66075ab775ce27084203999859ea6941efec6d2154a6987994b48e011c7c50` |
| capability preregistration | `da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3` |
| task source | `bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2` |
| full logical authority | `191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b` |
| FIT authority file | `69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94` |
| FIT record digest | `c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8` |
| compiler source | `43ff54a930338127670f9291bb7bac66e914a11cdd04e919f222a5a13bb89390` |
| compiled contract | `5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2` |
| call manifest | `ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e` |
| metric manifest | `e8cab6e2fb8000bd144f92182abd71c7774d3afcd2dc1b1de50f9c1a9ec79faf` |

All nineteen adapter closure roles rehashed exactly. In addition to the task-specific rows above, those include result
contract `af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272`, experiment spec
`64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c`, artifact package
`6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc`, battery contract
`b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e`, managed entry
`1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81`, empty `jacclust` package initializer
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, model source
`49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2`, observed-model facade
`b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c`, canary-1 source
`3316a60e18d518f4c619d69b95ec4db34e1c72ad159f6bc4842405231b6a84f8`, and canary-2 source
`cc092508a9d7eee357cbe87d10c226357fcc3257ca6c456efa4a8054b4bf5a23`.

## Authorization binding cannot be bypassed in the exact adapter

`EXECUTION_AUTHORIZED=True` does not skip managed capture. Real dispatch calls `capture(None)`, which first safely
bootstraps the hash-bound experiment and managed-entry modules, constructs the exact execution spec, and captures all
nineteen roles. Real capture's expected-role equality requires every role, including `producer_review` and
`authorization_amendment`. `validate_captured_bytes()` independently requires both authorization roles to be present
and rehashes every captured payload; `load_verified_closure()` validates them again before loading the producer.

Thus deletion, substitution, or mutation of either amendment/review fails before the producer is loaded. Coordinately
changing a role digest in the adapter changes the adapter's externally bound SHA-256. The exact adapter itself is
therefore the root object that the already-reviewed hash-bound managed queue must capture and verify before execution.
The checked dryrun records that same adapter digest and both authorization roles, but is not treated as the external
root of authority.

A synthetic real-mode test monkeypatched `capture`, `load_verified_closure`, and `producer.run_science`; it observed
exactly one ordered `capture(None) -> load(real=True) -> run_science(captured)` chain. It made no model import, CUDA
call, checkpoint read, model forward, or publication. Invalid managed modes and unmanaged command-line arguments
reject.

## Exact scientific and publication scope

The producer remains byte-identical to the independently approved build. It recompiles only the exact captured FIT
authority and revalidates the whole compiled-contract hash. The physical schedule remains four base calls followed by
four donor calls, every call exactly `21 x 8`, for 8 forwards and 168 explicit row-side evaluations. Each row-side
retains only one answer logit and one maximum registered-foil logit, as contiguous finite `float32[21]` arrays:
`168 * 2 * 4 = 1,344` raw numeric bytes and 24 evidence files including call metadata.

Targets are the registered jointly tokenized side-specific one-token answers. Foils are exactly the nonempty token-ID
set occurring in the linked base/donor rows with that side's target removed. The evaluator gathers target logits and
maximum foil logits only from the final sequence position after the native soft-capped forward. Full logits,
activations, hidden states, gradients, backward calls, parameter updates, component labels, and localization fields
are absent.

Only FIT authority is present. SELECT, TEST, and OOD are forbidden, not generated, not captured, and not opened. A
capability pass emits only the frozen aggregate projection. Its exact logical complement is `hard_abort`, and every
scientific projection field is null. Neither terminal selects a reader, identifies a circuit, opens a localization
namespace, or licenses a later phase.

At review time, the producer's absent-namespace guard passed without opening task-21 result/evidence bytes. The final
paths are checked by `lstat`, so files, directories, and dangling symlinks are occupied. Staged evidence, result, and
receipt are installed in that order with Linux `renameat2(RENAME_NOREPLACE)`. Late races never overwrite; rollback
acts only on inode-identical entries installed by the invocation; receipt-last makes incomplete publication
non-scientific. Namespace, runtime, canary, checkpoint, call, array, or price failure does not authorize an automatic
retry.

The dormant real producer retains exact runtime versions; model revision
`ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`; config SHA-256
`428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`; weight SHA-256
`680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`; weight size
`2,067,738,635`; exact model/facade/canary source hashes; CUDA availability and one-device float32 placement; finite
`21 x 8 x 50,304` native logits; and the live canary predicates plus canary-2 composition/fingerprint
`6b22b221a811382775e6a64b4198a61f2f9bcc55b826d0d12d0512d1a28be99c`. These gates were audited statically and
with synthetic fixtures only; no live checkpoint, model, GPU, or canary-result bytes were accessed in this review.

## Mutations and tests

Temporary/in-memory mutations had these outcomes:

```text
authorization_amendment mutation: rejected
producer_review mutation: rejected
producer mutation: rejected
adapter mutation versus externally bound SHA: rejected
call/metric manifest mutation: rejected before evaluator
missing amendment or review: rejected
```

Checked-in attacks additionally reject changed compiler/task/FIT bytes, import-cache and disk substitutions, unsafe
source paths, malformed arrays, nonfinite evidence, incomplete/duplicate primitives, nested localization result keys,
non-null hard-abort projections, occupied and dangling namespaces, every late publication race, every injected crash
point, and external inode substitution during rollback. The bound producer review separately records rejection of
post-publication result tampering through receipt binding; the relevant package code and producer are unchanged here.

From `basis_aligned/bilinear_quotient`, with `PYTHONPATH=../..:.:ops`, `BQLIB_NO_MODEL=1`, and empty
`CUDA_VISIBLE_DEVICES`:

```text
focused producer + authorized-adapter suite: 40 passed in 2.29s
relevant task21/compiler/spec/integration/package/result suite: 151 passed in 6.42s
selected authorization/mutation/monkeypatch checks: 9 passed in 1.33s
```

All relevant tests pass. The three experiment-index tests used in the builder's wider accounting were not rerun
because one regenerates a tracked index, contrary to this review's read-only constraint; that index is not in the
adapter closure. The inherited task-17 namespace-only failures documented in the producer review remain unrelated and
were not needed to adjudicate this authorization successor.

## Final verdict

**APPROVE exact adapter SHA-256 `43564464637c7c0fa7a609ec55bc05377c1d872ad0d0cdf1ef80e957e5026779`
from commit `fc3f5c16deb7d3bcb035e3def6fcf53bf75ac9c4` for exactly one hash-bound managed enqueue under amendment
`a31cf24ec79d86f084c29bdc18a909e1ff0457b4e0921fd6249f722adf2b08d1`.** The enqueue record must capture that
adapter digest, and the trusted runner must verify and execute those captured bytes rather than reopen a mutable path.
The normal namespace/runtime/canary/checkpoint gates remain execution-time dependencies. Any preflight mismatch is a
stop, not permission to repair or retry. This review itself performs no enqueue and no execution.

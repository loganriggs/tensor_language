# Task 21 capability FIT producer and blocked-adapter review

**Reviewed:** 2026-09-04 UTC. **Verdict: APPROVE exact build
`000a113eed35c7e8fac0d2ceed126925963cd0d7` only as the input to a later, separate prospective authorization.**
This review does not authorize execution, changing `EXECUTION_AUTHORIZED`, model/checkpoint access, GPU use, queue or
enqueue activity, publication, localization, or any later phase.

The review was CPU/model-free. It did not open a task-21 result or evidence artifact, inspect a checkpoint, or enter
the real adapter branch. Synthetic execution tests used monkeypatched objects and temporary namespaces only.

## Immutable object and complete hash census

`git rev-parse` resolves the requested object to exactly
`000a113eed35c7e8fac0d2ceed126925963cd0d7`. Both the approved compiler commit
`9ebab94615eade27b1eb63e4f2c6239337b71dc9` and its independent review commit
`ca088ce0906160958a2586cff50b707699b7eb88` are ancestors. The six task-21 files introduced by the reviewed commit
are byte-identical in the tested tree and have these recomputed raw SHA-256 digests:

| Reviewed file | SHA-256 |
|---|---|
| producer | `395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf` |
| blocked adapter | `f7721d1b484ec7a9891dc72fc22618d403330c65092ebbb5d6d8fac68b31eced` |
| producer tests | `18257c00300ba1df6f67a0e277dea0e600ea25c47ef7ff218b2f4a77a937a6ac` |
| adapter tests | `da9ce1d7023e399ca5b0eb665afc480c0dbfdc4185191be518e7e00146b63b62` |
| implementation preregistration | `3009aff99543e34e8a7d33a486035e5168c136f168c18ebb8e3fd8a3ad290882` |
| checked-in producer dryrun | `58c3821a8812062fd8fd5b0cd4dcb8aff7166dfbbe76ba10d85138e1dfa96bd6` |

The target commit's board blob has raw SHA-256
`263ab02b8c3043b8e92eb7c3cf1aef7d9017f6c0f96ef1b8d6ebbf98c06f7aad`.

Every source, authority, preregistration, and review role frozen by the adapter was independently rehashed. Expected
and observed digests were identical:

| Frozen role | SHA-256 | Dryrun |
|---|---|---:|
| result contract | `af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272` | yes |
| experiment spec | `64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c` | yes |
| artifact package | `6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc` | yes |
| battery integration contract | `b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e` | yes |
| managed entry | `1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81` | yes |
| task-21 authority adapter | `bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2` | yes |
| capability compiler | `43ff54a930338127670f9291bb7bac66e914a11cdd04e919f222a5a13bb89390` | yes |
| producer | `395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf` | yes |
| capability preregistration | `da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3` | yes |
| producer implementation preregistration | `3009aff99543e34e8a7d33a486035e5168c136f168c18ebb8e3fd8a3ad290882` | yes |
| compiler review | `3f66075ab775ce27084203999859ea6941efec6d2154a6987994b48e011c7c50` | yes |
| FIT authority file | `69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94` | yes |
| empty `jacclust` package initializer | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | no |
| model source | `49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2` | no |
| observed-model facade | `b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c` | no |
| canary-1 source | `3316a60e18d518f4c619d69b95ec4db34e1c72ad159f6bc4842405231b6a84f8` | no |
| canary-2 source | `cc092508a9d7eee357cbe87d10c226357fcc3257ca6c456efa4a8054b4bf5a23` | no |

This reproduces the approved compiler bindings: full task authority
`191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b`, FIT records
`c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8`, call manifest
`ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e`, metric manifest
`e8cab6e2fb8000bd144f92182abd71c7774d3afcd2dc1b1de50f9c1a9ec79faf`, and compiled contract
`5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2`.

## Physical calls and candidate/logit semantics

The producer accepts only the compiler reconstructed from the captured FIT authority and rechecks the complete
compiled-contract digest before use. It then walks the exact four base calls followed by four donor calls without
deduplicating repeated prompts. Every call is `21 x 8`; there are exactly 168 explicit row-side evaluations.

I independently reconstructed all 168 inputs. Every integer sequence equals the registered side-specific prompt IDs
in manifest order. Every target equals that side's registered jointly tokenized one-token answer. Every foil list is
exactly the sorted token-ID set occurring in the linked base/donor sequences with the target removed; all are nonempty
(two foils for C, three for A1/A2/P), and no target appears among its foils.

The dormant evaluator applies the native bilin18 recurrence—embedding, initial RMS normalization, all blocks with
`first_value` and `x0`, final RMS normalization, unembedding, and the required native soft cap
`30*tanh(logits/30)`. It requires finite logits of exact shape `21 x 8 x 50,304`, reads only the final sequence
position, gathers the registered target, and takes the maximum only over the registered foil IDs. It returns two
finite C-contiguous `float32[21]` arrays. Eight call JSON files plus sixteen arrays give 24 evidence files and exactly
`8 * 2 * 21 * 4 = 1,344` raw numeric bytes. Full logits, states, activations, gradients, backward calls, updates, and
localization labels are absent.

## Decision, runtime, and closure

The frozen capability compiler revalidates the call, metric, price, and whole-contract hashes before interpreting
primitive values. A synthetic passing fixture reaches `ok`. Its exact logical complement reaches `hard_abort`; all
seven scientific projection fields are null, and the projector is not called after a failed hard-abort predicate.
Extra primitive fields, missing/duplicate coverage, malformed or nonfinite arrays, changed row order, and nested
reader/writer/component/activation/localization/selection result keys reject.

Dryrun captures exactly twelve non-runtime roles and exactly one authority role, `fit_authority`. It excludes the five
runtime-only roles (`jacclust_package`, model source, observed-model facade, and both canary sources), imports no Torch,
makes zero model calls, and opens no later phase or outcome. The checked-in dryrun has SHA-256
`58c3821a8812062fd8fd5b0cd4dcb8aff7166dfbbe76ba10d85138e1dfa96bd6` and equals a fresh model-free adapter report.

The dormant real closure freezes CPython/NumPy/Torch/CUDA/tiktoken/einops versions, model revision
`ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`, config SHA-256
`428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`, weights SHA-256
`680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`, weight size
`2,067,738,635`, model/facade sources, expected CUDA availability and float32 placement, and the canary-2 composition
and fingerprint `6b22b221a811382775e6a64b4198a61f2f9bcc55b826d0d12d0512d1a28be99c`. No live
checkpoint or canary result was opened in this review. If separately authorized later, the producer checks namespace,
runtime, live canary results, captured FIT compilation, verified facade constants, and checkpoint bytes before any
scientific interpretation.

Captured Python sources replace both import-cache and earlier-on-path substitutions and are loaded in explicit
dependency order. The producer's compiler, framework, and package identities are asserted to be those exact captured
module objects. Safe reads use `O_NOFOLLOW`, require regular files, bind descriptor identity before/after capture, and
check SHA-256.

## Authorization ordering and publication

The adapter accepts only absent `BQLIB_DRYRUN` or literal `1`. With real mode requested, `dispatch()` checks
`EXECUTION_AUTHORIZED=False` and raises before calling `capture()`. Since `capture()` is the only path to `bootstrap()`
and `safe_read()`, the blocked real branch cannot bootstrap, capture files, preload runtime modules, call the producer,
touch CUDA, or open a model/checkpoint. A planted test replaced `bootstrap`, `capture`, and `safe_read` with forbidden
functions; none ran.

The final namespace is new and task-specific. `lstat` treats regular files, directories, valid symlinks, and dangling
symlinks as occupied. Publication validates a staged package, then installs evidence, result, and receipt in that
order with Linux `renameat2(RENAME_NOREPLACE)`. Every late destination race fails without replacement; earlier owned
moves roll back. Rollback moves only an inode whose identity still matches this invocation and refuses an externally
substituted entry. The receipt is last, so a partial hard crash cannot validate as complete. A complete `hard_abort`
package is valid and still has every projection null.

The adapter's self-hash in the dryrun is informational, as it must be: a file cannot recursively establish its own
authority. I mutated the adapter bytes in a temporary file and confirmed that comparison against the exact expected
digest `f7721d1b484ec7a9891dc72fc22618d403330c65092ebbb5d6d8fac68b31eced` rejects. Therefore the later authorization
and managed enqueue record must externally bind that exact adapter digest (or the exact digest of a prospectively
reviewed authorized amendment) before execution.

## Mutation and test evidence

Explicit temporary attacks produced this result:

```text
adapter_hash_mutation=rejected
producer_mutation=rejected
manifest_mutation=rejected_before_evaluator
published_result_mutation=rejected_by_receipt_binding
```

The result mutation appended bytes after create-only publication; complete-package validation rejected its mismatch
with the receipt. Additional checked-in attacks reject changed compiler/producer/FIT bytes, symlink and nonregular
sources, import-cache and disk substitution, call/metric row-order changes, malformed arrays, undeclared nested result
surfaces, occupied/dangling namespaces, all three late-race points, all three crash points, and an externally replaced
rollback inode.

From `basis_aligned/bilinear_quotient`, with `PYTHONPATH=../..:.:ops`, `BQLIB_NO_MODEL=1`, and an empty
`CUDA_VISIBLE_DEVICES`:

```text
producer + blocked-adapter tests: 38 passed in 2.19s
task21 + compiler/spec/integration/package/result boundary suite: 149 passed in 6.65s
selected explicit mutation checks: 6 passed in 1.30s
```

The builder's 152-test statement additionally included three experiment-index tests. I did not rerun that group
because its final test regenerates a tracked JSON index, contrary to this review's read-only constraint; the index is
not in the adapter closure. The 149-test read-only subset includes all producer, adapter, authority, compiler,
integration, package, and result-contract boundaries material to this verdict.

For inherited-failure accounting, I added the task-17 suites to the same broad command. The result was 202 passed and
5 failed. All five failures arise from one unrelated condition: task 17's pre-execution dryrun tests call its
`require_unused_namespaces()`, which correctly finds the already-published task-17 result/receipt/evidence namespace
occupied. They occur only in two task-17 test modules, before any task-21 code. The reviewed build changes no task-17
file, and the task-21-focused and relevant broad suites remain fully passing. Published task-17 artifacts must not be
moved merely to make old pre-execution tests green.

## Verdict and remaining dependency

The exact build is correctly bound to the approved authority/compiler, preserves the 21-by-8 native metric and exact
price, is model-free in dryrun, fails closed under the tested mutations, publishes create-only with receipt last, and
keeps its real path blocked before the first managed read.

**APPROVE `000a113eed35c7e8fac0d2ceed126925963cd0d7` for a later separate authorization process.** Before any execution,
a prospective amendment must bind this review digest, externally hash-bind the authorized adapter and producer,
preserve the exact runtime/canary/checkpoint closure and unused task-21 namespace, and receive another independent
ordinary review. Only that final reviewed adapter may enter the repository's hash-bound managed queue. This review
does not flip the authorization flag and does not authorize or enqueue anything.

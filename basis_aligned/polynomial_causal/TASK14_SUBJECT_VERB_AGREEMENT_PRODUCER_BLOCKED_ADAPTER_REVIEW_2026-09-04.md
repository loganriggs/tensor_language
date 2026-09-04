# Task 14 subject–verb agreement producer and blocked-adapter review

**Reviewed:** 2026-09-04 UTC. **Verdict: APPROVE exact build
`26d45e89797515240eec368bc313728925d5f48a` only for construction of a later, separate prospective authorization
successor.** This verdict does not authorize execution, changing the authorization state, model/checkpoint/GPU
access, queue/enqueue activity, publication, localization, or any later phase.

This was an independent CPU-only, model-free review from immutable Git objects. I did not enter real adapter mode,
read a checkpoint or task-14 outcome, import the real model closure, touch a GPU or queue, or edit the producer,
adapter, preregistration, authority, tests, or dryrun. Synthetic runtime tests used fake modules and temporary
namespaces only. My sole repository writes are this review and append-only board receipts.

## Immutable target, ancestry, and file hashes

`git rev-parse` resolves the requested object exactly to
`26d45e89797515240eec368bc313728925d5f48a`. Compiler commit
`fc586c1158ddeee7df8f4b502deec54189609c4c` and its independent review commit
`10afc5d6005d169879b07e92cb5fcb4e3a65f312` are both strict ancestors. The target's direct parent has no task-14
producer, adapter, implementation preregistration, or producer dryrun. No strict task-14 outcome/result/evidence
namespace exists in the target.

All six reviewed files are byte-identical between the tested tree and exact target objects. Raw SHA-256 digests
recomputed with `git show 26d45e...:<path>` are:

| Reviewed file | SHA-256 |
|---|---|
| producer | `9ba9448fcebcd764aa2b91e91333b3bbb2549a899b1f8304f2ce3f83bf741e3e` |
| execution-blocked adapter | `7c0ef18db572dede3a65a355860efbc8d15787e7486c10f48e2643c0aa6f4f38` |
| producer tests | `f770d78bc60c9b62e1c43239638a0f9baeb1f38246d57a371da0b0a2d161d9f5` |
| adapter tests | `775a5432655b4d250232ab55d0a1d0fd17e98febe50e6dd122132bd1c36df57e` |
| implementation preregistration | `d84d345c8d2b4183979cd09a57d60c87fccc5a36f03bddf0fa9316f07779a6f3` |
| checked-in producer dryrun | `ae7652a7e297301f048dd277525eac70020bae21d192c7072ac2c4b5058ede84` |

The target board blob is `c6c246351102fa9629e6f9d3291849c2a53e0f34dfd1ce7fcc67e94af1b0d3d7`.

The adapter's frozen upstream chain also matches the exact target bytes:

| Frozen role | SHA-256 |
|---|---|
| result contract | `af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272` |
| experiment spec | `64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c` |
| artifact package | `6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc` |
| battery integration contract | `b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e` |
| managed entry | `1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81` |
| repaired task-14 generator | `33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94` |
| capability compiler | `98b2d263c5120c1a7b700dc4bb451f65cc9f9b338740d2cfbc7ae25a3ba5aab1` |
| capability preregistration | `06a9747b4707999e11637a45cf83588bfd9cb8671d6b3a25790518af62900f8b` |
| compiler review | `a1707dd88949a9b5beb439b275e665cda1a7a62a6d5eedf076d20d192c852e59` |
| FIT authority file | `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f` |

The adapter also binds the producer and implementation preregistration digests above. Any changed captured role is
rejected before module loading.

## Exact calls, primitive bindings, and price

The producer compiles only from the captured FIT authority and rechecks the entire compiled-contract digest
`84f8e1cf85323dba94d13c7c716afef448b8621bff6b534c2025715420e86a82`. The frozen call and metric manifests remain
`4b4da44c5090914f87d52e018bc9a8d18b74a202bdb82667283a9f1564682e0e` and
`5da9f66829156e352afe087c75f92a7a6a37f06fe1ec5177efeffd9442609dcc`.

The literal order is base A1, A2, P, C, followed by donor A1, A2, P, C. Every call contains 32 distinct authority
rows. Sequence lengths are `[5, 8, 5, 8]` on both sides. I independently joined all call row IDs to the frozen
authority and reconstructed all eight token matrices, target IDs, opposite-copula foil IDs, final prediction
positions, `incongruent` flags, and `answer_changes` flags. Every binding matches. The producer creates exactly 256
primitive rows, each unique by `(row_id, side)`, with no undeclared field.

Each evaluator response must contain exactly two finite C-contiguous `float32[32]` arrays: `answer_logit` and
`foil_logit`. Shape, dtype, stride, NaN, and Infinity mutations all reject before scoring. Saving the 16 arrays plus
eight exact call JSON files yields 24 evidence files and exactly `8 * 2 * 32 * 4 = 2,048` raw numeric bytes. Full
logits, hidden states, activations, gradients, backwards, updates, and localization values are absent.

The dormant evaluator reproduces the native bilin18 path: embeddings, input RMS normalization, all 18 native blocks
with shared first-value state and initial residual, final RMS normalization, unembedding, and the native
`30*tanh(logits/30)` soft cap. It reads only the final position and gathers the two registered token logits.

## Decision and phase isolation

The dryrun exercises three distinct fixtures. A passing fixture reaches `ok`. A valid scientific miss reaches
`hard_abort` with `metric_evidence_contract=true`, `answer_relation_contract=true`, and
`native_capability_gate=false`. A malformed typed scalar reaches the earlier instrument hard abort with
`metric_evidence_contract=false`. Both aborts set every scientific projection field to null, but their predicate
records remain distinguishable; nonfinite JSON/arrays reject still earlier.

Only the 128-row FIT authority is declared. Dryrun captures twelve non-runtime roles and exactly one authority role,
`fit_authority`. The sole phase is FIT; SELECT, TEST, and OOD are forbidden, no future authority or outcome role is
present, and `compile_from_captured` rejects a planted future-phase role. Result-surface traversal rejects nested
reader/writer/component/attention/MLP/activation/localization/selection keys. The capability decision has no
component choice, localization namespace, or phase opener.

The checked-in report equals a fresh adapter dryrun and has logical SHA-256
`43558a82bfb328b44352d0c01ad0a9ed952372d4534565223021d46ba97549d9`. Its serialized stdout hash is identical under
Python hash seeds 0, 1, and 999. It records zero model loads, GPU access, forwards, backwards, updates, publication,
and queue activity.

## Runtime, checkpoint, canary, and import topology

The future-only runtime contract pins CPython 3.12.14, NumPy 2.5.2, Torch 2.11.0+cu128, CUDA 12.8, tiktoken 0.14.0,
and einops 0.8.2. Synthetic modules establish that an exact runtime passes while changed version or absent CUDA
fails before checkpoint loading.

The closure pins model revision `ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`, config SHA-256
`428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`, weights SHA-256
`680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`, weight size `2,067,738,635`, tokenizer
vocabulary 50,257, and logit vocabulary 50,304. The producer requests a full weights SHA before loading and again
after all eight calls, requires the receipts to be identical, validates topology before and after the one float32
CUDA move, and compares canary summaries before and after the calls. I inspected these source contracts but opened no
checkpoint or live canary receipt.

Runtime-only exact source hashes are:

| Role | SHA-256 |
|---|---|
| receipt helper | `ced8065d262d3ae8b1ac958424848ccf75d4264174b0f4b1b3144d0f4be99708` |
| empty `jacclust` initializer | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `jacclust.tt_model` | `49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2` |
| observed-model facade | `b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c` |
| `mlp_in_situ_usage_rank_map_probe` fastload dependency | `c701af71491d29f33f5ad691f89380a9fa7c2d86514a61fd7423ad8a78fd4d16` |
| `fastload.py` | `5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90` |
| canary-1 source | `3316a60e18d518f4c619d69b95ec4db34e1c72ad159f6bc4842405231b6a84f8` |
| canary-2 source | `cc092508a9d7eee357cbe87d10c226357fcc3257ca6c456efa4a8054b4bf5a23` |

Their declared real load order is dependency-correct: receipt, `jacclust` package, `jacclust.tt_model`, facade,
MLP-in-situ dependency, then fastload. The facade imports the preloaded `jacclust.tt_model` and enforces exact
`type(model) is TT.GPT`; fastload imports the preloaded MLP dependency and `jacclust.tt_model`. Producer imports of
facade and fastload are therefore resolvable to captured module names, not disk search, once the successor enables
the real closure.

This exact build deliberately cannot execute that dynamic identity check: `load_verified_closure(..., real=True)`
raises. The authorization successor must capture all runtime-only source bytes, load them in `REAL_LOAD_ORDER`, and
assert facade-to-`tt_model`, fastload-to-MLP-dependency/`tt_model`, and producer-to-facade/fastload object identities
before calling `run_science`. This is an explicit successor requirement, not a reason to weaken the current real-mode
block.

## Authorization ordering and substitution resistance

The adapter accepts only absent `BQLIB_DRYRUN` or literal `1`. When real mode is requested, it raises unconditionally
before `bootstrap()`, `safe_read()`, artifact capture, module load, producer import, or runtime access. Monkeypatching
`EXECUTION_AUTHORIZED=True` does not bypass the block. The blocked closure rejects planted authorization/review roles
and cannot accidentally advertise itself as authorized.

Dryrun source capture uses `O_NOFOLLOW`, regular-file checks, descriptor identity before/after read, and exact SHA.
Captured modules replace earlier `sys.modules` poison and on-disk/PYTHONPATH substitutions. Producer imports of the
compiler, package, and framework are asserted to be the exact captured module objects. Adapter self-hash in the
dryrun is informational only; the future authorization successor and managed enqueue must externally bind either
this exact adapter SHA or the separately reviewed successor SHA.

## Create-only publication

The reserved namespace is `circuit_battery_task14_capability_fit_v1`. `lstat` counts regular entries, directories,
valid symlinks, and dangling symlinks as occupied. A new run checks result, receipt, and evidence freshness before
runtime access, and a completed package makes every later invocation fail that guard.

Publication validates a fully staged package, then installs evidence, result, and receipt in that order using Linux
`renameat2(RENAME_NOREPLACE)`. There is no weaker rename fallback and no automatic retry loop. Late races at each
destination do not overwrite the racer; earlier installed entries roll back. Rollback moves only an inode whose
device/inode/mode/size identity still equals the invocation-owned source and refuses to remove a substituted entry.
An independent hostile-inode test confirmed the attacker entry remains untouched. Receipt is last, so an incomplete
publication cannot validate as complete. Mutating a completed result breaks its receipt hash and is rejected.

## Test and mutation evidence

With bytecode/pytest caches disabled, `BQLIB_NO_MODEL=1`, and empty `CUDA_VISIBLE_DEVICES`:

```text
producer + blocked-adapter focused tests: 47 passed in 2.12s
task14 authority/compiler/producer/adapter + framework boundary suite: 173 passed in 9.64s
independent call/closure/runtime/publication mutation assertions: 78/78 passed
explicit adapter-hash and completed-package mutation assertions: 4/4 passed
```

Independent attacks covered FIT/source/compiler/review/prereg/producer mutation, future-role planting, coherent call
and metric-label changes, array shape/dtype/stride/nonfinite values, forbidden nested result fields, authorization
role planting, `sys.modules` substitution, runtime version/CUDA changes through fake modules, canary fingerprint
change through synthetic files, dangling symlink, late race, staged crash, externally replaced rollback inode,
adapter bytes, and published-result bytes. No attack reached a scientific projection or overwrote an existing entry.

## Verdict and remaining dependency

**APPROVE exact producer SHA-256
`9ba9448fcebcd764aa2b91e91333b3bbb2549a899b1f8304f2ce3f83bf741e3e` and blocked-adapter SHA-256
`7c0ef18db572dede3a65a355860efbc8d15787e7486c10f48e2643c0aa6f4f38` from build
`26d45e89797515240eec368bc313728925d5f48a` only as inputs to a later prospective authorization successor.**

Before execution, that successor must freeze this review digest and the exact build hashes, add a prospective
authorization amendment, enable and object-identity-check the captured real closure described above, preserve all
runtime/canary/checkpoint/FIT/price/publication gates, and receive a final independent review. Only the final reviewed
authorized adapter may be submitted through the managed hash-bound enqueue path. This review does not authorize or
enqueue anything.

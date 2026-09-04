# Independent review: task-14 FIT-localization-v2 physical compiler v2

**Review completed:** 2026-09-04 12:25 UTC

**Reviewer:** Codex, independent of the candidate builder

**Candidate:** exact immutable commit
`6b7fb09ff30080e73cad0414d8315db660e04ca0`

**Verdict:** **BLOCK**. These exact bytes do not license producer construction.

This was a CPU-only, exact-Git-object review. I did not build a producer, import a
model, open a checkpoint, use CUDA/GPU, inspect an activation or task-14
localization outcome, access a run namespace, or touch a queue/runner. The block
is prospective and leaves the logical v2 authority unchanged.

## Immutable review basis

The candidate has parent `b73919e97e2f0d9bfae0d416c8a021e635db1eb4`
and tree `610803a25c3805f37458b26ec71b8c095635e9d4`. I verified that all
of the following are strict ancestors of it:

- logical authority `8f41f51cdf7e073063201cc48760622607ce91b9`;
- independent authority approval
  `2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923`;
- rejected compiler `ea16e22d28d125274ca4353f46e434c2826e0b02`
  and its block review
  `45db7e2f2e2df3627c594b7df67dc0173aae318b`;
- producer acceptance
  `ecb37c0abeb1869e4122eed29179a268712bda3b`, first addendum
  `ea50dcfdfe0282ca3cbdc39df03f32400d06bc9d`, and controlling
  second addendum `3b3920ac8a43155f29ab87e9e191dad78f9378a6`.

The frozen logical preregistration is SHA-256
`3ea31387f611d0d095895dec6ed0859e1d99b2ad91a5d5adfb7be178bf127f59`;
the authority review is
`2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384`;
the prior block review is
`673389c02ec4d7e9122557fe4fb44ab9f90950ccf8e6efbbd310ac6d543548b1`.
The prospective physical-compiler rubric at commit
`ef18d5ada4b3c5cf1692310cce9384ddc3f17b47` is SHA-256
`c85704d9124fa57e920558672110e72ec6afcf0b6faabe34e0501f086180f0b9`.
The three producer-acceptance documents are respectively SHA-256
`1724fa6de7ece875cd633976841159302e04033ca008af6e6437ee159a935b46`,
`c28e6dc2a453a08027673a2420bbf2053e94a0cb02b18a6f0579f747c81a4d96`,
and `30f11f8a6c4efd8e9dc6e3eb97cbb79bfbdbed21f0d27a622d268002824be18b`.

Exact candidate objects independently hashed as follows:

| role | SHA-256 | Git blob |
|---|---|---|
| compiler | `6024009bc045200bc3525765dc1dd66261f84f9ccee0dbf9da7b2ddff3101415` | `d37d155580ff6ade5d1af1590d25c5ca4e65aeb2` |
| manifest | `5f870a292e9e2db0830156d09d17af10d6d2c8201cb134c80aee12d9261f1b2e` | `921ebb6c635c46ffa54ae39b4a6e838d47a05844` |
| binary call index | `ae399e393d03af9b6232b7fc5339dd892b418ec7c88943735f8b72fc064c8ad9` | `c09a37097ada822dc992b23323beb17e6c3c4684` |
| dry run | `6cae7b207e372d82c061d189c67bff05bd4772da366d41ebf03a5cdd0c58c0dd` | `2adb71ff33d703544726749a75e56e6f9b224980` |
| focused tests | `462610026c34311f1b7a3074446bf9cc3b4c80ee96f89874523e4047b1aded46` | `2cdf5071958a2137346da8599dbf14608d1bb3cc` |
| compiler preregistration | `c79911049d401f169c63876f197c1287c39caf0730ea2b885e72627d79cf5d8a` | `1405397bb802e6b2c437f922ad3346f887954742` |

Every one of the 19 files in `artifact_closure` was read from the candidate
tree and independently rehashed; all 19 matched its declared SHA-256. This
includes the exact FIT authority (`e88fd860...`), partition (`1f43b767...`),
donors (`ff702f29...`), authority builder/tests/review/preregistration, old
block review, three acceptance documents, contracts, facade/model source, and
future runtime dependencies.

## What does pass

The candidate repairs the previously reported static coherent-mutation holes.
`validate_manifest` now reconstructs every static contract from canonical
constructors and freezes both the chunk census/root and index census/hash. My
own changes to the ordinary recovery rule, selected-H validation gate,
higher-rank threshold, terminal precedence, initialization seed, logical batch
size, boundary-17 semantics, and maximum-forward price were all rejected after
recomputing the outer contract hash. A coherent whole-chunk deletion with
offsets, prices, roots, index metadata, and contract hash recomputed was also
rejected.

I independently reconstructed the materialized index rather than trusting its
summary:

- 3,821 unique conditional chunks and 743,881 ordered 32-byte call IDs;
- 23,804,192 call-index bytes, SHA-256 `ae399e393...`;
- chunk root `073ed886dd051aae2610d1aa771bce6c3012e25dca007c7614283ea9cac732ef`;
- manifest contract
  `8b7377135fb09db6666ad7f7ffad647a8aa617a9d7af5427c740abd627a9bd02`;
- one native-cache chunk, one gradient chunk, 38 DISCOVERY-ceiling chunks,
  950 fit templates, 38 spectral chunks, 38 VALIDATION-ceiling chunks,
  950 validation-evaluation templates, 95 singleton-necessity templates,
  855 two-Q templates, and 855 ordered-reader templates.

Every chunk offset is the preceding slice sum; every slice SHA and hash-chain
root replays; the global index length is exactly 32 times its call count. Call
descriptors bind the physical batch index, target/donor roles and independent
H/Q positions, prediction/answer/foil fields, fit slots including `A_C`,
sequence length, boundary, rank, seed, logical step, cache reads/writes, and
numeric array contracts. Equal-length batching and deterministic tails are
explicit. The selected Q and the unordered top-two raw-transfer Q pair are
represented independently, and the focused test activates a pair that excludes
the selected Q.

The independently reconstructed conditional prices are:

| path | forwards | backwards | graph batches | updates | examples | tokens |
|---|---:|---:|---:|---:|---:|---:|
| finite prefix through all DISCOVERY ceilings | 145 | 4 | 4 | 0 | 14,304 | 91,152 |
| reference full single-necessity + reader | 119,177 | 60,004 | 118,004 | 60,000 | 9,207,024 | 63,776,268 |
| conservative maximum active path | 119,207 | 60,004 | 118,004 | 60,000 | 9,207,984 | 63,782,508 |

Fixed retained numeric storage is 61,694,592 bytes. The declared maximum is
63,394,944 bytes: fixed plus the exact retained-site formula, 593,984 selected
bytes, 1,280 singleton-necessity bytes, 3,840 redundancy bytes, and 19,200
reader bytes. The implementation-dependent temporary peak is appropriately
not invented by the CPU compiler: it defers the digest and measured numbers to
a separately preregistered/reviewed canary on the exact future implementation.

Boundary `-1` correctly edits normalized embedding before establishing the
edited trajectory's `x0` and live block-0 `v1`; boundaries `0..17` retain the
target prefix's live `x0/v1`; the pure composed-edit tripwire passes the full
first edited trajectory to the later site. Spectral outputs remain diagnostics
and have no selection or success predicate. The manifest is FIT-only, names
SELECT/TEST/OOD as forbidden, and the dry run reports zero model calls and zero
GPU accesses. No task outcome is in the closure.

Full deterministic compilation at `PYTHONHASHSEED=0`, `1`, and `999` reproduced
the same manifest/index/chunk/contract hashes and the same 3,821/743,881
censuses. The checked dry-run SHA also reproduced.

## Blocking defects

### 1. The executable traversal is not causally stage ordered

The manifest is a complete set of mutually exclusive templates, but
`_compile_chunks()` places **all** fit templates before spectral or validation
chunks, while `_fit_templates()` places one site's selected-only A1/A2/rank-2/
rank-4 fits immediately after that site's joint-rank-1 fit. Concrete immutable
indices are:

- index 40: `fit:H:-1:joint:rank1:seed14001`;
- index 45: `fit:H:-1:A1_only:rank1:seed14001`;
- index 65: `fit:H:0:joint:rank1:seed14001`;
- index 990: `spectral:H:-1`.

Thus a selected-family fit is physically visited before even the next site's
joint fit and long before all joint fits, seed-health evaluation, spectral
diagnostics, and discovery selection can complete. `replay_active_path()` makes
one global traversal using a **final** `ActivePlanState` supplied before the
first call. A prospective producer therefore has to know later discovery and
health decisions in advance or violate the index order. This contradicts the
frozen DAG and the controlling addendum's rule to evaluate only predicates
available at each stage.

Repair requires an executable stage-transition schedule, not just different
prose: whole-index verification may remain pre-model, but physical execution
must seek/replay stage slices in causal order using a state that evolves only
from completed prior-stage primitives. Joint rank-1 must complete before its
finite-health decision and selection; selected family/higher-rank fits must
follow selection; locked validation must follow their completed health gate;
necessity, redundancy, and reader must follow their respective frozen gates.
Every active/inactive receipt and final path root must bind that causal order.

### 2. Whole-index preflight is neither exact nor required by replay

`preflight_global_call_index()` calls `validate_call_index()` on whatever
manifest-like mapping it is handed, but does not call `validate_manifest()` or
require the frozen 3,821/743,881/root/hash identity. The checked-in focused test
demonstrates the gap by constructing a two-chunk, 12-call mini-manifest,
monkeypatching the iterator to 12 descriptors, and obtaining status
`whole_global_index_verified_before_model`. That status is false as a statement
about this experiment's complete global index.

Moreover, `replay_active_path()` accepts no preflight receipt or token and can
be invoked directly; its own focused test does exactly that. Revalidating only
the supplied slices is not proof that the exact full compiler/manifest/index
closure was verified before model access.

Repair requires the public preflight to validate the exact frozen manifest,
count, roots, raw index hash, and all regenerated descriptors, returning a
receipt bound to those identities. Every stage replay must require and verify
that exact receipt. A private synthetic helper may remain for unit tests, but it
must not be able to emit the production success status.

### 3. Late operational aborts are impossible to represent honestly

The controlling second addendum distinguishes a completed, finite optimizer/
seed-health `instrument_invalid` scientific terminal from runtime, nonfinite,
incomplete, deadline, OOM, hash/canary, and publication faults, which abort
without a scientific package while still charging work that occurred.

The compiler's `BranchState` has only an undifferentiated
`operational_fault=True`. `project_terminal()` then requires every scientific
field null and `_node_statuses()` always says preflight failed, regardless of
where the fault occurred. `validate_execution_compatibility()` rejects an
operational fault if preflight, native cache, gradients, retained sites, or
selection are already present. An injected operational fault after an otherwise
valid scientific prefix therefore raises `CompileError`; its standalone
projection erases that prefix and reports only `preflight=failed` plus terminal
projection. `guarded_terminal_projection()` has the same problem when its
after-call deadline check fails: the action may have completed, yet the returned
projection says preflight failed and carries no incurred-work ledger.

Repair requires a typed operational-abort state bound to the exact fault stage,
active chunk/call position, completed and failed node statuses, active slices,
and incurred forward/graph/backward/update/example/token ledger. It must never
permit a scientific result/receipt package, but it must preserve completed work
and distinguish a before-call refusal from a completed call followed by failed
validation.

### 4. The static DAG conflates operational faults with scientific terminals

The hash-bound DAG labels failure of `preflight`, `native_cache`, and
`discovery_gradients` as `instrument_invalid`. It labels discovery-ceiling
failure `instrument_invalid_or_no_intervention_ceiling`. The same conflation
continues later: `spectral_operator`, `joint_rank1_fits`,
`spectral_finite_diagnostic`, `discovery_selection`, and
`selected_family_and_rank_fits` expose only `instrument_invalid`; locked
validation exposes only a rank/semantic scientific terminal.

Those labels cannot distinguish an incomplete/nonfinite/runtime fault, which
must leave no package, from a fully completed finite fit-health failure, which
may publish `instrument_invalid`. Nor can they distinguish a complete finite
absence of an intervention ceiling from a runtime/nonfinite ceiling fault. The
ambiguity persists through validation and the conditional intervention stages.

Repair every model-bearing node with an explicit operational-abort edge for
preflight/hash/canary/runtime/nonfinite/incomplete/deadline/OOM faults, plus only
the scientifically meaningful fully-completed finite edge at nodes where such a
terminal exists. Bind terminal projection to exact node completion and retained
finite primitives.

### 5. Runtime state is not actually type strict, and two gates are collapsed

Dataclass annotations do not enforce runtime types. Independent attacks were
accepted:

- `{"operational_fault":"false"}` produced an operational abort because a
  nonempty string is truthy;
- `eligible_h_count=True` was treated as integer 1 and could reach
  `no_intervention_ceiling`;
- `higher_rank_rescue=1` reached
  `fit_binary_state_rejected_higher_rank_needed_or_better`.

The same risk applies to boolean fields in `ActivePlanState` and boolean-valued
boundaries because Python `bool` is a subclass of `int`.

In addition, `ActivePlanState` combines the higher-rank falsifier and semantic
gate into one field, `semantic_and_falsifier_gates_pass`. `BranchState` keeps
`higher_rank_rescue` and `semantic_gates_pass` separate, but compatibility only
compares the semantic field to the composite field. It cannot independently
bind which predicate fired to the physical path/evidence.

Repair requires exact `type(value) is bool` / `type(value) is int` checks,
strict nullable-state validation, and separate active-state fields and receipts
for the higher-rank falsifier and semantic gate.

### 6. Eligible-H census is confused with retained top-three H

The authority may have more than three eligible H sites and then retain the
top three. The compiler caps `ActivePlanState.retained_h` at three, but
`validate_execution_compatibility()` requires
`branch_state.eligible_h_count == len(active_state.retained_h)`. A legal state
with four eligible H sites and three retained H sites is rejected with
`terminal H/Q census differs from active physical path`.

Repair must carry eligible counts independently from retained sites and enforce
`len(retained_h) == min(3, eligible_h_count)` with the frozen ordering/tie rule.
The Q eligible census and retained-Q set must likewise be explicitly related by
the authority's uncapped 19-site rule.

### 7. Deadline and namespace helpers are caller-bypassable

The manifest correctly says the hard deadline is 28,800 seconds and defers an
unknown throughput digest to a future reviewed per-physical-call-shape receipt.
The executable helpers do not bind either fact. `deadline_check()` and
`deadline_check_after()` accept caller-chosen `limit_seconds`; a 30,000-second
elapsed time was accepted when I passed 1,000,000. The p99 is also an unbound
caller float, rather than a verified future receipt and exact call-shape entry.
The after-call check compares only to `start`, not to the immediately preceding
timestamp: clock ticks 5 then 4 were accepted and returned a scientific
terminal.

`preflight_namespace_absent()` accepts an arbitrary sequence. An empty list
returned `all_absent`, so it does not prove absence of the exact result,
evidence, and receipt paths fixed by the manifest. A partial or substitute path
set has the same issue.

Repair must freeze the 28,800-second limit in the production API, require a
hash-bound future throughput receipt whose digest is intentionally unknown at
compiler time and whose p99 entry matches the exact physical call shape,
compare monotonic timestamps before and after each call, and bind namespace
preflight to exactly the three reserved destinations. Do not invent a canary or
timing digest in the compiler.

### 8. Skip receipts overstate calls and do not close an operational path

Every inactive receipt says `status=inactive_skip_zero_calls` but also places
the template's positive count in a field named `call_count`. That can be read as
incurred work and is incompatible with precise failed-work accounting. It
should distinguish `template_call_count` from `executed_call_count=0` and bind
the exact evaluated guard/stage state. More importantly, current receipts are
generated from the final-state traversal in defect 1 and cannot represent the
prefix and failed call in defect 3.

## Tests and exact attack results

With `PYTHONDONTWRITEBYTECODE=1`, `BQLIB_NO_MODEL=1`,
`CUDA_VISIBLE_DEVICES=''`, and `PYTHONPATH=.:basis_aligned/bilinear_quotient/ops`:

- relevant broad suite (v2 repaired compiler, rejected predecessor compiler,
  logical-v2 builder, and task-14 authority): **93 passed + 42 subtests** in
  7.89 s at `PYTHONHASHSEED=0`;
- focused v2 compiler suite: **30 passed + 32 subtests** at each of
  `PYTHONHASHSEED=1` and `999`;
- full source regeneration at seeds 0, 1, and 999 reproduced manifest
  `5f870a29...`, index `ae399e39...`, chunk root `073ed886...`, contract
  `8b737713...`, 3,821 chunks, and 743,881 calls;
- all eight coherent static mutations and a coherent chunk deletion/reindex
  were rejected;
- the seven executable attacks described above reproduced: three non-strict
  types accepted, caller deadline accepted, clock rollback accepted, empty
  namespace accepted, legal eligible-4/retained-3 rejected, and late
  operational fault rejected/erased;
- the candidate's own test establishes that a 12-call mini-manifest can receive
  the production whole-global-preflight success status and that active replay
  proceeds without a preflight receipt.

No test or review command imported `torch`, `jax`, `fastload`, `tt_model`, or a
checkpoint. No model, GPU, outcome, publication, runner, or enqueue operation
was performed.

## Decision and minimal prospective successor

**BLOCK** exact commit `6b7fb09ff30080e73cad0414d8315db660e04ca0`.
Its frozen scientific authority, complete call-template census, descriptor
content, prices, retained-byte formula, spectral nonselection, and boundary
semantics are useful inputs to a successor. The successor must nevertheless
replace the runtime contract with a causal staged replay; exact mandatory
full-preflight receipt; strict and independently represented predicates;
eligible-versus-retained census; exact namespace/deadline/receipt binding; and
a late-fault operational ledger with explicit operational edges at every DAG
stage. It then needs a new immutable commit and fresh independent CPU review.

This verdict does not authorize producer construction, model/checkpoint access,
a canary, GPU execution, publication, queueing, or enqueueing.

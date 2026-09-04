# Independent review: task14 FIT-localization-v2 physical compiler

**Reviewed:** 2026-09-04 UTC

**Exact candidate:** `ea16e22d28d125274ca4353f46e434c2826e0b02`

**Exact acceptance rubric:** commit `ef18d5ada4b3c5cf1692310cce9384ddc3f17b47`, SHA-256
`c85704d9124fa57e920558672110e72ec6afcf0b6faabe34e0501f086180f0b9`

**Verdict:** **BLOCK** this exact compiler from producer construction. The deterministic descriptor generator,
materialized call index, price arithmetic, and static runtime contract are unusually complete, but the manifest's public validator
accepts coherent changes to decision-critical fields after its self-hash is recomputed. The compiler also does not provide executable
guard/terminal projection or the rubric-required stale-namespace and deadline-as-null adversarial checks. Those are fail-closed
acceptance conditions, not details a later producer may invent.

I reviewed exact Git objects and frozen source roles using CPU-only operations. I did not instantiate or inspect a model/checkpoint,
use CUDA/GPU, read activation or outcome bytes, access queue/runner state, open SELECT/TEST/OOD, or execute a scientific run.

## Exact-object and authority closure

The candidate resolves to full commit `ea16e22d28d125274ca4353f46e434c2826e0b02`, parent
`c6810f61fd9ee17e64612edd1ca6574b5bac3764`, tree `bbae1f9c6bd49cc3df664241c5e6328bc1f4f541`. The acceptance rubric,
logical v2 authority `8f41f51cdf7e073063201cc48760622607ce91b9`, and independent authority approval
`2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923` are strict ancestors. The commit adds only the compiler, manifest, binary index,
dry run, focused tests, preregistration, and board receipt.

| Candidate object | Git blob | Raw SHA-256 |
|---|---|---|
| compiler | `e22cc59f4e00cc34454c0c28b1388c4244a88478` | `ffa56273f6fee686e193fa53cb8021f782536e79fbb629d30020a78cce065e6b` |
| call manifest | `6caee4f064d232b858eb2b904d46165ec491e3cc` | `f264ef64c03a2053f2c5344588d0adc8eb03ef3a8cb257d7d02c04f3a478568d` |
| binary call index | `c09a37097ada822dc992b23323beb17e6c3c4684` | `ae399e393d03af9b6232b7fc5339dd892b418ec7c88943735f8b72fc064c8ad9` |
| deterministic dry run | `2284d2927ddfd469d977ba3704a7e5d81616f771` | `c9c113dcd1b99fcd51a11046b984cde50d29d31be200aa778242eab079ab13a7` |
| focused tests | `54fe3524935d1a9f89ecbc82a0cf7afe2292db95` | `5bf582950bf1d14bef73cd6605839ef1a88856af6c03984968fc02f6fc9fd256` |
| physical preregistration | `e82c20e9dba14a21b3fba51fee79bb50f902bf86` | `c4b8f91f0a929c531d8c66785305d9f5fe23f601dd91ab5b7856650b3f0e7ac7` |

All 15 manifest closure entries recompute from the exact candidate Git tree to their printed SHA-256 values: FIT authority,
v2 builder/partition/donors/tests/preregistration/review, spectral derivation, experiment/artifact/result contracts, model source and
facade, fast loader, and its dependency. Only the FIT authority/partition/donor paths are allowed prompt authorities. The compiler
imports standard-library modules only and performs no model, checkpoint, CUDA, activation, result, or managed-queue operation.

## Call graph, descriptor, and schedule reconstruction

The checked-in manifest has contract root
`e2b0a1691eebc9846151818c9967a123459b0b5f944813c68ebd185b9659b5c7`, chunk root
`073ed886dd051aae2610d1aa771bce6c3012e25dca007c7614283ea9cac732ef`, 3,821 conditional chunks, and exactly 743,881 ordered
32-byte call IDs (23,804,192 bytes). I independently replayed every index slice and call hash chain from raw bytes; all offsets,
counts, slice hashes, and roots match. The chunk census is:

| Template class | Chunks |
|---|---:|
| native cache / discovery gradients | 1 / 1 |
| DISCOVERY full ceilings | 38 |
| fit templates | 950 |
| spectral diagnostic templates | 38 |
| VALIDATION ceiling templates | 38 |
| VALIDATION projected-evaluation templates | 950 |
| selected-Q necessity templates | 95 |
| top-two-Q redundancy templates | 855 |
| ordered H→Q reader templates | 855 |

The exact 400-step streams contain 12,800 logical slots per seed/configuration. Independent replay matches the registered cycles for
H/Q joint and A1/A2-only fits. Q-joint has 1,828 A_C endpoint slots; each family-only Q stream has 2,560. Every A_C slot names a
DISCOVERY C endpoint; every record slot and full-current-projector normalizer names a DISCOVERY donor record. Roles, cells,
target/donor token IDs and positions, ranks, seeds, boundaries, steps, batch ordinals, backward point, and array contracts are in the
canonical call preimage. Mutating an interior role, A_C endpoint, or boundary changes the call ID, and replacing that ID in the
binary index is rejected.

The boundary semantics are explicit: `-1` is normalized embedding input, `0..17` are after complete blocks, and boundary 17 resumes
with final RMSNorm, head, and softcap. One-site/full-state/necessity interventions and ordered two-site/reset/rescue variants are
stated correctly. Equal-length batching is enforced, native caches are shared once, and model-forward calls that differ by fitted
state or causal order are distinct.

## Independent price and storage reconstruction

I reconstructed the maximum active path from exact chunk IDs rather than accepting the summary. Boundary-specific prices are
invariant within each position/configuration, so the representative H/Q sites are valid. Results exactly match the manifest:

| Active path | Forwards | Backwards | Graph batches | Updates | Examples | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| valid prefix through all DISCOVERY ceilings | 145 | 4 | 4 | 0 | 14,304 | 91,152 |
| full singleton-necessity + reader path | 119,177 | 60,004 | 118,004 | 60,000 | 9,207,024 | 63,776,268 |
| branch-complete maximum with redundancy + reader | 119,207 | 60,004 | 118,004 | 60,000 | 9,207,984 | 63,782,508 |

The retained-array arithmetic also reconstructs exactly. The fixed prefix is 61,694,592 raw float32 bytes; 22 retained H/Q sites
add 1,082,048; selected fits/evaluations add 593,984; singleton, redundancy, and reader branches add 1,280, 3,840, and 19,200.
The exact branch-complete maximum is therefore 63,394,944 bytes. The compiler correctly refuses to invent a CUDA allocator/graph
peak before an implementation exists. It requires a separately hash-bound largest-shape canary receipt, the stated free-memory
margin, an external eight-hour watchdog, p99 remaining-time checks, and hard abort rather than partial scientific publication.

The spectral operator is DISCOVERY-only, nonselective, and unable to satisfy any gate or change registered DAS initialization. Its
current static contract passes review.

## Blocking defect 1: coherent scientific and physical mutations pass validation

`validate_manifest()` verifies the manifest's self-computed `contract_sha256`, but it does not compare several critical sections
with canonical expected objects or recompute price from chunks. I changed one field at a time, removed the old contract hash,
recomputed it with the compiler's own canonical function, and called the public validator. All of these were **accepted**:

1. ordinary recovery threshold `0.50 → 0.0`;
2. H selection `largest DISCOVERY objective → largest VALIDATION objective`;
3. higher-rank improvement threshold `0.10 → 999.0`, suppressing the registered falsifier;
4. reversal of the nine decision-terminal labels;
5. first initialization seed `14001 → 99999`;
6. logical relations per update `32 → 1`;
7. boundary-17 semantics `after block 17 → before block 17`; and
8. maximum active forward count `119207 → 1`.

The exact on-disk `check_manifest()` happens to reject altered materialized bytes because it compares against a freshly generated
object. That does not satisfy the acceptance rubric: the advertised validator accepts a coherently self-hashed in-memory contract,
the focused suite omits the required validation-selection/rank-promotion/seed/price attacks, and a later consumer can unknowingly
trust `validate_manifest()` after parsing or transformation. This violates acceptance conditions 3, 4, 6, 8, 11, and 12.

The repair is prospective and mechanical: factor canonical constructors for `science`/decision gates and terminals,
`model_contract`, `physical_batching`, `initialization`, and every other static section, then require exact structural equality.
Recompute `conditional_price` exactly from the supplied chunks and require equality, including all active/template counts and
formulas. Validation must not treat a recomputed self-hash as authority. Add planted coherent mutations for every rubric item,
including seed removal/replacement, validation selection, higher-rank promotion, terminal change, batch/schedule/dtype, boundary,
price, intervention ordering, and disconnected spectral diagnostics.

## Blocking defect 2: branch and terminal behavior remains declarative

The 14-node DAG is acyclic and every broad node has a stable hash, but its conditions and failures are free-text strings. There is no
executable guard evaluator, terminal projection, namespace preflight, or deadline-failure projection in this compiler. In
particular:

- `discovery_full_ceilings` combines `instrument_invalid_or_no_intervention_ceiling` without a typed split;
- `locked_validation` combines `rank_or_semantic_terminal`;
- single necessity's failure label says `sufficiency_only` even though failure may activate redundancy;
- `terminal_projection` depends only on `ordered_reader`, with no specified skipped-node completion semantics or explicit early
  edges for no ceiling, semantic failure, failed necessity, H≥Q, or reader-unresolved branches; and
- no test executes the required simultaneous single-site/redundancy-success rejection, stale/occupied namespace rejection, or
  eight-hour overrun forbidden from becoming a scientific null.

Consequently a producer would still have to invent how booleans activate/skips propagate and which exact terminal is emitted on
several conditional paths. The immutable logical preregistration states the intended answers, but rubric condition 2 requires the
physical compiler to bind them into stable executable guards rather than leave them to a later implementation.

The minimal repair is an immutable successor with a typed branch-state schema and pure CPU guard/terminal projector. Give every
guard and terminal edge a canonical ID, explicitly define skipped-node completion, and prove exhaustive one-terminal coverage for
instrument failure, empty H/Q, rank rescue, semantic failure, necessity success/failure, redundancy available/unavailable/pass/fail,
H/Q ordering, and reader pass/fail. Add a tempfile-only create-exclusive namespace preflight and a synthetic monotonic-deadline
test demonstrating that expiration can only hard-abort without a scientific terminal or partial package. These do not require a
model or GPU.

## Blocking defect 3: active-index replay and boundary -1 are ambiguous

The manifest says a runtime must compare each regenerated descriptor with the “next” 32-byte global index entry, while the same
manifest says only activated conditional chunks are executed. Those rules conflict on the first skipped template: advancing only on
active calls compares against an inactive chunk, while skipping index entries without a specified rule no longer means “next.” The
per-chunk offsets and slice hashes contain enough information for a repair, but the execution algorithm is not stated. Freeze two
separate operations: (1) before any model access, regenerate the entire descriptor universe and verify the complete global index;
then (2) for each activated chunk, seek its immutable offset, replay exactly its indexed slice/root, and intentionally skip every
inactive slice without consuming it as an active call. Bind the activated-chunk sequence and skip receipts into the run root.

Boundary `-1` is also not fully specified for this exact model. The frozen source normalizes token embeddings, assigns that tensor to
the persistent `x0`, initializes `v1=None`, and lets block 0 derive the attention value cache `v1`. “Normalized embedding input before
block 0” does not say whether a `-1` edit occurs before or after `x0` is captured. The two choices differ because every later block
mixes in `x0`; they also differ if block-0 `v1` is cached natively rather than derived from the edited input. Freeze the causal
semantics as follows: at `b=-1`, edit the normalized embedding at the registered token first, set `x0` from that edited tensor, and
derive block-0/`v1` live. At every `b>=0`, run the target prefix natively, retain its target `x0` and live target `v1`, edit only the
registered residual position after block `b`, and continue at block `b+1`. Add synthetic tripwires that distinguish these paths.
Without this, two conforming-looking producers can implement different interventions and terminal values.

## Test record and narrow conclusion

- Exact `--check`, including regeneration of all 743,881 descriptors and the binary index: **PASS**, zero model/GPU/outcome access.
- Focused compiler + v2 builder + frozen task14 authority: **63 passed, 10 subtests passed in 4.49 s**.
- Independent index-chain, chunk census, schedule-role, active-price, retained-byte, DAG-topology, and closure checks: **PASS**.
- Interior call role/A_C/boundary mutations: IDs changed and patched index rejected.
- Eight coherent decision/seed/batching/boundary/price mutations: **8/8 incorrectly accepted** by `validate_manifest()`.
- Active-vs-global call-index replay and `b=-1` handling of persistent `x0`/block-0 `v1`: **decision-changing ambiguities**.
- Rubric-required executable early-terminal, occupied-namespace, simultaneous-route, and deadline-null tests: **absent**.

Therefore exact commit `ea16e22d...` is reproducible as a call-template artifact but is not yet a fail-closed physical compiler under
the prospective acceptance rubric. No producer, adapter, authorization successor, execution, or enqueue should derive from it. A
new immutable compiler successor needs fresh independent review.

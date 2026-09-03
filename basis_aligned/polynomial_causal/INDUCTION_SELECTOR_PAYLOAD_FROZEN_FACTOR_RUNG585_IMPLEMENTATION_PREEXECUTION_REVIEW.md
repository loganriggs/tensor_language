# R585 frozen selector × payload implementation pre-execution review

**Review time:** 2026-09-03 UTC  
**Reviewed producer commit:** `a4e7c46c6339c75fc7f89c1e35339e15e3b74fd9`  
**Verdict:** **BLOCKED — do not execute or enqueue the reviewed producer bytes**

<!-- BQLANE: cpu -->

## Scope and outcome boundary

This is an independent, CPU-only review of the exact committed R585 producer,
owner test, dry run, builder handoff, replacement amendment, model-free
manifest, dependency lock, and the prior independent specification review. I
did not load the model, open CUDA, enqueue anything, or edit the committed
producer package. At the beginning of review and again at
`2026-09-03T21:07:56Z`, these three paths were absent:

- `induction_selector_payload_frozen_factor_rung585_results.json`;
- `induction_selector_payload_frozen_factor_rung585_receipt.json`; and
- `induction_selector_payload_frozen_factor_rung585_evidence/`.

I did not open or stat an R585 outcome. The review therefore remained
pre-outcome.

The implementation gets most of the scientific design right, but it does not
close several fail-closed requirements imposed by the already approved
replacement specification. In particular, the non-equality reconstruction is
circular, the endpoint × site × role execution census is not materialized or
hashed, the actually realized bootstrap cells are not checked against the
frozen 124-cell census, and a held result can validate without its required
evidence. These are execution blockers, not post-outcome documentation issues.

## Exact reviewed bytes

- producer:
  `4911200ae12dd9c27a609879fded8aab1b5704ef1116f25079b5df7a40162ff3`;
- owner test:
  `71eab693b578478d39201c267cbea7311972602aec739de19de85acab59ca67e`;
- producer dry run:
  `9b1b8c7c6e66a6b4835fa9ad10219fee16583f34d8a72c41a803cf6be5bfab7d`;
- builder handoff:
  `b35e957e088d7617f4a79f8843ccd1a0faaf9c19afe47f0765064899ee2c50d0`;
- replacement amendment:
  `98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf`;
- model-free manifest:
  `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962`;
- manifest dry run:
  `dc81109bed0ef44c51224988a53d57143751a3f078a889c156a7a8862e52114f`;
- dependency lock:
  `908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7`;
- prior replacement pre-outcome review:
  `df74a28df01c56e51aef3ac262302704d5d995970e1a40d1c4260d2aef3e55fb`;
- original preregistration review:
  `b8b4bcae6d2a24781383a5595a7c78d2d58623df209e9b98f7037ecc10566b2c`;
- R578 row authority:
  `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6`.

The four committed R585 target files in the first four bullets are byte-equal
to their versions at `a4e7c46c6`, even though the repository HEAD moved during
the review.

## What the producer computes

For each native endpoint, each of four fixed attention sites, and semantic
source role $r\in\{A,C\}$, the producer extracts

$$
e_h^x(r)=p_h^x(q_x,k_x(r))\mathbf 1[t^x_{k_x(r)-1}=t^x_{q_x}],
\qquad
u_h^x(r)=W_h^O v_h^x(k_x(r)).
$$

It freezes recipient and donor factors before any intervention. The replay,
score-only, payload-only, and joint terms at a site are respectively

$$
\sum_r e_r^x u_r^x,\qquad
\sum_r e_r^y u_r^x,\qquad
\sum_r e_r^x u_r^y,\qquad
\sum_r e_r^y u_r^y.
$$

During an intervention it recomputes the recipient's equality term from the
live state, removes that live term, and inserts the appropriate frozen hybrid.
Both L8H3 and L8H4 are computed from the same unmodified layer-8 input and then
added in one layer-8 transaction. This is the intended operational test of
whether the continuous attention score and projected value behave like
separable selector and payload factors; it is not a rank-reduction test.

## Checks that passed

1. **Authority and dependency parsing.** The runner hashes all frozen sources,
   strictly parses JSON, validates the locked R586/R587 verdicts, and checks the
   exact checkpoint before science.
2. **Semantic mapping.** The 2,808 included R578 rows rebuild 2,592 unique
   endpoints and 5,616 directed pairs. Source, payload, and query coordinates
   are checked against each token sequence. The native lengths are exactly
   19, 20, 21, 22, 27, 28, 29, and 30; roles are never found by token search.
3. **Frozen/live timing.** Native capture precedes frozen insertion construction,
   which precedes every intervention. The inserted term is frozen, while the
   removed recipient term is recomputed live after upstream changes.
4. **Complete site transaction.** Both A/C roles are summed inside each of L5H5,
   L7H3, L8H3, and L8H4. L8H3 and L8H4 share one pre-intervention state.
5. **Native and padding comparisons.** The code compares padded replay logits to
   a separately run unpadded native comparator and requires each shorter length
   to activate the padding tripwire.
6. **Causal coordinates and scales.** Answer-changing and match-break directions
   use the frozen signs and logit/CE definitions. Insertion norm, margin-logit,
   and full-vocabulary-logit RMS remain distinct fields. The 192 control
   lookups are FIT-sourced and cover all arm/condition/control cells.
7. **Membership and aggregation.** `_records_for` rejects missing, duplicate, or
   wrong group membership for each scored cell. Recovery is the ratio of group
   means and ratio of group medians, not the mean of rowwise ratios. Active
   controls score all groups once the cell is active.
8. **Price and split closure.** FIT uses
   $54+3(117)+54=459$ forwards. SELECT uses
   $27+3(59)+27=231$. SELECT is not opened after any FIT failure; a complete
   held path is exactly 690 forwards, with zero backwards or updates.
9. **Basic output typing.** The top-level result and receipt use exact field sets,
   require a scalar string `next_step`, reject non-standard JSON constants when
   reading, preserve all failure clauses, and apply deterministic terminal
   precedence.
10. **Bootstrap formula on intact input.** Instrumentation of the current
    `score_split` implementation observed exactly 124 unique FIT bootstrap IDs,
    equal to the frozen manifest. The SHA draw rule and big-endian draw/statistic
    hashes also reproduce independently.

## Blocking findings and precise repairs

### 1. The non-equality reconstruction is circular

The producer sets

```python
remainder = head_output - canonical_term
```

and then checks `canonical_term + remainder == head_output`. This identity is
true for any proposed canonical term. It does not satisfy the prior review's
requirement to compute the non-equality contraction independently of the
equality contraction.

**Repair:** compute the non-equality head contribution directly from the
complement of the exact equality support, through a separately named and tested
contraction (for example `contract_without_induction_fetch`). Check both
`e*u == canonical equality contraction` and
`canonical + independently contracted remainder == full head output`. Save the
two observed maximum errors per endpoint/site in evidence.

### 2. The required endpoint × site × role operation manifest is absent

The dry run lists endpoint, site, and role names and the capture loop detects a
missing site for an endpoint, but this is not equivalent to a materialized,
ordered, hashed operation census. There is no field that binds each exact
`(split, endpoint_id, site, role)` key. A smaller or differently ordered runtime
operation set can therefore escape the result contract.

The independent census is 20,736 keys total: 13,824 FIT and 6,912 SELECT. Under
the explicit record schema `{split, endpoint_id, site, role}`, sorted by those
four fields and serialized as canonical JSON with sorted keys and compact
separators, the reviewer reconstruction hashes to
`82169667d6f658b993f882b7b9951e07ae93149e5d5138fce548f6205e88cc5e`.
That serialization is a repair proposal, not retroactive authority.

**Repair:** freeze the exact schema/order/hash in a revised dry run, materialize
the expected keys before model loading, collect the realized keys during
capture, and fail unless expected and realized lists are identical. Bind the
same hash and split counts into raw evidence and the validated result.

### 3. Expected bootstrap metadata is not bound to realized scoring

On intact planted data, current formulas do happen to issue all 124 bootstrap
objects per split. But `score_split` never compares the realized ordered cell
IDs to `expected_bootstrap_cells`. Removing one FIT target cell from the
manifest made scoring finish without raising. Merely copying the expected
`bootstrap_cell_ids_sha256` into result metadata does not prove that the 124
reported bootstrap objects were realized.

**Repair:** collect every bootstrap object's cell ID during each split, require
exact equality to the frozen ordered 124-ID list before a split can be complete,
and save both the realized count and realized-ID hash. The validator and future
auditor must reconstruct those IDs from `split_scores`, not trust the expected
hash field.

### 4. Held evidence and provenance validation are too shallow

`validate_result` accepts a planted held result after replacing
`evidence_files` with `[]` and `raw_evidence` with `{}`. It also accepts an
all-zero checkpoint hash. `validate_receipt` accepts a receipt whose
`result_path` points somewhere other than the R585 result. These are direct
fail-closed failures.

For a completed FIT+SELECT run the full evidence census should include:

- endpoint factors: 2,592 endpoints × 4 sites × 2 roles;
- endpoint arrays with exact shapes `[2592,4,2]`,
  `[2592,4,2,1152]`, and three `[2592,4,1152]` arrays;
- 16,848 directed-arm rows and matching live/delta arrays of shape
  `[16848,4,1152]`; and
- 2,592 endpoint measurement rows.

The current files contain enough primitive endpoint and row quantities to
recompute most target/control statistics, but the validator does not require
those files, their exact path-dependent counts/shapes/dtypes/order hashes, or
the correspondence between array and JSONL membership. It also does not save
the passing native-attention, equality-factor, independent-remainder,
replay/native-logit, and padding-tripwire maxima needed to audit the instrument
rather than only its failures.

**Repair:** define and enforce a terminal-dependent evidence schema. A held
result must have every exact descriptor, row count, shape, dtype, order hash,
membership join, instrument maximum, and realized census. Require
`checkpoint_weights_sha256 == CHECKPOINT_SHA256`; require the canonical R585
`result_path`; and verify the receipt against the exact result bytes rather than
only a reserialized in-memory object.

### 5. Nonfinite tensors can poison the final namespace before rejection

`validate_primitive_logit_identities` compares errors to thresholds without
first checking finiteness. A NaN margin produces no failure because every
ordered comparison with NaN is false. Although `allow_nan=False` eventually
rejects nonfinite JSON, that happens after `write_evidence` has created the
final evidence directory and may have written NaNs into `.npy` files.

**Repair:** check finiteness of every primitive scalar and every array before
scoring or writing. Nonfinite instrument data must become `invalid_instrument`
or abort into a recoverable staging namespace; it must never create a final
artifact.

### 6. Evidence, result, and receipt publication is not atomic

`write_evidence` creates the final evidence directory and writes files into it
one at a time. `_finish_result` then writes the final result before the receipt.
A crash can leave a partial final evidence directory or a result without a
receipt, after which the producer refuses to rerun because the namespace is
occupied.

**Repair:** write evidence, result, and receipt into unique same-filesystem
staging paths, validate hashes and schemas there, flush/fsync as appropriate,
and atomically rename only a complete package. Define a recoverable stale-stage
policy; never treat a partial final namespace as a scientific outcome.

## Managed execution adapter

Running the committed producer with no arguments and no environment exits 2
before model loading because it requires exactly one of `--dry-run` and
`--execute-science`. The queue preflight's `BQLIB_DRYRUN=1` path works, but the
managed runner's real no-argument call cannot execute the producer directly.

A separate, currently uncommitted adapter appeared during review:

- `ops/execute_induction_selector_payload_frozen_factor_rung585.py`, SHA-256
  `58827580fa97013e444c40dc01760a1754e4366f0188c9d076c759df5c3025fc`;
- owner test SHA-256
  `d7afd107fc449255aff8feffd44b76d5b6573d9df98a042c08afc01b75728575`.

Its behavior is correct for this narrow dispatch problem: preflight hashes the
exact producer, owner test, dry run, amendment, manifest, lock, and review; the
`BQLIB_DRYRUN=1` branch runs the producer dry run against a temporary path with
zero model calls; and the real no-argument branch uses `os.execv` with exactly
`[python, producer, "--execute-science"]`. Its nine owner tests, repository
gate, preflight, and literal managed dry run all pass.

This adapter does not repair the six scientific/artifact blockers above. It
must not be queued while it pins the blocked producer. After the producer is
repaired and reviewed, freeze a new adapter that pins the repaired exact bytes;
do not weaken the producer's explicit science flag.

## Resource review

The producer's factor, endpoint-logit, live/delta, and serialization buffers
fit comfortably within the stated under-4-GiB transient CPU-memory bound. A
two-copy atomic evidence publication should remain under the stated 2-GiB disk
staging/final bound. At review time, `/proc/meminfo` reported 26,521,432 KiB
available RAM (27,157,946,368 bytes), and `/workspace` had 9,322,115,072 bytes
free. Resource capacity is not a blocker. The missing atomic publication
protocol is.

## Test evidence

- The R578 row, manifest, prior replacement-adversarial, and committed owner
  suites pass: **59 passed**.
- The adapter owner suite passes: **9 passed**.
- Producer and adapter both pass repository gate and preflight.
- The producer's managed no-argument dry run and the adapter's isolated managed
  dry run are model-free and pass.
- The separate implementation adversarial file is
  `ops/test_induction_selector_payload_frozen_factor_rung585_implementation_adversarial.py`.
  It has 10 passing conformance tests and seven strict expected-failure repair
  contracts. Running it with `pytest --runxfail` exposes exactly the seven defects:
  independent remainder, operation census, realized bootstrap binding, held
  evidence, checkpoint binding, receipt path, and nonfinite rejection.

## Causal interpretation and remaining scientific limits

The counterfactuals are meaningful for the narrow R578 circuit. Selector-only,
payload-only, joint, and equality-break changes manipulate distinct semantic
coordinates, both physical directions are present, and each cell contains 72
FIT or 36 SELECT groups with varied tokens, layouts, fillers, and lengths.
Multiple valid realizations are therefore represented rather than one prompt
pair. Active unrelated controls and exact algebraic no-op identities help
distinguish the proposed computation from broad damage.

If held after repair, the result would show only that the complete four-site,
oracle-equality-supported term behaves like a selector/value factorization on
R578 FIT/SELECT. It would not identify unique Q/K features, prove OOD
generalization, isolate individual-site necessity, establish interchangeable
redundancy between sites, or yield a weight-level compiler. Shared computation,
as opposed to shared difficulty, requires the crossed score/payload predictions,
donor-directed margin and CE effects, exact no-op identities, group-disjoint
SELECT reuse, and preservation of sufficiently active controls together.

## Five-part reusable knowledge packet

1. **Dataset/audit pattern.** The useful pattern is a literal semantic row
   authority, both directions, multiple counterfactual families, separate FIT
   and SELECT groups, typed active controls, and independent reconstruction from
   primitive evidence. What failed here was treating expected hashes as proof
   that runtime operations/statistics were actually realized.
2. **Reusable row/arm/site mapping.** Build endpoint records from saved semantic
   A/C source and payload positions; build each directed row by explicit
   recipient/donor endpoint IDs; freeze both endpoint factors before any arm;
   sum A/C inside each site; apply score, payload, and joint arms separately;
   keep same-layer sites in one transaction.
3. **Smallest exact term.** The audited atom is one site/role product
   $e_h^x(r)u_h^x(r)$. The scientific intervention uses the two-role sum at each
   of four sites. Exactness needs an independent equality contraction and an
   independent non-equality contraction, not subtract-and-add self-consistency.
4. **Active-control checks.** Activity is the median of the four actual delta
   norms and uses a FIT-frozen insertion-norm scale. Outcome preservation uses
   separate, unit-matched margin and vocabulary-logit scales. At least two
   distinct control families per arm/direction/condition must be active, and
   all groups in an active cell must be scored.
5. **Failure class and unresolved risk.** Current failures are instrument
   exactness, expected-to-realized census binding, evidence/provenance
   validation, nonfinite handling, and atomic publication. Even after repair,
   the scientific risk is that an all-four oracle-supported intervention shows
   operational factorization without identifying a reusable learned basis or
   separating duplicated from distributed computation.

## Ranked failures for the next builder and wave-2 prompt amendment

The next builder prompt should preempt, in order: (1) circular exactness;
(2) expected manifests not compared to runtime realizations; (3) incomplete
evidence accepted on a held path; (4) nonfinite data reaching storage;
(5) result/receipt/path/checkpoint provenance gaps; (6) non-atomic final
publication; (7) wrong semantic positions or frozen/live timing; (8) unit
collisions; (9) FIT/SELECT or price leakage; and (10) overclaiming an all-four
result.

Recommended critic-prompt addition:

> Before approving execution, independently enumerate and hash both the
> endpoint×site×role operation list and the 124 bootstrap cells per split. Then
> instrument the producer so the exact runtime lists—not expected metadata—must
> equal those frozen lists. Mutate each list by one omission and require a hard
> failure. Require a held fixture to fail if any evidence file, shape, row,
> order hash, checkpoint hash, result path, finite-value check, or independent
> equality/non-equality reconstruction is missing. Crash-inject after each
> write and prove no partial final result/evidence/receipt namespace remains.

## Disposition

Do not run or queue R585 on producer SHA
`4911200ae12dd9c27a609879fded8aab1b5704ef1116f25079b5df7a40162ff3`.
Repair and refreeze the producer, dry run, owner/adversarial tests, and managed
adapter; then repeat this outcome-blind implementation review before the first
model call.

# R585 repaired execution package: independent pre-execution review

<!-- BQLANE: cpu -->

**Review date:** 2026-09-03 UTC
**Reviewed commit:** `27e4beaaf94704031cc07fabbcdd8888aa990f46`
**Verdict:** **BLOCKED — do not execute or enqueue these exact bytes**

## Exact candidate and outcome boundary

This review is bound to Git commit `27e4beaaf`, not later working-tree edits:

- producer:
  `dcdb6470e481dcbc58e86997f4a4d0e3203607ae29a0b74b0e58f59abf62db58`;
- owner test:
  `cf4326ba6500814767e4b5ee17952753cbda39d6368e91c45a47a1ddce10cc63`;
- dry run:
  `a30d8206b11beb691e2b9dd2ce33a3a3c2df6752388643f13f0fc81442c69118`;
- managed adapter:
  `e96c72a83f199f84896ab17e3ce5e9aa9d01c8ec973e319cf4039a0866bcb301`;
- adapter test:
  `8fc015ec973c7159c59231fd52f567ff6d11c15322433d4f5ff3e4bdf3dbaf60`.

At the beginning of review and again at `2026-09-03T21:44:25Z`, the R585
result, receipt, and evidence-directory namespaces were all absent. I used only
existence checks and did not open or create an R585 outcome, load the model,
open CUDA, enqueue work, or touch the board or registry.

## Status of the six frozen blockers

The repair closes the narrow literal attacks from the first BLOCK review:

1. **Independent remainder:** `contract_without_induction_fetch` now computes
   the non-equality contraction directly. The old
   `remainder = head_output - canonical_term` identity is absent. Equality and
   independent-complement reconstruction errors are recorded.
2. **Operation census:** the exact ordered 20,736-key
   `(split, endpoint, site, role)` manifest is materialized and hashes to
   `82169667d6f658b993f882b7b9951e07ae93149e5d5138fce548f6205e88cc5e`,
   with 13,824 FIT and 6,912 SELECT keys. Removing one realized key raises.
3. **Bootstrap realization:** intact scoring realizes exactly 124 unique IDs
   per split, and the result validator reconstructs IDs from the nested score
   objects. Removing one target cell raises rather than accepting copied
   expected metadata.
4. **Held/checkpoint/receipt envelope:** a held fixture without raw evidence is
   rejected; the checkpoint must equal the frozen checkpoint; receipt path and
   exact canonical result bytes are checked.
5. **NaN/Inf:** primitive scalars, arrays, JSON, and JSONL receive explicit
   finite checks before final publication. The planted NaN attack now fails.
6. **Staged publication:** evidence, result, and receipt are written and
   validated in a same-filesystem stage, fsynced, and renamed in that order with
   the receipt last. Injected Python exceptions after each rename roll the
   package back to the stage, and the producer can quarantine stale stages or
   partial finals without deleting them.

The semantic row mapping, frozen donor/recipient factors, live removal,
same-state L8H3/H4 transaction, typed scales, target/control predicates, and
FIT-first decision remain unchanged. The declared execution envelope is still

$$
54+3(117)+54=459\quad\text{FIT forwards},
$$

$$
27+3(59)+27=231\quad\text{SELECT forwards},
$$

for a maximum of 690 forwards, zero backwards, and zero updates. Only FIT and
SELECT are named as opened splits; FINAL_TEST and OOD remain forbidden.

## Remaining execution blockers

### 1. Completed scientific-null results still require no evidence

Evidence validation is conditional only on the held terminal. The public
validator accepts
`make_result_fixture("factor_capacity_null")` with 459 claimed forwards,
`evaluated_splits=["FIT"]`, `evidence_files=[]`, and fixture-only raw evidence.
The same applies to other non-held complete scientific terminals. This leaves
the earlier requirement for a **terminal/phase-dependent evidence schema**
unmet: a scientific null can be published without the primitive rows needed to
distinguish a real null from missing or malformed execution.

**Required repair:** separate planted-fixture validation from production-result
validation. For every completed scientific terminal, require the exact evidence
appropriate to its evaluated splits and stopping phase. A complete FIT null
must bind 1,728 endpoint rows, 11,232 directed-arm rows, 6,912 endpoint/site
factor rows, the corresponding arrays, realized operation/bootstrap censuses,
instrument maxima, and primitive statistics. A SELECT failure must bind the
full FIT+SELECT census. Only a genuinely early invalid-instrument path may use a
smaller explicitly recorded phase census.

### 2. Held evidence is self-consistent but not joined to the frozen authority

The held validator enforces counts, sorting, uniqueness, descriptor hashes,
shapes, dtypes, and algebraic array identities. It does not compare the actual
endpoint IDs, directed IDs/arms, or endpoint/site factor IDs in the JSONL files
to the frozen endpoint and direction authorities. It also does not join the
live/delta arrays back to the exact directed rows and frozen recipient/donor
factor sources.

The independent miniature attack supplied a valid expected operation manifest
for `expected-endpoint` while all evidence files used a self-consistent
`invented-endpoint` and `invented-direction`. `_validate_held_evidence` accepted
the package. Thus count plus self-generated row-order hash is not an authority
membership check.

**Required repair:** reconstruct exact expected orders from
`build_execution_authority`: endpoint order; every directed ID crossed with
the three arms; and every endpoint crossed with the four sites. Require literal
equality of evidence keys to those orders before accepting their hashes. Check
each directed row's recipient/donor endpoint IDs and semantic cell membership.
Then verify for each directed row/site that

$$
\text{live removed}+\Delta=\text{frozen inserted term}
$$

using the native endpoint $e/u$ tables and the registered arm mapping, and
recompute the saved activity norms and primitive score statistics.

### 3. Partial-final recovery is unreachable through the managed entrypoint

The producer's `recover_stale_publication` works when called directly. However,
the actual no-argument managed adapter runs `require_unused_namespaces()` in
`preflight()` before it invokes the producer. After an uncatchable process or
machine failure between publishing evidence and publishing the receipt, a
partial final namespace exists. Every managed retry therefore stops in the
adapter and never reaches the producer's quarantine code.

The in-process rollback tests do not cover SIGKILL, power loss, or interpreter
termination, which are the reason restart recovery exists.

**Required repair:** make partial-publication classification and quarantine
reachable from the exact managed no-argument path after byte verification and
before the unused-namespace guard. Preserve a complete receipt-marked package;
quarantine only incomplete final/staging paths, report that recovery occurred,
and require a clean subsequent invocation before science. Add a managed-entry
test with partial evidence/result finals, not only a direct producer-helper
test.

## Validation performed

- The row, manifest, prior specification-adversarial, repaired owner, and
  adapter suites passed: **81 passed**.
- Producer gate and preflight: **PASS**.
- Adapter gate and preflight: **PASS**.
- Literal managed `BQLIB_DRYRUN=1` adapter execution passed with zero model
  forwards/backwards/updates, no model/CUDA load, and no outcomes opened.
- The old frozen BLOCK test reports seven strict XPASS repairs plus its expected
  obsolete-byte-pin failure, confirming that its literal attacks are repaired
  but that it is not an acceptance suite for new bytes.
- Fresh independent regression file:
  `basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_repair_review_adversarial.py`.
  Default result: **4 passed, 3 strict xfailed**. With `--runxfail`, the three
  attacks above fail explicitly: evidence-free scientific null accepted,
  invented held-evidence membership accepted, and managed partial-final
  recovery absent.

## New reusable playbook lesson

Two forms of local correctness are insufficient for a runnable circuit
experiment:

1. **Self-consistency is not authority binding.** Counts, sorting, and hashes
   prove that an artifact is internally consistent only after its exact IDs are
   compared to the independently frozen endpoint/direction/site/arm authority.
2. **Recovery must be tested through the deployed entrypoint.** A correct
   low-level recovery helper does not provide crash recovery when an outer
   adapter rejects the partial state before that helper can run.

Future builder/critic prompts should always include one invented-but-consistent
membership attack, one evidence-free non-held scientific terminal, and one
hard-crash restart test through the same executable that the queue invokes.

## Disposition

Do not execute or enqueue commit `27e4beaaf` with the five hashes above. Repair
the three fail-closed gaps prospectively, regenerate the owner test, dry run,
adapter, and adapter test, and request another exact-byte outcome-blind review.

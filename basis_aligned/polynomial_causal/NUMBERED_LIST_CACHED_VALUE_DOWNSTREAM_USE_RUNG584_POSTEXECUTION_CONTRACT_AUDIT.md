# R584 numbered-list cached-value downstream use: post-execution code/evidence-contract audit

**Audited:** 2026-09-03 UTC  
**Boundary:** post-execution, CPU/model-free contract audit  
**Verdict:** **AUDIT BLOCK** for the exact R584 producer/evidence contract

This is not a pre-GPU approval and cannot retroactively authorize the run. The R584 result already existed before
this review lane began. During initial orientation I ran a filename-wide `sha256sum` over `*rung584*`, which exposed
the result filename and SHA-256 before I noticed the execution had landed. I did not parse its scientific values at
that point. After the parent explicitly relabeled the task, I inspected only the fields needed to reconstruct schema,
membership, computations, split decisions, and publication integrity. Therefore this review makes no outcome-blindness
claim.

## Exact audited provenance

The current worktree bytes equal the corresponding committed blobs:

- repaired producer commit: `55b138ed7d270fa6b103f06006091f761cf54af8`;
- producer: `50609756d97de2f13f717774f13d72b1c743f38a172375e9b08efc2b055336c7`;
- owner test: `37cc8f73ed128ebdb17b5cfcdb1248bc240291e9a10d38c526ac7d4a76ea3cce`;
- pre-outcome adversarial test: `900883046b648c7c9aa0714fff3d7d0da678b70ab8598623321e4f9d32bb5cd2`;
- implementation note: `612005760bccda8f1a9f16b540b0734de3241e5da1c40246f514509733539181`;
- dry run: `b2ebe65c92ea5170ab13394c1ffee8562ff4241f481a6fce392a00b200149fe8`;
- landed result: `7980753636fab422ed6c609a1afd054f99ed7f903e2bb3e61eddf0617316fdf6`, added by commit
  `49ee596000e83456a73f3008de84b306e8167e60`;
- existing independent R588 audit: `9d3bb3fbcc8760659d7eba5ca723fec320b34249911aa4036219adc6edf9a99d`, added by
  commit `8fe40bb171f4776d1dff81e3d277ec268fbca2f0`.

The result commit adds one JSON result file and a board entry. There is no R584 result receipt or separately published
evidence namespace.

## What mechanically holds for the landed artifact

The saved artifact itself passes the independent R588 reconstruction. Without a model call, that reconstruction:

- requires strict finite JSON and the exact frozen R582 row authority;
- verifies the 576-row FIT capture and every one of the 12 FIT intervention arms;
- binds row tokens, semantic coordinates, conditions, answers, sites, components, and arm names;
- recomputes endpoint identities, intervention effects, all candidate reports, 432 deterministic bootstrap cells,
  the fixed first-pass selection rule, split opening, forward count, predicates, interactions, decision, and next step;
- confirms the FIT-only 379-forward path, no SELECT evidence, and no FINAL_TEST/OOD opening.

The result had no provisional FIT candidate, so scientific null interventions were correctly not executed. The two
frozen donor maps are nevertheless deterministic and pass the existing donor-validity tests; there are simply no
landed null-arm rows to audit on this terminal path.

The counterfactual authority also preserves the intended primary match: successor versus copy at fixed group,
representation, source level/value, and final source token. It includes ordinary and surface-rewrite realizations plus
relation-break and `+2` conflict controls. The preregistration explicitly leaves two interpretations—relation-conditioned
use and last-value successor use—as characterization alternatives rather than silently choosing one. Since no candidate
was promoted, the landed null does not license either interpretation.

These checks support the narrow already-recorded R588 statement about the 12 fixed coarse terms. They do not repair the
producer boundary below.

## Blocking defects in the exact producer contract

### 1. The final validator does not derive the scientific result from the saved evidence

`validate_scientific_result` validates only the capture table with the generic result contract. It does not validate
the membership or primitive identities of `fit_raw`/null/SELECT arms, recompute any candidate or null report, rerun the
bootstrap, derive selection, recompute interactions and predicates, or require the terminal decision and next step to
follow from those values.

A planted mutation can erase `fit_reports`, change `decision`, and replace `next_step`; the exact producer validator
still accepts the package. R588 catches this, but a later auditor is not a substitute for the producer's pre-publication
boundary. The producer must call a reconstruction at least as strict as R588 before publishing.

### 2. Unretained implementation checks can become a scientific decomposition null

Native attention replay and several exact tensor equalities are reduced to saved error scalars. The full tensors needed
to reconstruct those checks are not retained. The execution code uses those scalars to set `fit_exact_pass`; a failure
sets the provisional candidate to null and can therefore flow into the same `downstream_use_decomposition_null` terminal
used when a valid instrument finds no component.

The final validator neither checks the exactness threshold nor derives `fit_exactness` from capture rows. A planted
package with replay errors far above `1e-10` and correspondingly bad `fit_exactness` is accepted. Under the shared v3
contract, an unretained replay/native or full-tensor exactness failure must hard-abort before any final namespace is
published. It cannot be reported as a scientific null. Retained algebraic checks may instead produce a separately typed,
evidence-derived invalid-instrument terminal.

### 3. Publication is neither atomic nor mutually receipt-bound

The real path ends with direct `OUT.write_text(...)`. It has no same-filesystem staging directory, `fsync`, atomic rename,
crash injection/recovery policy, result receipt, or mutual result/evidence/receipt hashes. A crash can expose a partial
final JSON file, and the existing `OUT.exists()` guard then makes retry fail without distinguishing a complete result
from arbitrary or truncated bytes.

Git later made the observed result bytes durable, and R588 audited those observed bytes, but that does not prove the
runtime publication transition was atomic. The accumulated circuit workflow requires a prospectively new namespace
with staged validation and an atomic, mutually bound package.

## Why this is a block rather than a rejection of the existing narrow result

The actual landed bytes survive the stronger R588 content reconstruction, so this review found no mismatch in the
observed FIT membership, calculations, or decision. The block is specifically against treating the exact R584 producer
and publication path as a reusable current-generation evidence contract. Its validator accepts invented summaries and
invalid exactness, and its direct final write is crash-unsafe.

No historical R584 byte should be edited. Any new model execution based on this design should use a new prospective
namespace that:

1. validates every opened arm independently against authority;
2. recomputes reports, bootstraps, selection, predicates, interactions, and terminal fields from primitive rows;
3. hard-aborts every unretained live-computation failure before publication;
4. separates evidence-derived invalid-instrument terminals from scientific nulls; and
5. stages, validates, and atomically publishes a mutually hash-bound result/evidence/receipt package with safe retry.

Those repaired bytes would require a fresh independent review before execution. This audit does not self-approve such
a repair.

## Executable checks

The new post-execution regression file is
`ops/test_numbered_list_cached_value_downstream_use_rung584_postexecution_contract_audit.py`. Its positive tests bind
the exact result and reproduce R588's mechanical audit. Three strict expected failures preserve the exact attacks:
summary/terminal mutation, invalid replay/exactness publication, and missing atomic receipt-bound publication.

```text
pytest -q \
  basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_downstream_use_rung582.py \
  basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_downstream_use_rung584.py \
  basis_aligned/bilinear_quotient/ops/r584_preoutcome_adversarial_tests.py \
  basis_aligned/bilinear_quotient/ops/test_result_contract.py \
  basis_aligned/bilinear_quotient/ops/test_audit_numbered_list_cached_value_downstream_use_rung588.py \
  basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_downstream_use_rung584_postexecution_contract_audit.py
```

Result: `71 passed, 3 xfailed in 33.52s`.

The exact R584 producer still passes its static checks:

```text
gate.py: no findings; GATE: PASS
preflight.py: no findings
```

Those static passes do not exercise final scientific derivation or crash-safe publication, which are precisely the
three planted expected failures above.

## Five-part knowledge packet

1. **Dataset/manifest pattern:** source-matched successor/copy groups with two surface realizations and active
   relation/conflict controls work as a fully enumerable authority. Keeping relation-conditioned and last-value
   interpretations explicit is better than forcing one latent label. The missing piece is a machine-readable list of
   those interpretation alternatives for future positive promotion.
2. **Reusable semantic-coordinate mapping:** `row_coordinates` plus R588's exact identity join binds token IDs, query
   position, source position/value/token, and all registered answer IDs. This transfers unchanged.
3. **Smallest exact intervention term:** at each MLP site, remove `C`, `Q`, or `C+Q`, where `C` is the gauge-invariant
   background-by-cached-value cross term and `Q` is the cached-value self-interaction. The exact local identity is useful;
   the coarse terms did not establish a selective behavior component.
4. **Active-control pattern:** deterministic different-group/same-cell and same-source/other-action donor maps, with
   per-cell norm matching and relation/conflict rows, are reusable. A terminal that never reaches null arms must state
   that they were not evaluated rather than imply they passed.
5. **Failure class/unresolved risk:** the observed artifact is computationally reconstructable, but the producer has a
   bookkeeping/implementation failure: non-derived terminal fields, invalid-instrument/scientific-null conflation, and
   crash-unsafe publication. Scientifically, the remaining risk is that useful downstream use is finer than these 12
   coarse MLP terms or distributed across sites.

**Prompt/test improvement for the next wave:** require the builder to mutate one complete planted package at each
boundary—raw arm, report, exactness check, predicate, terminal, and every publication step—and prove the real final
validator rejects it. A separately passing auditor is necessary but must never be counted as evidence that the producer
itself failed closed.

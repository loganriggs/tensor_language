# Hourly strategic review — 2026-08-29 18:55 UTC

## Outcome first

The highest-priority experiment ran rather than remaining a plan.  The physical
MLP2 K=512 validation completed all 48 batches (576 full-model arm forwards) on the
192-document VALIDATION role.  It then failed its final semantic replay before a
receipt could be published.

This is an **invalid run with no scientific decision**.  The failure is narrower
than a numerical or model failure: the raw-payload guard rejected the metadata path
`role_summary.tensor_hashes.rows`.  That field contains only a SHA-256 hash string;
no raw row, token, target, logit, loss, product, state, or response payload was
published.  The failure artifact binds the exact authority, ledger, result, parent,
source-commit, and source-file hashes.  The invalid result's scientific metrics have
not been used.

The run took about eleven minutes wall time.  This is expected from 12 arms across
48 batches and is not evidence of a data-loading bottleneck.  In contrast, the
10.54-second mean fit stopped immediately after MLP2 and the 16.13-second calibration
used only 48 native forwards.

The special eight-hour exploration window ended at 12:00 UTC and was not reopened.

## What fraction of the model is actually explained?

No strict ledger moves on an invalid run:

| Currency | Current value | Remaining gap |
|---|---:|---:|
| Structural intervention surfaces | 36/36 | semantics and autonomous interfaces remain open |
| Consequence-certified whole-program storage removal | 5.348245316% | 94.651754684% uncertified |
| Named causal CE | 0.57968 / 5.30682 = 10.923302467% | 89.076697533% unnamed |
| Unexplained causal CE | 4.72714 nat | dominant quantitative gap |
| Complete terminal actions | 0/68 | extraction/removal/OOD chain open |

The other compiler lane has useful engineering compression, but that is not a named
causal mechanism and does not change these strict quantities.

## Largest remaining gaps

1. **MLP2 finite consequence is still unknown.**  All computation completed, but the
   receipt-last protocol correctly refused to validate an output whose metadata
   tripped the raw-payload guard.  We cannot use the stored metrics.
2. **MLP0--MLP2 compensation remains unresolved.**  C512 is a strong physical MLP0
   compression, but we still do not know whether a small MLP2 program composes with
   it or whether MLP2 needs a jointly refactored basis.
3. **Native product coordinates still lack a finite certificate.**  The suffix
   selector is downstream-aware, but its advantage over local, RMS, mass, deranged,
   and random selections has not received a valid held-out decision.
4. **Named circuits are sparse.**  Copy/induction is the strongest localized late
   consumer, while capitalization, numeric, syntax, and entity-continuation consumers
   are not yet a verified bank for assigning semantics to early MLP coordinates.
5. **Terminal utility remains untested.**  No complete extraction, selective-removal,
   and OOD-transport action has passed end to end.

## Candidate pruning and priority

The ranking uses expected information gain, causal relevance, whole-model
composability, falsifiability, GPU price, and duplication of completed work.

1. **Source-close a repaired MLP2 validation under a fresh namespace, then rerun it.**
   The existing numerical work is unusable, but the defect is a single false-positive
   metadata path.  A repair must whitelist only the exact
   `role_summary.tensor_hashes.rows` SHA-256 leaf, retain rejection of every actual
   raw payload, use a new create-only authority/result/receipt/failure namespace, and
   pass independent source audit.  This remains highest return because it answers the
   largest open finite interface question in one modest GPU run.
2. **If SUFFIX survives, cross it with MLP0 C512 and then the compressed copy edge.**
   This directly tests whether independent simplifications compose and measures the
   compensation term.  It is conditional on a valid rank-1 outcome.
3. **If all six native-coordinate arms fail, fit response-conditioned or balanced
   block factors.**  This changes the atoms rather than tuning K.  It uses downstream
   controllability/observability, but costs more and is therefore conditional on the
   native grammar failing.
4. **Complete a fresh terminal test of the exact copy edge.**  This is the best route
   to moving 0/68: test extraction, selective removal, collateral behavior, and OOD
   transport for the already-localized copy mechanism.
5. **Build a complementary verified late-consumer bank.**  Add capitalization,
   numeric succession, syntactic closure, and entity continuation only after each has
   a causal and selective-removal certificate.  Factor early-component effects across
   that bank to seek a sparse semantic coordinate system.

More MLP1 native-gate screening, raw Down-space reconstruction, nearby K sweeps, and
SAE/HOSVD reconstruction without final consequences remain pruned.  They duplicate
completed negatives or optimize a currency Family F already showed can disagree with
whole-model behavior.

## Action executed and preserved failure

Before execution, the exact five-file source passed an independent GO audit and 58
focused tests.  It was committed at `4ef38f38` and pushed; current pushed `main`
contains it as an ancestor.  The GPU was idle and the canonical namespace absent.

The run completed batches 1--48 and wrote the following partial artifacts before
semantic replay failed:

- authority SHA-256 `c22e1fe9e075953d95668893c371f73616823592ba035d6a6fb05c3cf9826bab`;
- ledger SHA-256 `f7146285e2206872184a26e4df39c45faa3f66e8b0b45959590fc2e38cad5a01`;
- invalid result SHA-256 `743ef3d5d503c170963202413edc216ba9ec6ce781c5bbb5c2371e49da655178`;
- failure SHA-256 `47473c67cd8d651af67a767620ede4338d9be3a705039697c04ff24b0ea24f46`.

The failure status is
`mlp2_cmr_v1_validation_failed_invalid_no_scientific_decision`, and replication
remains sealed.  No file will be overwritten and no metric from the invalid result
will be promoted.  The next safe CPU action is the narrow path-aware payload-guard
repair plus fresh-namespace contract and tests; a rerun requires that repaired source
to be committed, pushed, and independently audited first.

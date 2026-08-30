# Induction equality tensor FINAL/OOD v2 execution audit: NO-GO

Status: **NO-GO** for execution source commit
`f93a52f26bc7175573e65d28097a45e35ecbcf38`.

This is a static, non-authorizing audit record.  It is deliberately not named
`induction_equality_tensor_final_ood_v2_independent_audit.json`, does not use the
GO schema accepted by the execution owner, and cannot authorize authority creation
or model execution.

The review was outcome-blind.  No model or checkpoint was loaded, no protected role
tensor was opened or deserialized, and no execution outcome was inspected.  The row
receipt was read as JSON metadata only.  At review time the execution authority,
ledger, result, manifest, receipt, and failure namespaces were all absent.

## Immutable identities reviewed

- Execution source commit: `f93a52f26bc7175573e65d28097a45e35ecbcf38`
- Pinned execution-source SHA-256:
  `d3412c0a88ee7b4dd6cd0954edd514cdedf4eddce31f315e10b76cb5ffdb03c4`
- Pinned execution-test SHA-256:
  `9b77a397e3c76c933912c78423085d32b4545b01d777de6bd69bd56e6da1e464`
- Committed fresh-row receipt SHA-256:
  `755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e`
- Row receipt source commit:
  `331f34d8ebdc4ef060ce85042bddb691715f8b67`
- Declared transitive execution-source paths: 38
- Fresh roles recorded by the receipt: `label_fit`, `final_natural`, `ood_code`
- Receipt records `old_v1_role_tensors_deserialized=false` and
  `outcome_access=false`.

## What passed inspection

The following parts are scientifically coherent but do not override the blockers
below.

1. The observed-model facade is local-only, validates the production topology, and
   exposes explicit attention and MLP dispatch at all 18 sites without hooks.
2. The plan contains exactly six arms: native, full replay, equality removal, full
   selected-head deletion, equality extraction, and the fixed vocabulary-cycle null.
   Every analytical arm replaces attention only at L5H5, L7H3, L8H3, and L8H4;
   every MLP remains native.
3. A successful one-use forward owner proves one native or replacement attention and
   one native MLP call at every site.  Aggregated production expectations are exactly
   48 batches and 192 documents per role and arm, with zero native selected-attention
   calls in analytical arms.
4. Raw logits are immediately reduced to document/cell token count, NLL sum, correct
   count, directed native-to-arm KL sum, and a support digest.  The normal collection
   path does not serialize rows, masks, logits, or hidden states.
5. Point estimates are pooled token-weighted statistics.  The 20,000-draw bootstrap
   shares multiplicities within a role, uses independent role groups, and constructs
   one simultaneous family over all ten preregistered role coordinates.  A
   nonpositive extraction stake in the point estimate or any draw fails without
   clamping, redraw, or repair.
6. The natural and code gates implement the preregistered target, specificity,
   extraction, collateral, deranged-null, support, replay, and call-census thresholds.
7. Ledger, result, and manifest publication is create-only; ledger and result are
   reloaded and recomputed before terminal publication.  The failure path is
   create-only and checks for a rival receipt under the owned lock.

## Blocking defects

### B1. The authority does not bind the independently reviewed row parent

`freeze_authority()` hashes whichever file happens to occupy
`rows_v2.RECEIPT` at authority time.  The expected committed receipt hash
`755c456d...702e` is not frozen in the source and is not compared before authority
publication.  `_row_receipt()` checks only a small schema/status subset.  A different
schema-compatible receipt can therefore become the authority parent even though it
was not the receipt reviewed at `f93a52f2`.

Required repair: freeze the exact expected row-receipt SHA-256 in source (or derive it
from and compare it to the immutable audited commit), require exact receipt semantic
replay, and require the authority's row receipt and role-file map to equal that exact
parent before authority publication and on every later validation.

### B2. The checkpoint is not validated before authority publication

`protected_snapshot()` records the current config and weight hashes, but
`freeze_authority()` does not compare them to the facade constants or call the full
snapshot validator.  It writes `checkpoint_weights_sha256=facade.WEIGHTS_SHA256`
regardless of the hash stored in the protected snapshot.  During execution,
`load_bilin18(..., verify_weights_sha256=False)` may deserialize a wrong checkpoint;
the explicit equality check occurs only after loading it (although still before any
forward).

Required repair: run the full config, size, revision, and weight-hash validation
before authority creation; bind the resulting exact checkpoint receipt; and replay
that identity before model load.  No wrong checkpoint may be deserialized first and
rejected afterward.

### B3. `validate_authority()` ignores four claimed parent fields

The validator checks the source, audit, protected snapshot, and output paths, but it
does not reconcile these authority fields with their authoritative values:

- `row_receipt_sha256`
- `role_file_sha256s`
- `discovery_sha256`
- `checkpoint_weights_sha256`

An exact-key authority carrying false values in all four fields passes when the
protected snapshot itself is unchanged.

Required repair: reconstruct the complete expected authority dictionary from the
exact source audit, pinned row receipt, exact role entries, discovery parent,
checkpoint receipt, protected snapshot, and output namespace, then require exact
JSON-native equality.  The authority publisher should semantically replay its
temporary bytes before the create-only link.

### B4. Stored-ledger semantic replay is not document- or schema-closed

`semantic_validate()` accepts a role ledger with only 40 documents while its stored
outer census says 192.  It also accepts the same unregistered extra cell for every
document and an unregistered extra arm (and can likewise accept extra valid directed
KL pairs).  Analysis ignores those additions, so a recomputed result still validates.
This leaves an unregistered sufficient-statistics/covert-payload channel and prevents
the ledger from proving complete 192-document closure.

Required repair: for each role require exactly 192 unique document entries; exactly
the four cells `positive`, `matched_negative`, `off_target`, and `all`; exactly the six
arms in canonical order; and exactly the five directed `native -> analytical arm` KL
pairs in canonical order.  Tie the ledger document count to both the outer census and
the receipt-bound role count.  Reject every extra or missing field before analysis.

### B5. The terminal receipt is neither semantically replayed nor nonfallible after link

The receipt uses the inherited generic `write_json_create_only()`.  That helper
fsyncs the temporary bytes but never parses and compares them before linking.  A
mutation immediately before `os.link` therefore publishes corrupt receipt bytes.
After the hard link, directory fsync can raise: the receipt then exists but the
publisher reports an exception.  The outer handler correctly avoids publishing a
contradictory failure once the receipt exists, but the preregistered “receipt is the
last fallible action” lifecycle is not achieved and the receipt itself has no exact
semantic validator.

Required repair: use a dedicated exact receipt publisher that JSON-normalizes and
semantically reloads the temporary file before the terminal hard link, validates the
receipt's hashes and `passed_both_roles` against reloaded artifacts, and treats an
already-linked exact receipt as the terminal state if post-link durability handling
raises.  Add adversarial link-mutation and post-link-fsync tests proving that corrupt
receipt bytes cannot become authoritative and that no contradictory failure appears.

## Reproduced tests and counterexamples

The immutable source was extracted rather than imported from the concurrently moving
shared worktree:

```bash
audit_tmp=$(mktemp -d)
git archive f93a52f2 basis_aligned/polynomial_causal jacclust | tar -x -C "$audit_tmp"
PYTHONPATH="$audit_tmp/basis_aligned/polynomial_causal:$audit_tmp" pytest -q \
  "$audit_tmp/basis_aligned/polynomial_causal/test_induction_equality_tensor_final_ood_v2.py" \
  "$audit_tmp/basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py" \
  "$audit_tmp/basis_aligned/polynomial_causal/test_circuit_campaign_runtime.py" \
  "$audit_tmp/basis_aligned/polynomial_causal/test_circuit_campaign_statistics.py" \
  "$audit_tmp/basis_aligned/polynomial_causal/test_induction_equality_tensor_discovery.py" \
  "$audit_tmp/basis_aligned/polynomial_causal/test_circuit_induction_tensor.py"
```

Result: **37 passed in 5.15 seconds**.  These tests establish the positive components
above but do not test B1-B5.

Outcome-blind Python counterexample harnesses imported only that extracted immutable
tree and monkeypatched filesystem/model boundaries.  The observed terminal lines
were:

```text
counterexample_authority_accepts_moving_row_parent=ACCEPTED
counterexample_authority_claims_constant_over_wrong_checkpoint_snapshot=ACCEPTED
counterexample_authority_unreconciled=ACCEPTED
counterexample_40_documents=ACCEPTED
counterexample_extra_cell_and_arm=ACCEPTED
counterexample_receipt_no_semantic_replay=CORRUPT_RECEIPT_LINKED
counterexample_postlink_fsync=RECEIPT_EXISTS_BUT_CALL_RAISED
```

The authority harness replaced `_row_receipt`, `protected_snapshot`, and publication
with pure in-memory fakes, then called `freeze_authority()` and
`validate_authority()`.  The ledger harness built finite `DocumentCellSums`, reduced
the bootstrap to 50 synthetic draws, added an extra cell/arm, retained only 40
documents, recomputed the result, and called `semantic_validate()`.  The receipt
harness mutated the temporary file inside a wrapped `os.link`, then separately raised
on the second `os.fsync` (the directory fsync).  None of these harnesses accessed a
row tensor, checkpoint, model, or scientific outcome.

Reviewer: Codex independent outcome-blind reviewer.

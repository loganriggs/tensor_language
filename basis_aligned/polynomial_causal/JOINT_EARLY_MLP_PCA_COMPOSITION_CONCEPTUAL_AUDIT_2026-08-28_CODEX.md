# Joint early-MLP PCA composition: conceptual and launch audit

Date: 2026-08-28  
Auditor: Codex artifact-audit lane  
Scope: read-only review of the concurrent, untracked
`joint_early_mlp_pca_composition.py`,
`joint_early_mlp_pca_composition_development.py`, and
`test_joint_early_mlp_pca_composition.py`. This note does not authorize execution.

## Verdict

The registered causal question is coherent, but the current implementation is
**NO-GO for GPU execution**. It can become a valid authority-none, reused-curated-row
test of a *coupled oracle output-subspace interface*. It cannot establish an
executable compressed MLP0/1/2 program, independent site predictors, an MLP2
subspace, or fresh heldout/generalization performance.

The numerical core is mostly right: exact and projected arms are evaluated on the
same rows and realization; selected corrections are recomputed sequentially at each
arm's current live state; and the fraction denominators are same-run exact gains in
the same CE currency. The launch blockers are source/lifecycle closure, exact parent
and row-provenance validation, and missing runtime-semantic integrity tests.

## Evidence inspected

- Preregistration SHA256:
  `25eb5c3e8cd17ef46c9acd891164d8b893e82a7b804382fc4dacead23bcc3316`.
- Concurrent source SHA256 values at audit time:
  - pure contract: `b0dff91fa04c92c42d333997ec42222627b941b67ceb7a7668d50792e44b62ba`;
  - development runner:
    `154129bdcb198f205d801e1af5723b3cbe534f0654d9782f83a9bf9573a711e7`;
  - pure tests: `2eb77428751e76be4935cffba8e9ea79e2d7396be3cba19fc3a530527d8900fb`.
- Frozen split identities from the pinned PCA-control manifest:
  - ship fit: 480 rows / 240 documents,
    `ec422df720e8541da9e3dfa53a68689e52e57ee3a30db9983282cd2db9ed4948`;
  - PCA basis: 96 rows / 48 documents,
    `40f7a703404cb327def5beb51cbe2cb422c3f4487ea1c51b866c58b127ec763a`;
  - discovery: 192 rows / 168 documents,
    `d7230927085d997dba891d37c8ef997813a5d6fed00689726bab7c0c3200ee19`;
  - heldout: 192 rows / 192 documents,
    `f828fb2e1aab9dcbbee26fd8b2b0b3bf9c2a5a65f9683dc52f6946281a440867`.

All five curated roles are document-disjoint in the frozen manifest.

## Conceptual findings

### Split separation and PCA provenance

There is no direct heldout-to-PCA-basis leakage. `allocate_whole_document_splits`
places the 96 basis rows in 48 documents disjoint from discovery and heldout, and
the serialized local PCA bases were fit from exact residuals on that basis role.
The runner reuses those frozen rank-64 bases without refitting
(`joint_early_mlp_pca_composition_development.py:121-127,242-249`).

However, the label `heldout` must not be read as a fresh confirmatory sample. The
prior PCA singleton result and prior exact factorial both evaluated these same 192
rows, and those outcomes explicitly motivated this preregistration. The new joint
projected arms were not previously scored, so the run remains useful as
out-of-basis-row causal evidence, but it is adaptively selected reused-evaluation
evidence. It licenses neither an untouched test-set claim nor generalization.

The discovery-derived `rare_vocab` is reused on heldout
(`joint_early_mlp_pca_composition_development.py:183`), but every registered
composition decision uses `row_global_ce`, which is the unstratified mean over the
192 suffix targets. Thus this does not leak into the registered global CE decisions;
any reported rare/frequent cell metrics remain discovery-defined diagnostics.

### Intervention semantics

The runtime semantics match the preregistration. For each selected site,
`add_oracle_correction` computes

`residual = native_block.mlp(current_z) - deployed_write(current_state)`

and injects either the whole residual or its fixed output-space projection. Because
the forward visits MLP0, MLP1, then MLP2, downstream residuals are recomputed at the
state produced by that exact arm. The projected cube correctly shares only baseline
and MLP2-only with the exact cube and scores all other Boolean arms separately
(`joint_early_mlp_pca_composition_development.py:203-270`).

This is an oracle intervention, not an executable simplification. Even a projected
MLP0 or MLP1 arm calls the full native MLP to obtain the residual before projecting
it. Likewise, “exact MLP2 compensation” is only the conditional effect of restoring
the native MLP2 write at the current, upstream-projected state. It is interpretable
as a live causal positive control, but it is not:

- an independently executable MLP2 compensator;
- a learned map from available compressed state;
- evidence that MLP2 coefficients can be predicted without the native call; or
- evidence that independently trained MLP0/1/2 components compose.

A precise result label would be “native-MLP2 live-restoration marginal after the
fixed PCA0/PCA1 oracle interface.”

### CE currency and denominators

`_score_content_rows` drops target positions 0--63 and averages positions 64--255,
so each row contributes exactly 192 targets. Heldout contains one row per document;
the registered heldout row bootstrap is therefore also a document bootstrap. The
baseline and every arm have the same number of targets, so mean row CE differences
equal the corresponding pooled-token CE differences here.

The two registered fractions are dimensionally valid:

- projected MLP0+1 gain divided by same-run exact MLP0+1 gain; and
- projected MLP0+1+exact2 gain divided by same-run exact MLP0+1+2 gain.

They are fractions of *exact intervention gain relative to the deployed ship*, not
fractions of total model error, total ship residual, or a simplicity/compression
budget. The implementation correctly rejects nonpositive exact denominators and
does not import a cross-run residual denominator. It reports uncertainty only for
the two projected absolute gains, as preregistered; the 40% ratios and conditional
MLP2 marginal have no uncertainty interval and should not be described as
statistically certified fractions.

One pure-contract defect remains: CI inputs are checked for length and finiteness but
not `lower <= upper` (`joint_early_mlp_pca_composition.py:50-54`). A reversed interval
can incorrectly pass the lower-bound gate.

## GPU launch blockers

1. **No committed, pushed source closure or pre-outcome execution authority.** The
   runner and pure contract are untracked. The runtime merely records `HEAD`; it
   does not require the working bytes to equal a reachable commit. Its
   `source_hashes` omit at least `joint_early_mlp_oracle_factorial_development.py`,
   `local_ship_oracle_development.py`,
   `oracle_local_pca_strength_control_development.py`,
   `factorial_causal_attribution.py`, the model loader/runtime, and focused tests
   (`joint_early_mlp_pca_composition_development.py:139-166`). Commit and push an
   exact transitive closure, then mint a no-outcome receipt before any model load.

2. **Parent and row provenance are not fully replayed before model access.** The
   runner reconstructs splits and pins result bytes, but it does not exact-compare
   all role receipts against the PCA and exact parent manifests before entering the
   model (`:118-127`). Rowwise reproduction later is valuable but occurs after
   forwards and covers discovery/heldout outputs, not the PCA basis-role provenance.
   Require exact parent status/authority/schema/cross-links and exact equality of
   ship-fit, basis, discovery, heldout, and spare row receipts before the callback.

3. **Input TOCTOU is open.** Corpus, saved ship, and saved bases are hashed before
   `torch.load`, but not immediately rehashed after deserialization (`:103-123`).
   Hash-before/load/hash-after each file and validate the loaded tensor-tree identity
   before making the validation roles accessible.

4. **Publication is overwrite-based and not transactionally authoritative.** Both
   manifest and result use `os.replace` repeatedly (`:61-68`); an actor can populate
   a previously empty namespace after the initial check and be overwritten. The
   directory lock has no nonce/inode ownership checks (`:367-394`). Use an exclusive
   run claim, create-only stage artifacts, protected/namespace revalidation before
   every publication, and a last-written terminal receipt. A failed run must retain
   immutable partial bytes in a spent namespace rather than rewrite them in place.

5. **Finalization precedes outer closure.** The callback marks the result and
   manifest completed before `sa.main` has returned (`:292-328`). There is no
   measured callback/outer-forward call ledger, no hook restoration/inertness test,
   and no final component-tree equality after outer return. Publish terminal status
   only after the outer call returns and all hooks/shared correction state are inert.

6. **Runtime intervention integrity is untested.** Existing tests exercise only the
   scalar analyzer. Add synthetic model tests proving all eight arm maps, sequential
   current-state semantics, fixed-basis projection, baseline/MLP2 sharing, exact
   native restoration, one intended native oracle call per selected site, and
   correction-state cleanup on success and exception.

7. **Pure result validation is incomplete.** Add interval-order validation and a
   semantic final-result validator that recomputes all gains, ratios, conditional
   marginals, decisions, and bootstrap inputs from the stored paired row ledgers.

## Claim boundary after repair

A passing repaired run would establish only that, on this frozen realization and
these reused curated evaluation rows, the fixed rank-64 MLP0/1 residual-output
subspaces retain the registered share of live-oracle CE gain when used jointly, and
that native MLP2 restoration remains beneficial after them. It would justify a new,
separately preregistered coefficient-predictor experiment.

It would not establish parameter compression, lower multiply/storage price,
native-call removal, independent site executability, an MLP2 subspace, intervention
modularity under independently learned errors, fresh-corpus behavior, or OOD
generalization.

## Pure checks

The following read-only suite passed at audit time:

```text
pytest -q \
  basis_aligned/polynomial_causal/test_joint_early_mlp_pca_composition.py \
  basis_aligned/polynomial_causal/test_joint_early_mlp_oracle_factorial.py \
  basis_aligned/polynomial_causal/test_factorial_causal_attribution.py

12 passed in 0.18s
```

No GPU job, checkpoint/model forward, runner execution, or artifact mutation was
performed during this audit.

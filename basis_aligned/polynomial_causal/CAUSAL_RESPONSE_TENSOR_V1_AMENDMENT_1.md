# Causal-response tensor v1 — prospective amendment 1

Frozen after the outcome-blind audit of commit `94fe6016` and before any v1 bilin18
model load, forward, direction, response, or protected outcome.  The original
preregistration remains immutable evidence of the defect; this amendment controls.

## Why an amendment is necessary

The original analysis boundary correctly said that response-factor candidates would be
fit on FIT documents and evaluated on EVAL documents.  The audited backend, however,
fit only activation directions on FIT and collected response cells only on EVAL.  It
therefore produced no lawful FIT response tensor: factor selection would either be
impossible or would leak EVAL.

This is a launch-fatal scientific mismatch.  No v1 outcome has been opened.  We repair
it prospectively with two separately published stages.

## Controlling two-stage protocol

### Stage 1: FIT program and FIT responses

Use exactly the frozen FIT source documents to:

1. make one native capture sweep and estimate the `full`, shared, and `residual`
   directions exactly as in the original preregistration;
2. make one separate native CE sweep;
3. make one sweep for each of the 2 phases and 49 source interventions;
4. aggregate all 49 target cells per source document;
5. publish a create-only **FIT program bundle** containing directions, fit counts,
   exact circuit ordering, masks/support hashes, FIT document IDs, FIT response
   sufficient statistics, validation, and its exact physical-call ledger;
6. semantically reload the bundle and verify every hash, dtype, shape, ordering,
   support, statistic, and ledger before EVAL can be opened.

The exact source/target ordering is: component order
`a8, a16, m16, a3, m14, m13` from the frozen split, then lexicographic circuit tag
within component.  The ordered `(component, tag)` text has SHA-256
`86d0bd7250102fc8dcdee517562fcadda74f2f6bf6d026582bcab71a33f24ca0`;
its exact UTF-8 serialization is one `${component}\t${tag}\n` line per pair.

The frozen role contains 496 rows, hence 124 four-row batches.  With 49 sources and
two phases, the exact FIT outer-forward census is

\[
124\,[1\text{ direction-capture}+1\text{ native CE}+98\text{ interventions}]
=12{,}400.
\]

Every outer forward must call each of the 18 native attention and 18 native MLP sites
exactly once.  Direction capture occurs at each of the six registered owner components
once per capture batch: (6\times124=744) component-capture events.  Projection occurs
at exactly one source component in every intervention batch:
(2\times49\times124=12{,}152) projection events, keyed by phase, source tag,
component, and batch—not merely aggregated by component.

### Stage 2: EVAL responses

Stage 2 must start in a new process/collector after the FIT bundle exists and passes
semantic reload.  It may load directions only from that bound bundle; it may not
refit, renormalize, reorder, choose a sign, or inspect FIT activations again.  It then:

1. makes one native CE sweep over the frozen EVAL documents;
2. makes one sweep for each of the 98 frozen phase/source interventions;
3. aggregates all 49 target cells per source document;
4. publishes a create-only EVAL result, exact ledger, manifest, and receipt last.

The frozen role contains 504 rows, hence 126 four-row batches.  The exact EVAL
outer-forward census is

\[
126\,[1\text{ native CE}+98\text{ interventions}]=12{,}474.
\]

Each native site therefore has exactly 12,474 calls.  There are exactly
(2\times49\times126=12{,}348) projection events, with the same structured keys as
FIT.  FIT and EVAL together use 24,874 outer forwards, but their ledgers and terminal
artifacts remain separate.

Factor family/rank/sparsity selection uses only the published FIT response tensor.
After a candidate is frozen and reloaded, EVAL scores response prediction and literal
price.  EVAL is never used to choose among candidate structures.

## Frozen numerical currency

- checkpoint parameters and native model execution: float32;
- RMSNorm, component writes, rank-one inner products, projection subtraction, logits,
  softcap, and unreduced cross-entropy: float32 in the facade's written order;
- FIT member/off write sums, means, normalization norms, and component SVD: CPU
  float64;
- normalized fitted directions: cast once from CPU float64 to float32 and frozen in
  the FIT bundle;
- per-position float32 cross-entropy is cast to CPU float64 before document sums;
- all published response sums are float64 and counts are int64, without permissive
  dtype coercion on reload.

For each component SVD, signs are fixed exactly as in the original preregistration.
If (\sigma_1\) is nonfinite/nonpositive or, for the existing multi-circuit components,

\[
\frac{\sigma_1-\sigma_2}{\sigma_1}\le 10^{-6},
\]

the FIT stage fails because the shared direction is tied/unstable.  The existing
zero/full-residual norm failure rules remain unchanged.

## Additional controlling integrity requirements

The execution wrapper, not a caller, must reconstruct and seal the exact 49 specs and
496/504 row roles from hash-bound parents.  It must reject negative, duplicate,
out-of-range, or caller-supplied roles; clone all tensors into owned storage; and permit
each collector exactly one stage call.  It must verify zero forward/pre-hooks and exact
model state/config/checkpoint hashes before and after each stage.

Validators must reject unexpected statistic keys, dtype changes, nonzero values in
unsupported member slots, incomplete structured ledgers, and any drift in frozen
parents/source.  Authority is create-only and precedes parent `torch.load` and model
load.  An inode/nonce claim, stable before/after reads, mutually exclusive guarded
failure, semantic reload, final source/parent/model replay, and receipt-last publication
are mandatory.

The outcome-blind audit is a NO-GO, not an authorization.  Stage 1 may run only after
the revised backend/lifecycle and attack tests are committed, pushed, and independently
audited GO.  Stage 2 needs a separate GO over the sealed FIT bundle and EVAL wrapper.

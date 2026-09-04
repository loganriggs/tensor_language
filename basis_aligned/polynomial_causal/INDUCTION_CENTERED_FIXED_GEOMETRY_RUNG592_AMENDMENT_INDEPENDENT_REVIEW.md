# Independent review: R592 centered fixed-geometry amendment

Date: 2026-09-04 UTC

Reviewed commit: `eaeee8e7cd728a345a5e24421ab6aeccef4fefae`

Reviewed amendment SHA-256:
`5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094`

Verdict: **BLOCKED before implementation**

The amendment resolves the scientific and mathematical ambiguities in the first
three repair groups and gives an exact complete-phase evidence contract. One
implementation-determining ambiguity remains in partial invalid-diagnostic
publication. It permits stopping midway through a directed chunk, but only
defines rectangular arrays whose arm axis is complete. Two incompatible
diagnostic encodings therefore satisfy the prose, and some allowed failures
cannot be encoded without forbidden padding.

This review used immutable specification/code blobs and small CPU fixtures. It
did not inspect an R592 implementation or outcome, load a model, open CUDA, use
a GPU or queue, edit the amendment, or touch R590.

## Exact authorities

The amendment correctly binds the blocked specification and its review, along
with the inherited R585/v7 authorities:

| authority | SHA-256 |
|---|---|
| amended R592 specification | `5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094` |
| blocked R592 preregistration | `870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a` |
| blocking review | `9b76b91995374697b8a828ce042e59d81bfddcbaa5f6e843cb0f32f6b01e57f7` |
| blocking five-test packet | `7356aebd017ba6c6c5ce92176ff95fbffd01d5924b5b7d4cc91dd90e0618b07c` |
| R585 replacement amendment | `98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf` |
| R585 row/scoring manifest | `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962` |
| v7 operational-scope handoff | `595b43156117e0ba2e568972f76af81ac4e716ed5537861ac48f13b23d4ed9fd` |

Regenerating the outcome-blind R585 manifest gives the stated 1,872/936 rows,
1,728/864 endpoints, 3,744/1,872 directions, 13,824/6,912 endpoint-site-role
operations, 20 target and 32 control cells per split, 32 structural identities
per split, and 124 bootstrap cells per split.

## Repair 1: instrument-gate supersession is now coherent

This repair passes. The amendment separates inherited scientific gates from
replaced numerical instrument checks:

- native full-write reconstruction, native equality plus non-equality
  remainder, semantic support/census, finiteness, planned-versus-actual hook
  delta, and full-output structural identities remain hard `1e-5` checks;
- factorized `B(E_x,U_x)` versus the differently contracted native equality
  term is descriptive for R592 and blocks only the unclaimed literal
  remove-and-insert level;
- padded/unpadded and length-sorted comparisons are deleted;
- fixed width 30, exact token manifests, and same-tensor directed native versus
  literal-zero replay replace them.

The eight new invalid predicate IDs are literal and exhaustive. The old
`canonical_term_failure`, `factor_mismatch`, and `padding_failure` identifiers
are expressly forbidden from an R592 terminal. This preserves the tolerance
while changing only which mathematical equality it applies to.

## Repair 2: machine arm and bootstrap identity is fixed

This repair passes. Machine keys remain exactly:

```text
replay, score, payload, joint
```

Only display labels say coefficient or projected content. The R585 manifest
still regenerates the namespace
`a8-r585-replacement-group-bootstrap-v1`, the same 248 cell IDs, and the same
SHA-defined draws. Scale lookups, structural identities, evidence joins, and
failure clauses therefore retain their prospective identity. A mixed renamed
machine namespace is explicitly illegal.

## Repair 3: the mediator and transport object is exact

This repair passes. Every scientific delta uses the native width-30 endpoint
cache:

```text
replay = 0
score = B(E_y,U_x) - B(E_x,U_x)
payload = B(E_x,U_y) - B(E_x,U_x)
joint = B(E_y,U_y) - B(E_x,U_x)
```

The replay is constructed by `zeros_like`, not subtraction. No live equality
term is removed. Later sites retain the endogenous response to earlier changes
and receive only the fixed cached delta; the two layer-8 heads share one
pre-modification state. The directed native observer checks recipient `e` and
`u` componentwise and reconstructs the three recipient-containing hybrids.
The reverse direction supplies the donor-as-recipient check. This closes the
equal-self/unequal-hybrid counterexample from the first review.

Activity is now the median across the four sites of the norm of the actual
centered delta. It is no longer an inserted-minus-live-removed quantity, so the
known factorized/native contraction discrepancy cannot silently alter control
activity or FIT scales.

## Repair 4: complete evidence and price pass, partial diagnostics do not

The complete-phase schedule and evidence contract are exact:

| split | endpoint calls | directed chunks | five calls/chunk | total |
|---|---:|---:|---:|---:|
| FIT | 54 | 117 | 585 | 639 |
| SELECT | 27 | 59 | 295 | 322 |

The SELECT direction tail is exactly 16 rows and cannot be filled. Full
execution is 961 forwards, zero backwards, and zero updates. FIT must completely
hold before SELECT opens; FINAL/OOD stay closed.

The main raw-array arithmetic also reconstructs exactly:

- FIT/SELECT logit differences: 3,010,595,328 / 1,505,297,664 bytes;
- FIT/SELECT hook deltas: 276,037,632 / 138,018,816 bytes;
- FIT/SELECT directed live projected content: 138,018,816 / 69,009,408 bytes;
- stated principal total: 5,136,977,664 bytes.

For a complete FIT or SELECT phase, `factor_e`, `factor_u`, live factors,
planned-versus-actual hook deltas, and native/replay/three-arm raw vocabulary
differences suffice to reconstruct transport, hook, structural, vocabulary RMS,
and native/replay comparisons. Complete scientific nulls cannot omit these
arrays or pretend that a short array is complete.

### Remaining blocker: a mid-chunk invalid diagnostic has no legal exact shape

The terminal section says a runtime failure may stop at the first failing
completed **call or chunk**, and that partial arrays are never padded. But the
only hook/logit schemas have fixed complete arm axes:

```text
hook_deltas       [direction, 4 machine arms, 4 sites, 1152]
logit_differences [direction, 4 difference arms, 50257]
```

Consider the first directed chunk after its native, replay, and score calls.
If score produces `nonfinite_observation` or `centered_hook_delta_failed`, the
call-level rule permits immediate publication. Payload and joint do not exist.
Writing a first-axis direction row with four arm slots pads missing evidence,
which is forbidden. Writing a two-arm second axis changes the declared schema.
Dropping the incomplete chunk loses the evidence needed to derive the first
failing predicate. Continuing through payload and joint instead chooses
chunk-level stopping and executes calls after the first detected failure.

The prose also says the diagnostic auditor reconstructs both prefix and failure,
but does not define a per-arm event table, a ragged prefix array, or raw encoding
for a nonfinite failing call. Therefore it does not uniquely determine:

1. the exact executed-call prefix for a mid-chunk failure;
2. each diagnostic array's legal shape and row order at that prefix;
3. whether the failing call itself counts as completed evidence;
4. how `nonfinite_observation` is retained while saved values are required to
   be finite; and
5. whether later calls in the same chunk are forbidden or required.

This is the same class as the original review's fourth blocker, narrowed to
invalid diagnostics. Complete normal-result evidence is no longer ambiguous.

## Required narrow repair

Freeze one of these policies prospectively:

1. **Exact call-prefix evidence:** store completed directed arm outputs as an
   ordered per-call/per-arm table (or arrays with arm as the first prefix axis),
   specify shapes and bytes as functions of the manifest prefix, and define the
   raw failing-call record for every one of the eight predicates; or
2. **Complete-chunk diagnostics:** evaluate validity only after all five calls
   of a directed chunk, require every publishable directed prefix to end at a
   chunk boundary, and separately classify failures that cannot safely finish
   a chunk as hard aborts with no diagnostic artifact.

In either policy, state whether nonfinite raw arrays are legal diagnostic
evidence or whether nonfinite is an unpublishable hard abort. Bind exact
diagnostic result/evidence/receipt names and require receipt-last atomic
publication. No row, scientific gate, threshold, factor formula, bootstrap,
claim boundary, or complete-phase price needs to change.

## Adversarial packet

`test_induction_centered_fixed_geometry_rung592_amendment_review.py` reports
`10 passed, 1 strict xfailed`.

The passing checks bind every authority, reconstruct the manifest and legacy
bootstrap namespace, test gate supersession, plant equal-self/unequal-hybrid
transport, verify frozen centered algebra and activity, reconstruct call/byte
prices, reject missing native differences and fake complete partial bytes, and
reject premature SELECT/mixed arm/live-removal semantics. The strict expected
failure is the missing unique unpadded schema for a mid-chunk diagnostic.

## Decision

**BLOCK exact amendment commit `eaeee8e7cd728a345a5e24421ab6aeccef4fefae` from implementation.**
The scientific object, legacy identity, factor transport, complete evidence,
and 961-forward design should be retained. A narrow prospective diagnostic-
prefix amendment can close the remaining block. It must then receive a new
independent exact-byte review before implementation begins.

# Reusable circuit causal-validity critic checklist

<!-- BQLANE: cpu -->

Use this before a circuit builder runs a model and again before accepting its
result. A pass at one level never licenses the later levels automatically.

## 1. Freeze the object being tested

- Pin the dataset, semantic metadata, producer, tests, dry run, dependencies,
  checkpoint, thresholds, operation manifest, and bootstrap manifest by exact
  path and SHA-256.
- Materialize both the expected operation list and the runtime-realized list.
  Compare the ordered lists, counts, and hashes; an expected hash copied into a
  result is not runtime evidence.
- Record every split opened, model forward/backward, update, and output path.
  Decide FIT completely before opening SELECT, and never use held-out results to
  repair a frozen rule.

## 2. Validate the counterfactuals themselves

**Meaningful donor/recipient counterfactuals** change the registered causal
variable while directly verifying the variables claimed held fixed from token
sequences and semantic positions. The recipient is the state being changed;
the donor supplies only the registered factor. Test both physical directions
when direction changes the causal meaning. Reject identical endpoints,
ambiguous token-based role lookup, changed “held-fixed” fields, and donor labels
that disagree with actual factor provenance.

**Multiple valid counterfactuals** means more than two directions of one prompt.
Each cell should contain group-disjoint lexical, layout, and length variants,
with exact per-cell counts. Report the diversity census and show the effect is
not carried by one template, token pair, or direction.

## 3. Keep the claim ladder separate

1. **Interaction isolation:** replay, factor A, factor B, and joint states must
   share the same group and primitive outcome coordinate. Recompute the
   factorial interaction from all four outcomes. Opposing single-factor
   predictions, exact no-op identities, and active unrelated controls are
   needed to distinguish an interaction from general task difficulty.
2. **Held-out prediction:** all choices come from FIT, SELECT groups are
   disjoint, SELECT is opened only after FIT passes, and every realized SELECT
   cell matches the frozen manifest.
3. **OOD prediction:** use a separately preregistered distribution shift, no
   basis or threshold refit, and predictions written before opening OOD. A
   group-disjoint in-distribution SELECT result is not OOD evidence.
4. **Sufficiency:** donor-directed interchange recovers the target behavior
   with a valid natural denominator and full-vocabulary/CE checks. Sufficiency
   does not imply necessity.
5. **Selective removal:** a separate live-term removal must reduce the target
   behavior while preserving adequately active unrelated circuits and broad
   model behavior. Interchange alone is not removal evidence.
6. **Composition/reuse:** the same identified factor must make frozen
   predictions in multiple registered compositions and beat an independent
   per-context memorization baseline. Similar effects in several modules do not
   by themselves show reuse.
7. **Stable identification:** independent seeds/resamples should recover the
   same equivalence class after quotienting known gauge freedoms, and held-out
   interventions should select that class. Low reconstruction error from one
   fit is not identification.

## 4. Require audit-ready primitive evidence

- Save endpoint factors once and row/arm/site references separately, with exact
  shapes, dtypes, byte counts, row-order hashes, membership maps, and finite
  checks.
- Recompute score, CE, vocabulary RMS, recovery numerator/denominator,
  interaction, activity, and controls from primitive values. Do not trust only
  aggregate pass flags.
- Independently compute the canonical term and its complement. Subtracting a
  proposed term from a head output and adding it back is circular.
- Use unit-matched frozen scales. Keep residual-vector norms, logit margins, and
  vocabulary-logit RMS distinct.

## 5. Crash and tamper before approval

- Inject failure during evidence writing, after evidence, after result writing,
  and during publication. After every exception, either the complete validated
  package exists or no final artifact exists.
- Write into unique same-filesystem staging paths, validate and flush there,
  then atomically publish. A retry may clean stale staging but must never delete
  a valid completed outcome.
- Tamper one operation, bootstrap cell, evidence row, array shape, order hash,
  checkpoint hash, result path, receipt byte, and finite value. Each mutation
  must fail closed.

## 6. Exact handoff to the next two agents

The machine-readable companion is
`basis_aligned/bilinear_quotient/ops/circuit_causal_validity_next_wave_handoff_rung585.json`.
The builder and critic must return every field in
`next_agent_handoff_required_fields`, preserve the claim-by-claim status in
`claim_ladder`, and report every `required_test_id`. Missing fields mean
**BLOCK**, not “unknown but acceptable.” The builder authors the repair; the
critic independently reconstructs manifests, counterfactuals, metrics, and
crash behavior rather than importing builder decision functions.

## Five-part knowledge packet

1. **Pattern:** literal semantic datasets plus independent primitive-evidence
   reconstruction work; metadata-only expected hashes do not.
2. **Mapping:** endpoint → semantic role → site factor; directed row → explicit
   recipient/donor endpoint; arm → fixed factor provenance; same-layer sites →
   one pre-intervention state.
3. **Smallest exact term:** one site/role product $e_h(r)u_h(r)$, checked against
   independent equality and non-equality contractions.
4. **Controls:** require actual intervention activity, unit-matched thresholds,
   multiple unrelated control families, and all groups in each active cell.
5. **Residual risk:** an operational, all-site, oracle-supported factorization
   can still fail OOD, removal, reuse, and stable-identification goals. Keep
   those claims explicitly `not_tested` until their own preregistered tests.

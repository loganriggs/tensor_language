# R581 result review: R580 induction native capability

**Reviewed:** 2026-09-03 UTC

## Verdict

**Do not canonicalize R580 yet under its frozen audit protocol.** The scientific result is strong and exactly reproducible from the saved evidence, but the saved R581 artifact has terminal verdict `failed_independent_audit`. Its sole failure is `envelope.next_step:value`: R580 serialized `next_step` as a one-element JSON list, while the frozen R581 auditor requires the same text as a JSON string. This does not alter any model measurement or scientific gate, but R581 preregistered that *any* envelope mismatch fails the audit. Canonicalizing despite that failure would bypass the stopping rule.

This is a **secondary audit**, not an independent replication: I authored the R580 implementation and focused tests, although I did not author R581 or the model run. The saved R581 reconstruction is the independently preregistered audit.

Subject to resolving the envelope failure with an explicit, provenance-preserving integrity procedure, the evidence supports only the narrow claim that the unmodified model has the repaired selector-by-payload behavior on FIT and SELECT. It does not identify a causal model site, establish a reusable decomposition, test selective removal, or open FINAL_TEST/OOD.

## Independent recomputation from saved evidence

I reconstructed the expected prompt set directly from the frozen R578 groups and rows, without importing either scorer. All checks below held:

- Authority census: 108 groups (72 FIT, 36 SELECT), 3,240 rows (2,160 FIT, 1,080 SELECT), and 3,024 unique prompt sequences (2,016 FIT, 1,008 SELECT).
- Every saved sequence ID, group, split, length, final position, answer token, and B/D token pair matches the R578 authority; there are zero missing or duplicate IDs.
- Both saved CE values obey `CE(token) = log_normalizer - logit(token)` exactly at saved precision; maximum absolute error is 0.
- All 3,240 row records reconstruct exactly from the sequence logits and R578 row definitions.
- All 108 factorial records and all 432 selected/neutral/contrast records reconstruct exactly.
- Every reported aggregate matches the recomputation. All 86 SHA-defined, 2,000-replicate bootstrap cells reproduce, including every draw-matrix hash and statistic-vector hash. The combined trace hash is `2f7d3ad3dc2eb3779722c586e29b4f9ac4865b0d297875a5bc9b05c4cc75571f`.

The gate results are:

| Frozen gate | FIT | SELECT | Result |
|---|---:|---:|---|
| Four factorial cells: worst positive-margin fraction | 77.78% | 80.56% | pass (threshold 75%) |
| Four factorial cells: worst bootstrap lower bound | 1.3432 | 2.1325 | pass (>0) |
| Selector-by-payload interaction: mean | 3.3915 | 3.5387 | — |
| Selector-by-payload interaction: lower bound | 3.0527 | 3.0627 | pass (>0) |
| 32 control endpoint cells: worst positive-margin fraction | 76.39% | 80.56% | pass (threshold 75%) |
| 32 control endpoint cells: worst lower bound | 1.0610 | 1.9753 | pass (>0) |
| Selected-match removal: positive-drop fraction | 97.92% | 98.61% | pass (threshold 70%) |
| Selected-match removal: mean drop / lower bound | 3.3179 / 2.9923 | 3.5729 / 3.0398 | pass (>0) |
| Selected versus neutral: mean gap / lower bound | 2.6266 / 2.3077 | 3.0303 / 2.5432 | pass (>0) |

Thus all three runner predicates recompute true and the scientific verdict recomputes as `held_capability_screen`, with no failed scientific clause. The non-gated contrast-source mean changes range from -0.1450 to 0.0312 on FIT and -0.0279 to 0.0798 on SELECT; their cellwise confidence-interval envelope is [-0.3483, 0.2117] and [-0.3114, 0.3284], respectively.

## Counterfactual liveness

R580 contains prompt counterfactuals, not internal activation interventions, so activation-intervention liveness is not applicable. All 3,240 prompt pairs use distinct base and donor sequences. Every pair changes the measured correct-answer margin by more than `1e-12`. In particular:

| Counterfactual family | Rows | Minimum absolute margin change | Median absolute margin change |
|---|---:|---:|---:|
| selected-match break | 432 | 0.02653 | 2.99190 |
| neutral-source edit | 432 | 0.00213 | 0.33989 |
| neutral-payload edit | 432 | 0.00177 | 0.36421 |
| filler/lag controls | 864 | 0.00082 | 0.88335 |
| contrast-source edit | 432 | 0.00110 | 0.45404 |

The remaining answer-changing factorial families are also live on every row. This verifies that no counterfactual silently reused the identical prompt or produced an exactly unchanged margin; it does not promote those edits to a localized causal circuit.

## Hash, split, and price audit

- R580 result SHA-256: `7c7463a95931a51cd848ff9e8033bed77a26f7889a1a5fd1a3512ec2d1224b84`; the receipt binds these exact bytes.
- R580 receipt SHA-256: `6a1ef728bca424ed27ec145adad1918923e91f190b96a9ff452b6838413b670a`.
- R581 audit SHA-256: `8ecc1562632212ee876a794377e31966776ec15de02b5cb8d31798e438502cdb`.
- All ten authority files in R581 match their pinned hashes. The R580 implementation, focused test, preregistration, and dry-run hashes are respectively `62d11395...a73249`, `9f166a61...c64550`, `8f80926d...d12580`, and `3d21b629...d68588`.
- The checkpoint file itself hashes to the reported and frozen `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
- Saved evidence contains only FIT and SELECT. No raw sequence, row, factorial, or condition-effect record belongs to FINAL_TEST/OOD. The four split token banks are pairwise disjoint, and the result and receipt both report no forbidden split opened.
- The implementation evaluates each of 3,024 unique prompts once in batches of 32: `ceil(3024/32) = 95` model calls, with a final batch of 16. The result and receipt report 95 forwards, zero backwards, and no weight update.

The R580 run log is nine lines containing only its terminal JSON summary. The R581 `.2` log is eleven lines containing only its audit summary. Neither contains a traceback, warning, nonfinite value, foreign-rung output, or evidence of a second model run. The `.2` filename is a runner naming detail, not content contamination.

## Exact audit failure and required disposition

R580's source constructs `next_step` with a trailing comma inside parentheses, so Python makes a one-element tuple and JSON writes:

```json
"next_step": ["independent_CPU_audit_then_separate_R557_R558_adaptation_preregistration"]
```

R581 expects the same value as a scalar string. Its focused fixtures use a scalar string, which is why the test suite passes while the real result fails. This is an instrument/test coverage defect, not a scientific-gate failure.

The safest disposition is to preserve the original R580 result and receipt unchanged, record this review, and run a separately frozen integrity-repair audit that explicitly treats the singleton-list representation as the known serialization defect while requiring every scientific value, raw record, hash, split, and price check to remain unchanged. Do not silently edit the result or retroactively weaken R581. Canonicalization should wait for that held repair audit.

## Verification command

```bash
/venv/main/bin/python -m pytest -q \
  basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_native_capability_rung580.py \
  basis_aligned/bilinear_quotient/ops/test_audit_induction_selector_payload_native_capability_rung581.py
```

Result: `17 passed in 7.31s`. This confirms the current instruments' focused tests; as noted above, those tests did not include the real runner's singleton-list serialization case.

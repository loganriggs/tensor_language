# Larger-panel follow-up for learned local quartic corrections

22 September 2026, 04:21 UTC. Implementation complete, synthetic export-assembly check passed. The native learner remains queued; no learned result or follow-up score is available.

Consume all four frozen artifacts from LOCAL_QUARTIC_RESIDUAL_NATIVE_V1: Adam/Muon, seeds 25001/25002. Preserve its registered original-panel verdict. Evaluate every artifact on all 16,384 opened FineWeb states and 2,494 fixed same-token/position pairs; do not pick a winner before scoring. Report original output-coordinate errors, pooled and small-output RMS errors, and 256 document-level small-output RMS errors.

Verify each artifact hash against its terminal receipt, and its fixed CP1001 parent hash. Reproduce original-panel small-value/response scores within 1e-4 absolute tolerance. The GPU computation before export used float64 and the artifacts contain float32 parameters, so compare with a stated tolerance, not exact equality. Require outputs 0–3 to stay bitwise identical to the parent in the CPU assembly. The correction belongs only to outputs 4–15 and cannot fix coordinate-1 removal failures.

The preflight independently expands the packed 12-by-8 readout into a dense 12-by-96 coefficient matrix. Its prediction agrees within 1.3e-16 relative error, and protected outputs remain identical. Existing native helpers and queued scripts are unchanged.

Descriptively check whether both starts of either optimizer retain at least 15% improvement in small-output values and responses on the larger panel. This panel is already opened and the rule is a follow-up to the original registered screen, not a newly untouched confirmation. Preserve absolute errors so relative gains cannot disguise inaccurate components. No fitting, native intervention, semantic naming or deployment is authorized by a passing numerical comparison alone.

Run after the native result becomes terminal:

```bash
/venv/main/bin/python basis_aligned/polynomial_causal/direct_tensor_match/audit_local_quartic_followup.py
```

Do not rerun or modify the native learner merely because its result is still pending. Its measured duration and optimization history must be interpreted before attributing a negative result to the model class.

# R591 repaired-candidate independent pre-execution review

Date: 2026-09-04 UTC  
Reviewed commit: `a5e1dd022729c28dad99c1782f557b3162cdf45e`  
Verdict: **APPROVED for the registered diagnostic execution only**

This review used immutable Git blobs and CPU-only planted data. It did not load
the model, open CUDA, inspect an R585/R591 outcome, enqueue work, or create a
scientific namespace. Approval does not license R585 science and cannot turn an
R591 diagnostic classification into a scientific circuit result.

## Exact approved packet

| Artifact | SHA-256 |
|---|---|
| producer | `fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc` |
| producer owner test | `8a24a9903d10ada8a4048c7adcb33cb4ef3e8aeef11d6f9718f8e50e57b6212c` |
| deterministic dry run | `8a6331fb1a4d3800abff5ab6b7e291105872b06b41a43b003436312b6e50dc5d` |
| amended preregistration | `2dd8f918f767a6e5d91af357cfaa14770b79334ebac837d1bf52e8046ce190a5` |
| builder handoff | `202f1268e583a82f6cca385f4223b6edf4e8f8bbaee2c1cc975b09e51cd95f12` |
| managed adapter | `b0a0654c4b6fd28a9dfbfb947969049c203ef346cc580f87f5406701ac876d20` |
| adapter owner test | `338dd545838e75ae8de4a8bd6405f4bac601fe2ad8a81f594bab8104151de0ed` |
| managed-adapter handoff | `fab59548fd9529371f06156bbf2f9fa69c2c33a8a41abe2acb47a4780ff0ea96` |

## Closure of the four previous blockers

1. **Native-only causal labels:** padding is activated only by
   `N(L_30)-N(L_native)`, and membership/GEMM only by
   `N(M_30)-N(L_30)`. Planted R-only padding and F-only membership deviations
   remain descriptive and do not relabel the diagnostic. Planted N deviations
   activate exactly their registered classes.
2. **Retained panel support:** the dry run emits all 256 ordered FIT endpoint
   IDs, their direct ordered-list hash, the length-plus-ID membership hash, the
   FIT label, and the exact `64 × {19,20,27,28}` census. The IDs are unique and
   match an independent reconstruction from the pinned authority.
3. **Outcome-blind dry run:** endpoint authority is rebuilt from captured R578
   rows and the semantic manifest while the R585 outcome-dependent verifier is
   replaced only for this authority construction. A planted read trap on the
   R586 result, R586 receipt, and R587 audit remains untouched throughout the
   complete dry run.
4. **Immutable managed execution:** the adapter hashes the candidate and all
   producer-declared local dependencies. It embeds the verified producer bytes
   in the isolated Python command, so changing the producer pathname after
   command construction does not change the executed code. The producer then
   captures every local executable dependency before importing it. Shared
   handoff v6 is hash-bound.

## Frozen computation and boundary

The call manifest independently reconstructs exactly 234 forward calls:

- N: 132 batches;
- F: 24 batches;
- R: 78 batches;
- 7,488 endpoint-forwards;
- 26,112 endpoint-by-site-by-role factor operations;
- FIT only, with SELECT/FINAL/OOD absent;
- zero backwards and zero parameter updates.

F returns the untouched native attention-write object. R clones the native
write and adds only `term - canonical` at L5H5, L7H3, L8H3, and L8H4. Model and
checkpoint validation remain explicit, while the variable `(32, length)` token
shapes use the non-fixed facade check. The absolute `1e-5` threshold is unchanged.

The diagnostic has no result, receipt, evidence, score, selection, or
publication call. All conventional R585/R591 scientific namespaces were absent
at review. Its real output is one strict-finite JSON object on stdout with
`diagnostic_only_no_scientific_terminal` status. The source preflight's warning
about tuple equality is not applicable: that tuple compares a tensor shape, not
dictionary or JSON iteration order.

## Verification

- Candidate owner, adapter, v5, and v6 suites: `37 passed`.
- Independent exact-byte attacks: `11 passed`.
- Producer and adapter static gates: pass with no findings.
- Adapter preflight: no findings.
- Managed dry run: 234 registered future forwards, zero actual forwards, and
  `different_agent_review_required`.

Independent test:
`test_induction_replay_native_numerics_rung591_repair_independent_review.py`,
SHA-256
`eb295111a7230e31434275f2191250a94793ea6b129146dbb70905df46ae6f69`.


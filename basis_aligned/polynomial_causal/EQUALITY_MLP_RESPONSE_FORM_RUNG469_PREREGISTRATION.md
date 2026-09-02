# Rung 469 preregistration: quadratic response form versus state conditioning

Registered before computing any rung-469 gradients, response forms, state forms, or removal outcomes.

## Fixed scope

- Modules: MLP8, MLP9, MLP12.
- Equality sources: native L8H4 matcher (`N`) and the frozen L5H5-score transplant (`H`).
- Context cells, in fixed order: near repeat, far repeat, one previous occurrence, multiple previous occurrences.
- Code discovery: documents 0:96. Code validation: 96:192.
- Natural validation: the two fixed waves 0:96 and 96:192 from the already-open `final_natural` role.
- No new row role, attention-0 SEALED result, product-index selection, threshold tuning, rank search, or semantic label
  enters this rung.
- Literal deployed price: zero parameters saved and zero added. This is an identification test.

The exact objects and identity are derived in `EQUALITY_MLP_RESPONSE_FORM_RUNG469_DERIVATION.md`.

## Fixed calculations

For every module, source, cell, and data window, accumulate:

1. the mean 1,152-dimensional loss gradient at the MLP output;
2. the mean 1,152 by 1,152 quadratic reader `Q`;
3. the mean 1,152 by 1,152 equality-induced state form `S`;
4. the complete local first-order removal response `-E[<Q_i,S_i>]`;
5. its mean-form part `-<E[Q],E[S]>` and covariance remainder; and
6. the actual CE change after replacing all 4,608 product activations of each MLP, and of all three together, by
   their same-document equality-absent values while downstream layers recompute.

On code discovery only, fit one least-squares scalar per source and target (`MLP8`, `MLP9`, `MLP12`, union) from the
four local-response cells to the four exact causal-effect cells. Freeze those eight scalars. For every later window:

- `cross-form prediction = frozen code Q` contracted with that window's `S`;
- `causal prediction = frozen scalar * cross-form prediction`;
- naive control = the corresponding four exact code-discovery causal effects copied unchanged.

No target-window gradient or target-window CE outcome enters the cross-form prediction.

## Registered predictions

### A. Instrument and identity

- All frozen hashes, row identities, source scales, support counts, and expected dispatch counts match.
- Native versus analytical replay relative-squared error is at most `1e-12`.
- Exact MLP product reconstruction relative-squared error is at most `1e-10`.
- For every reported window/source/module/cell, mean-form plus covariance reconstructs the accumulated local response
  to relative error at most `1e-7` (absolute tolerance `1e-9`).
- No empty/full patch, source arm, or gradient capture is skipped; SEALED attention0 remains closed.

### B. The full quadratic reader is stable beyond product indices

For at least four of the six module/source pairs, the four-cell stack of `Q` matrices has cosine at least `.90`
between code discovery and code validation and at least `.75` between code discovery and each natural wave. The
same qualifying pairs must be present in both natural waves.

### C. The transfer failure has a reproducible location

For each source and natural wave, take the median code-to-natural cosine across MLP8/9/12 separately for `Q` and `S`,
and the median absolute covariance fraction `|covariance/local response|`.

- `state_shift`: reader median exceeds state median by at least `.15`;
- `reader_shift`: state median exceeds reader median by at least `.15`;
- `coupling_shift`: both medians are at least `.75`, while the median covariance fraction differs from code discovery
  by at least `.20`;
- otherwise `mixed_or_unresolved`.

Prediction C holds only if both sources receive the same non-unresolved label in both natural waves. This is a frozen
classification, not a search for the best module.

### D. Code reader plus target state predicts the target local computation

Sum the three module predictions before scoring. On code validation and both natural waves, under both sources, the
cross-form four-cell vector must match the actually accumulated local first-order union response with cosine at least
`.80` and projection onto the target between `.25` and `1.75`. It must also reduce normalized L2 error by at least
`10%` relative to copying the code-discovery local-response vector unchanged.

### E. The same object predicts the exact causal effect

After applying only the frozen code-discovery scale, the union prediction must match the exact complete-three-MLP
removal vector with cosine at least `.75` and projection between `.25` and `1.75` on code validation and both natural
waves under both sources. It must reduce normalized L2 error by at least `10%` relative to copying the code-discovery
exact causal vector. At least two individual MLPs must meet the same cosine and error-improvement conditions in every
target window.

## Strong null and decision rule

The strong null is true if A fails, if C is unresolved or changes across sources/waves, or if on natural text the
cross-form predictor fails both the union local-response test and the exact-causal improvement over the naive code
control.

- A/B/C/D/E all true: identify the quadratic reader plus state form as a factorization-independent, cross-register
  predictive circuit interface. The next test is an exact executable intervention derived from this interface.
- A/C true but D or E false: retain the failure localization, but do not call the forms a predictive circuit. Move to
  a context-conditioned state-level causal quotient.
- C unresolved: averages have erased the relevant conditional structure. Do not tune product terms or form ranks;
  build the state-level quotient with explicit distance, predecessor count, and downstream-use variables.

No outcome in this rung can establish compression or adoption by itself.

# CrossFirst direct third-derivative correction V1

## Question

The finite-scale curve established that the complete Hessian is locally valid
and that native-scale `licence/license` failure is higher-order.  A per-row
cubic coefficient estimated from only the quarter/half-scale arms reduced its
full-scale error by 97.7%, but that coefficient is an oracle diagnostic.
Replace it with the exact third directional derivative of the native suffix.

For suffix readout `f`, post-attention9 child tangent `a`, and remainder tangent
`b`, the finite cross interaction has Taylor expansion

`I(s) = s^2 H + s^3 C3 + O(s^4)`,

where

`H = D2f[a,b]`

and

`C3 = 0.5 * (D3f[a,a,b] + D3f[a,b,b])`.

Compute each third derivative in two symmetry-equivalent JVP orderings and use
their arithmetic mean at each prompt.  No
behavioral value, scale arm, or fitted coefficient enters `C3`.  This is an
opened-panel identity/utility diagnostic on the 48 new-endpoint rows, not a new
OOD claim.

## Predictions

- **A -- third-derivative instrument:** two derivative orderings for each of
  `D3[a,a,b]` and `D3[a,b,b]` agree within `2e-5` relative error; the bound
  Hessian and scale-one physical interaction replay exactly.
- **B -- finite cubic identification:** direct `C3` agrees with the frozen
  quarter/half-scale finite-difference cubic estimate within `0.15` relative L2
  for every family and endpoint concept.
- **C -- licence repair:** `H+C3` predicts the full-scale `licence/license`
  interaction with relative L2 at most `0.10` and cosine at least `0.98`.
- **D -- broad finite correction:** in every family and endpoint concept,
  `H+C3` leaves at most `0.10` of child-effect norm and has finite-interaction
  cosine at least `0.95`.
- **E -- improvement:** for every family, direct cubic relative L2 against the
  full-scale finite interaction is at least 30% lower than Hessian-only error.
- **F -- audit:** serialize promptwise `H`, both third-derivative terms, direct
  `C3`, finite-estimated `C3`, physical interactions, and both readers.

If A fails, repair derivative mechanics before interpretation.  B failure with
A passing means quarter/half scales still contain material fourth-and-higher
contamination.  C--E passing nominates `H+C3` for a genuinely fresh test; it
does not make the derivative evaluator a small extracted circuit.

## Price

One checkpoint load; 48 opened prefixes; one native upstream trace and four
triple-nested JVP evaluations per prefix; no physical suffix arms, fitted
coefficients, optimization, parameter updates, or backward-to-weight gradients.

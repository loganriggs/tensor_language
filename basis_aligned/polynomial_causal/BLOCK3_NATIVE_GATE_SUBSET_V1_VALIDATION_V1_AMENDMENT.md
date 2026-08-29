# Block-3 native-gate subset validation V1 — numerical replay amendment

Frozen after validation V0 failed on its first native replay guard, but before any
candidate write, candidate suffix, local candidate metric, candidate state, or
candidate logit was scored.  V0's authority and failure are preserved at commit
`843cf706`; V0 has no result and no success receipt.

All scientific arms, roles, budgets, thresholds, bootstrap rules, adaptive branches,
physical call requirements, and promotion rules remain exactly those in the V0
amendment.  V1 changes only the namespace, binds the preserved V0 failure lineage, and
repairs a scale-invalid float32 algebraic replay guard.

## Observed implementation failure

The V0 guard required native polarized replay maximum absolute error at most `3e-4`.
On the first validation batch, before any candidate arm was scored, the native write
had maximum magnitude `5493.65381` and RMS `600.992615`.  Its algebraically equivalent
float32 replay differed by maximum `0.009765625` and RMS `0.000526577`: max-relative
`1.77762e-6` and RMS-relative `8.76179e-7`.  RMS polarization itself replayed with
`max_abs(u+v-z)=9.53674e-7`.  The absolute guard was therefore not scale invariant.

## Frozen V1 replay rule

For observed tensor $y$ and algebraic replay $\widehat y$, define

$$
e_\infty =
\frac{\lVert \widehat y-y\rVert_\infty}
     {\max(\lVert y\rVert_\infty,\operatorname{tiny}_{32})},
\qquad
e_2 =
\frac{\operatorname{RMS}(\widehat y-y)}
     {\max(\operatorname{RMS}(y),\operatorname{tiny}_{32})}.
$$

Both tensors and all six reported scalars must be finite.  Native MLP replay and the
direct-K-product versus four-typed-term candidate replay each pass only if

$$
e_\infty \le 2\times10^{-5}
\quad\text{and}\quad
e_2 \le 2\times10^{-5}.
$$

The limit is more than eleven times the observed native maximum-relative discrepancy,
but remains about fifty times tighter than `1e-3`; it is intended only to distinguish
float32 evaluation order from an algebraic or bias error.  Absolute maximum, reference
maximum, relative maximum, absolute RMS, reference RMS, and relative RMS are retained
in the result for both native and candidate replay.

V1 uses create-only files whose names contain `validation_v1`; it must never modify the
preserved V0 files.  Its source closure includes this amendment and the committed V0
authority/failure bytes.  All prior independent audit requirements remain binding.

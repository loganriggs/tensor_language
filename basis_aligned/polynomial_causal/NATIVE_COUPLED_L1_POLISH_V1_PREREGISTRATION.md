# Native coupled L1 dictionary polish

Resume both completed OVERCOMPLETE_L1_READER_V1 fit caches, without changing
the normalized native-reader training split, 2304 features, L1 penalty .05,
native product pairings, or retained Down/bias. This isolates joint optimization
from the known restrictions of the reader-reconstruction proxy. No text is
read and no validation rows select the fitted features.

Reuse active_orthant_dictionary_v1: hold current code positions/signs, jointly
optimize their nonnegative magnitudes and normalized dictionary directions by
SciPy L-BFGS-B, then refresh all codes with the existing proximal solver. An
outer round allows at most 200 L-BFGS-B iterations / 60 soft seconds, followed
by at most 1000 proximal steps at 1e-7 relative stationarity. Limit each start
to 600 soft fit seconds / 20 rounds; save every completed round atomically.
Report actual time, function evaluations and CUDA memory. SciPy parameter
updates/transfers occur on CPU; native speed is not inferred from the toy test.

Unit-normalizing used atoms can preserve reconstruction and lower the penalty;
report the actual canonicalization effect rather than counting it as optimizer
progress. The original unit-ball objective supplies all convergence checks.
Require two successive full joint stationarity measurements <=1e-5 and their
relative objective change <=1e-8, matching the parent stopping convention.
Fixed-orthant solver success and conditional-code convergence do not suffice.

After fitting, use the same batched Lasso(.05)/OLS-completion encoder with 128
terms as OVERCOMPLETE_OLS_REENCODE_V1. Compare against that completed run's
learned dictionaries, not the older padded encoder. Score all 9216 native
readers, full-U coefficient reconstruction, and complete-function similarity
across starts with existing contractions. Retain training and historical test
scores; the repeatedly inspected weight split is not fresh validation.

Registered predictions:

- A: CPU/GPU objective and analytic-gradient preflight relative errors <=1e-9;
  parent objective replay <=1e-10; canonical reconstruction error <=1e-10;
  each round objective increase <=1e-10; finite diagnostics and atom norms
  <=1+1e-6. Final encoders converge with maximum KKT <=1e-5; sparse execution,
  support normal equations and parent folded-capture replay errors <=1e-8.
- B: both starts meet the full joint convergence rule above.
- C: both reduce full joint relative stationarity by at least 10 times.
- D: both gain at least .01 absolute full-U coefficient capture over their
  respective repaired-encoder parent functions.
- E: both gain at least .01 absolute historical test reader capture over those
  same parents.
- F: the two final complete functions have full-U coefficient cosine >=.9.

Opposing interpretation: B/C held with D/E/F missed would settle this optimizer
limitation locally while directing discovery toward a different objective or
structure. B missed leaves optimization unresolved; it cannot establish absent
structure. D held without F is improved approximation, not stable circuit units.
Preserve every original parent miss and report all results, including timeouts.

Managed lane1, after existing frozen re-encoding completes and is interpreted.
Whole-job alarm 2100 seconds. Zero body forwards or corpus accesses. Program
price unchanged: 9,142,272 matrix coefficients, 1,179,648 stored indices and
1152 bias values; runtime COO indices, unembedding and background also count.
This is an optimizer repair test, not a matched-capacity comparison against
smaller families or evidence of OOD prediction/removal/composition.

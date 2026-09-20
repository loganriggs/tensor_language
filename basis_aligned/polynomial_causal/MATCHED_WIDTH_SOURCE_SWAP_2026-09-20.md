# Source substitution at matched width

Replacing the early-write source (blocks0–3) with the residual complement
improves native selective success from42/48 to44/48 cells while retaining
five source features. The six-source control passes45/48. This identifies
a more useful source grouping on the opened panels, not a complete circuit.

The candidate keeps embedding recurrence, middle writes4–7, MLP8, MLP10
and the residual complement (attention8, attention9, MLP9, attention10 plus
rounding gauge). Other execution semantics are unchanged. All four remaining
failures are strength; maximum modal collateral is8.206% and quadratic
number-prediction error8.250%. Instrumentation and prediction gates pass;
the all-cell selective gate fails.

All six possible common source omissions were screened using analytic
derivatives. Omitting early writes predicted44/48 versus43,42,42,41,42 for
the other omissions in source order. Selection was on opened data; no fresh
transfer claim follows from this comparison. Five-source symmetric quadratic
storage is80values/context versus108 for six. Both require native context
and source generation. Zero-padding into a six-slot validation harness does
not increase the candidate's mathematical input width, but the harness itself
has not been optimized to reduce runtime.

Positive-result redteam found a hidden selection dependency: the five-source
screen used a projected six-source optimized vector as one initial point.
An independent selector now receives only the five-source gradient/Hessian,
its unit-B reference and its own exact-null LP initialization. It still
predicts44/48 cells. One of288 amplitude vectors changes by more than1e-5
(maximum coordinate change2), requiring new native validation. That run is
separately registered as `native_independent_source_swap_v1`.

Native matched-width receipt:
`../bilinear_quotient/circuits/followups/native_matched_width_source_swap_v1_result.json`.
Independent-selector frozen artifact: `INDEPENDENT_SOURCE_SWAP_V1.json`.
The fresh-panel rule should be frozen only after that numerical independence
check; support choice stays fixed across every input.

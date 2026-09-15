# Bracket suffix live-amplitude selection V1

The donor-free rank-two L13H8 source generator transferred on fifth and sixth constructions, while a fixed ordered-pair suffix scalar failed on the sixth because amplitude changed. Before creating or opening a seventh construction, select a live-amplitude suffix law using only those two opened panels.

Recompute the frozen donor-free source replacement on all fifth/sixth endpoints. For each target endpoint define its source delta relative to the native recipient opener term, `delta_t = t_program-t_recipient`, and two outcome-independent scalar features:

1. `direct`: dot `delta_t` with the frozen LM-head donor-minus-recipient closer row;
2. `norm`: Euclidean norm of `delta_t`.

The target is the already recorded donor-free program effect, keyed exactly by row ID and side. No new causal outcome is generated. Compare five through-origin linear laws:

- one global direct coefficient;
- one global norm coefficient;
- two global coefficients on direct and norm;
- six ordered-pair direct coefficients;
- six ordered-pair norm coefficients.

For each law, fit on the fifth panel and predict the sixth, then fit on the sixth and predict the fifth. A direction passes with cosine at least `0.90`, relative L2 error at most `0.40`, sign agreement at least `0.90`, and norm ratio in `[0.60,1.40]`. A law passes only if both directions pass. Select among passing laws by fewest coefficients, then lowest worst-direction relative error, then the fixed name order above. Refit only the selected form on the pooled fifth+sixth rows and serialize its coefficients for one future confirmation. Construction identity is never a feature.

Price is two forwards over 288 endpoints: the extended factor capture and an independent exact factor replay used solely to verify the capture. This is 576 sequence evaluations, eleven analytic least-squares fits (two directions for five candidates plus the pooled selected law), zero backwards, parameter updates, or quantization.

Predicates:

- **A — exact instrument:** extended versus independent replay maximum final-query logit error at most `1e-5`; exactly 144 target effects bind to the two immutable result files; exact 2/576 accounting.
- **B — candidate exists:** at least one registered law passes both transfer directions.
- **C — selected law:** the deterministic selected law passes every frozen transfer bar in both directions.
- **D — finite nondegenerate program:** all selected coefficients and feature scales are finite, and each selected design column has nonzero norm on both panels.
- **E — selection boundary:** the serialized program names only `direct`, `norm`, and optionally ordered pair; it contains no construction label, row identifier, seventh-row input, intercept, rank change, or quantized value.

All predicates passing yields `live_amplitude_law_selected`. A valid failure yields `live_amplitude_candidates_null` and closes this suffix-free route without adding features or parameters.

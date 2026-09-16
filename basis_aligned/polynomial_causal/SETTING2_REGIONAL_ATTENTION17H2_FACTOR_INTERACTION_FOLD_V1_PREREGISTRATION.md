# Head17.2 QK1 × QK2 × value interaction fold

Frozen after the nine-head response assay localized the attention17 response to
head17.2 and before this factor decomposition is scored. Reuse the already-open
48-row `ODD_FRAMING_FRESH_V1` discovery authority. This is a within-candidate
fold and causal discovery assay, not fresh generalization evidence.

For the fixed upstream removal of head9.8's late-touching QK1 route, capture
head17.2's native and edited factors at the final query: complete native QK1
score `S1`, complete native QK2 score `S2`, and the learned mixture of current
and inherited values `V`. Evaluate all eight native/edited factor corners of

$$
O_{17,2}\sum_s M_{ts}S1_{ts}S2_{ts}V_s.
$$

Use inclusion-exclusion to obtain exactly seven nonempty response terms:
`QK1`, `QK2`, `V`, `QK1×QK2`, `QK1×V`, `QK2×V`, and `QK1×QK2×V`. Their sum
must reproduce the edited-minus-native head17.2 write. Enumerate all supports of
at most three terms with coefficients fixed to one; no gains or refits are
allowed. Choose minimum aggregate paired-response relative L2, breaking ties by
fewer terms and then lexicographic term indices.

## Frozen checks and predictions

- `pred_a_exact_instrument`: exactly 12 full-model executions; upstream carry,
  manual head9.8 attention, native/edited attention17 and head17.2 projection,
  eight-corner endpoints, and seven-term inclusion-exclusion closure each pass
  at relative error at most `2e-6`; all outputs are finite.
- `pred_b_sparse_response_replay`: the selected at-most-three-term support
  replays the paired head17.2 response with aggregate relative L2 at most `.25`,
  cosine at least `.90`, and each-family relative L2 at most `.35`.
- `pred_c_causal_installation`: adding the selected interaction writes to the
  native layer-17 background predicts the full head17.2-response installation
  target effect in each family with cosine at least `.90`, relative L2 at most
  `.35`, and sign agreement at least `.80`.
- `pred_d_causal_removal`: subtracting the same selected writes from the
  upstream-edited layer-17 background predicts removal of the full head17.2
  response under the same per-family bars.
- `pred_e_control_nonworsening`: for installation and removal in each family,
  the selected program's maximum RMS over work/jobs, cat/dog, red/blue,
  Monday/Tuesday, and apple/orange is no larger than `1.25×` the corresponding
  full-head-response control RMS plus `1e-6`.
- `pred_f_equal_norm_random_null`: eight deterministic random writes drawn in
  the native head17.2 output subspace and normalized per row to the selected
  write norm are installed and removed at the same site. The selected program's
  aggregate target relative L2 must beat the best random installation and the
  best random removal by at least `.10`.

The price is 12 native/edited full-model prefixes (96 sequences), eight exact
factor-corner evaluations, exhaustive enumeration of 64 supports including the
empty support, four 48-sequence full/selected layer17-MLP/final-readout suffix
evaluations, and sixteen 48-sequence equal-norm random-null suffix evaluations.
Native and edited baselines are cached from the full executions. There are zero
fits, backwards, gradients, parameter updates, or quantization. All RMS, rotary,
causal-mask, output-projection, MLP17, final RMS, and softcap semantics remain
native.

A pass freezes a small factor-interaction support for a genuinely fresh causal
test. It does not establish OOD transfer, source generation, or independent
extraction because edited factors are still read from the upstream
counterfactual run. A valid null means head17.2 response localization does not
compress to three unit-gain QK/value interaction terms under these causal bars.

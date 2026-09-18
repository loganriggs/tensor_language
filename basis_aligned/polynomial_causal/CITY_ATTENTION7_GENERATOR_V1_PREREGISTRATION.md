# Generate the attention7 city source from an earlier residual prefix

CPU-only opened implementation check. Input is residual6 through the city token
and matching token IDs. Look up initial states and inherited first values from
frozen weight-derived token tables. Form mixed7 using learned lambda7; RMS it;
compute all nine heads' QK1*QK2*V contribution to the city query with both current
and inherited values, head RMS, native-rounded rotary and output projection.
No softmax, fit, term omission or magnitude selection.

Generated mixed7(city)+attention7(city) supplies normalized MLP7 input. Feed it
to the frozen three-reader program, and derive other_city_sources via lambda8
and initial(city). These replace two supplied city vectors with one earlier
residual6 prefix. Native head8 query fields remain external; native mixed8 RMS is
still supplied in the exact comparison. Generated-norm approximation is not
promoted by this check. Report weight and state growth, not merely array count.

Predictions on all40opened CPU-prefix fixtures:
- a: generated attention7 city output relative error<=1e-4 against direct CPU
  native capture, per sequence; initial/inherited tables unchanged.
- b: generated normalized MLP7 input<=1e-4 against direct CPU capture; generated
  three reader values<=1e-4 against native projected MLP7 output, per sequence.
- c: exact city-normalizer mode using generated inputs matches saved GPU full
  city-removal write<=1e-4relative per sequence. All tensors finite, no GPU use.

All source tokens beyond city are excluded by causality; report prefix length
and literal state scalars. Known396-token vocabulary only; unsupported IDs must
fail. This is an earlier-boundary implementation, not token-only extraction,
full-suffix verification, new OOD evidence, source selectivity or composition.

# Final width-five preservation-aware head17.2 port search

The exhaustive width≤4 search found `attn9 + mlp9 + attn10 + attn11`. It passes
both causal directions and all preservation controls, but its aggregate response
error `.2511880484` narrowly exceeds the frozen `.25` bar. Its minimax normalized
gate score is `1.0047521938`; this is a valid boundary null, not numerical error.

Keep every V2 state, factor construction, twelve token readouts, gate definition,
minimax ordering, threshold, seed, and unit gain. Extend the candidate family
only to all 6,885 supports of width zero through five. No coefficients are fit.
The V2 result and exact V2 runner are bound.

Predictions `pred_a` through `pred_f` retain their original meanings: exact
instrument, response replay, bidirectional causal fidelity, preservation,
improvement over the target-only support, and deterministic support specificity.
Replace the extension gate with:

- `pred_g_improves_width4`: the selected minimax score is strictly below V2's
  bound `1.0047521938` score.

Price: 48 opened rows; 12 native/edited full executions over 96 sequences; 6,885
factor supports; 660,960 candidate suffix sequences plus 96 complete-corner
suffix sequences; twelve required token logits per sequence; zero fitted gains,
backwards, gradients, parameter updates, or quantization. A pass is still only an
opened-panel candidate and must be frozen on fresh rows. A valid null ends
unit-gain whole-module support growth at five edges; larger supports are outside
the intended sparse graph and must not be tried as a rescue.

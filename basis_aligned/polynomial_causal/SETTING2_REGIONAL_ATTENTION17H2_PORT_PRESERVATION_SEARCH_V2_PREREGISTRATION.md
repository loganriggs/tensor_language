# Width-four extension of preservation-aware head17.2 port search

V1 exhaustively evaluated every unit-gain support of at most three propagated
layer9–16 module writes. Its minimax winner `attn9 + attn10 + attn11` is a valid
null with score `1.228257`: causal target gates pass, while aggregate response
error `.285297` exceeds `.25` and the worst preservation ratio `1.228257`
exceeds one. The earlier target-only support scores `3.220058`.

Keep V1's states, exact factor construction, twelve token readouts, normalized
gate definitions, minimax selection rule, thresholds, seeds, and unit gains.
Extend the candidate family only: enumerate all 2,517 supports of width zero
through four. No coefficient is fit. The V1 receipt and exact runner are bound.

Predictions `pred_a` through `pred_f` are unchanged from V1: exact instrument,
response replay, bidirectional causal fidelity, preservation, improvement over
the target-only support, and deterministic support specificity. Add:

- `pred_g_improves_width3`: the selected minimax normalized-gate score is at
  least 10% below V1's `1.228257` score.

Price: 48 opened rows; 12 native/edited full executions over 96 sequences; 2,517
factor supports; 241,632 candidate suffix sequences plus 96 complete-corner
suffix sequences; twelve required token logits per sequence; zero fitted gains,
backwards, gradients, parameter updates, or quantization. A pass remains an
opened-panel candidate requiring frozen fresh confirmation. A valid null closes
the complete unit-gain whole-module support class through four edges; it does not
rule out within-module directions, non-unit coefficients, or larger graphs.

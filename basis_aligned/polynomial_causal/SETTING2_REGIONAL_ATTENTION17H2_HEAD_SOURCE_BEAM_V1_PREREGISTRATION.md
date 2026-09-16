# Head-level source decomposition for the head17.2 factor corner

The frozen five-module source program transferred its target effect but failed
fresh unrelated-reader preservation.  This test asks whether the failure is a
coarse attention-module boundary artifact.  It reopens the 48-row
`ODD_FRAMING_FRESH_V1_ROWS.json` panel for discovery only.

Split the propagated `attn10` and `attn11` write differences exactly into their
nine projected native head writes.  The layer-9 intervention is already wholly
inside head 9.8, so expose its propagated attention difference as `attn9h8`.
Keep `mlp9` and `mlp15` as temporary whole-module ports.  Include whole
`attn15` and `mlp16` only so the earlier target-only baseline remains exactly
representable.  No gains, rotations, latent directions, gradients, or model
parameters may be fit.

Search supports of width one through eight with a deterministic beam.  At each
width, extend the preceding width's best 64 supports by one term, deduplicate,
evaluate every extension using the already frozen minimax response,
installation, removal, and five-reader preservation score, and retain the best
64 under `(score, width, response error, lexical support tuple)`.  The final
candidate is the best evaluated support across all widths.  Also evaluate:

- the exact 21-term expansion of the discovery five-module program;
- the exact `attn9h8 + attn15 + mlp16` target-only baseline; and
- sixteen seed-`202609160533` uniformly sampled same-width supports as nulls.

Predictions, fixed before execution:

- `pred_a_exact_instrument`: existing carry, attention, endpoint, and source
  closure audits pass; the explicit head partitions close their native module
  writes exactly after a recorded final-head floating-point closure correction.
- `pred_b_response_replay`: aggregate response error <= `.25`, cosine >= `.90`,
  and each family error <= `.35`.
- `pred_c_bidirectional_causality`: frozen family installation/removal gates
  pass in both directions.
- `pred_d_preservation`: every frozen unrelated-reader gate passes.
- `pred_e_improves_coarse_program`: width <= 8 and minimax score is strictly
  below the exact 21-term coarse-program score.
- `pred_f_support_specificity`: selected score plus `.10` is no larger than the
  median same-width null score.

A pass is an opened-panel candidate only and must be frozen on an independent
regional panel.  A valid null rejects native head identity as the missing
within-module coordinate and redirects the search to consumer/Hessian-derived
directions; it does not resurrect wider whole-module supports.

Maximum price: 48 opened rows; 12 native/edited full-model executions over 96
sequences; at most 7,400 support evaluations and 710,400 candidate suffix
sequences, plus 96 complete-corner suffix sequences; twelve token logits per
suffix sequence; zero backwards, gradients, fitted coefficients, parameter
updates, or quantization.

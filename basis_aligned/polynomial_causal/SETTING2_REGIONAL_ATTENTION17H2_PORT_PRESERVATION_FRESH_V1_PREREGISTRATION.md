# Fresh confirmation of the five-edge head17.2 source program

Freeze the discovery support exactly:

`attn9 + mlp9 + attn10 + attn11 + mlp15`.

Each edge is the unit-gain propagated native-vs-fixed-edit module-write change at
the layer-17 residual input. Recompute head17.2 QK2 and value from their sum while
keeping QK1 native. No support, coefficient, row, reader, threshold, or null may
change after execution.

Use the 48 rows in
`SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_ROWS`. They have zero
context overlap with the discovery rows and were used previously only to confirm
the downstream factor corner, not to select upstream module edges. This is fresh
for the source-support hypothesis, though not a claim about pretraining-corpus
OOD.

Frozen predictions:

1. `pred_a_exact_instrument`: exactly 12 full executions; carry, manual
   attention9, corrected source identity, zero/full endpoints, and independent
   head17.2 projection errors are at most `2e-6`; raw BF16 recurrence remainder
   is below `1e-5`; all readouts are finite.
2. `pred_b_fresh_response_replay`: aggregate response error is at most `.25`,
   cosine at least `.90`, and both family errors are at most `.35`.
3. `pred_c_fresh_bidirectional_causality`: installation and removal each have
   relative L2 at most `.35`, cosine at least `.90`, and sign agreement at least
   `.80` in both families.
4. `pred_d_fresh_preservation`: every unrelated-reader RMS is at most `1.25×`
   the complete-corner RMS plus `1e-6`, in both directions and families.
5. `pred_e_beats_target_only`: maximum preservation ratio is at least 10% below
   the frozen target-only support while all target gates pass.
6. `pred_f_frozen_support_specificity`: frozen minimax score plus `.10` is no
   greater than the median of sixteen preselected deterministic equal-width
   support nulls.

Price: 48 fresh-for-support rows; 12 native/edited executions over 96 sequences;
18 evaluated supports (one frozen, one target-only, sixteen nulls); 1,728
candidate suffix sequences plus 96 complete-corner suffix sequences; twelve
required token logits per sequence; zero selection, fitted gains, backwards,
gradients, parameter updates, or quantization. A pass is fresh conditional
composition, causal removal, and preservation for a five-edge graph. It remains
conditional on the five observed module-write differences and therefore is not
donor-free extraction.

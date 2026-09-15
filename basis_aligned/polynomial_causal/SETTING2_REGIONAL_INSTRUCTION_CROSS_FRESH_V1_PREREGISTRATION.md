# Fresh recursive test of the instruction-source head9.8 cross

Frozen 15 September 2026 before model execution on the 48 score-blind rows in
`SETTING2_REGIONAL_INSTRUCTION_CROSS_FRESH_V1_ROWS.json`. They use two new
templates, Glasgow/Seattle and London/Dallas, and have zero context overlap with
prior row manifests. The opened source census selected the entire instruction
region because it replayed the exact cross vector at `.191/.150` error.

Run native, additive-without-cross, instruction-cross, and full-cross arms. The
last three share $-Rv+P\Delta v$; instruction-cross adds

$$
-\sum_{k\in\mathrm{instruction}}R_{qk}\Delta v_k,
$$

while full-cross adds the sum over every framing source.

- `pred_a_exact_instrument`: carry, attention, midpoint, and joint-head audits
  close within $2\times10^{-6}$ over exactly 24 physical batches.
- `pred_b_native_capability`: at least 10/12 native UK/US pairs have the expected
  sign in each family.
- `pred_c_instruction_subset_replays_cross`: instruction-only versus full cross
  has head-vector relative error at most `.30` and recursive paired-logit relative
  error at most `.35` in each family.
- `pred_d_cross_live_and_directional`: full-cross paired-logit RMS is at least
  `.01` and instruction/full cross cosine is at least `.90` in each family.
- `pred_e_unrelated_reader_selectivity`: every full-joint unrelated-reader RMS
  ratio is at most `.75` in each family.

No fitting, row selection, gradients, parameter updates, or quantization. This is
fresh authored-task confirmation of one source subset, not corpus OOD evidence.

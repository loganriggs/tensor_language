# Fold the induced attention17 response into native heads

Prepared at the end of the 06:01 circuit phase and reserved for the next
`WEIGHT_FOLDING` phase. On the opened 48-row QK1-edit panel, reconstruct all nine
attention17 head writes from native QK1, QK2, value, and output weights. Subtract
edited from native head writes and contract each propagated final-position write
with the row-specific UK/US unembedding reader.

- `pred_a_exact_head_partition`: native/edited head writes sum to attention17
  within `1e-6`; head response terms sum to the prior attention17 response within
  `2e-6`; the prior `.24952` module ratio replays within `1e-5`; 12 batches run.
- `pred_b_top_head_concentrated`: the largest head has at least `.50` of the
  attention17 response norm.
- `pred_c_top_head_family_stable`: that head reaches `.35` of attention17 norm
  and positive aligned fraction in both template families.
- `pred_d_top3_replay_attention17`: the three largest heads replay the complete
  attention17 response within `.25` relative error.
- `pred_e_top_head_directional`: the largest head has cosine at least `.70` with
  the complete attention17 response.

Ranking is descriptive on opened rows. No head intervention, fit, gradient,
parameter update, new transfer, or quantization. A passing head becomes the
frozen candidate for QK/source folding on a later panel.

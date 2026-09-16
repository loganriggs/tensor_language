# Preservation-aware exhaustive search over head17.2 source ports

The prior target-only exhaustive search selected `attn9 + attn15 + mlp16` and
passed target response, installation, and removal, but failed unrelated-reader
preservation. This opened-panel experiment distinguishes a bad selection metric
from a bad edge granularity.

Reuse the same exact 16 propagated layer9–16 write differences and enumerate all
697 unit-gain supports of size zero through three. For every support, recompute
head17.2 QK2 and value from the hybrid layer-17 state while keeping QK1 native.
Execute both installation into the native layer-17 background and removal from
the edited background. Score paired target response and the five fixed unrelated
readers in both prompt families.

For each candidate form normalized violations of the frozen gates: aggregate
response error `/.25`; family response error `/.35`; family installation and
removal error `/.35`; cosine deficit `(1-cosine)/.10`; sign deficit
`max(0,.80-sign)/.20`; and every control RMS divided by
`1.25*full_RMS+1e-6`. The minimax score is the largest normalized violation.
Selection minimizes `(minimax score, support width, response error,
lexicographic support)`. No gains or coefficients are fit.

Frozen predictions:

1. `pred_a_exact_instrument`: exactly 12 full executions; carry, manual
   attention9, corrected source identity, zero/full factor endpoints, and
   independent head17.2 projection errors are at most `2e-6`; all candidate
   readouts are finite. The raw non-selectable BF16 recurrence remainder is
   reported separately and must remain below `1e-5`.
2. `pred_b_response_replay`: selected aggregate response error is at most `.25`,
   cosine at least `.90`, and each family error is at most `.35`.
3. `pred_c_bidirectional_causality`: installation and removal each have relative
   L2 at most `.35`, cosine at least `.90`, and sign agreement at least `.80` in
   both families.
4. `pred_d_preservation`: every selected unrelated-reader RMS is no more than
   `1.25×` its complete-corner RMS plus `1e-6`, for both directions and families.
5. `pred_e_improves_target_only`: the selected maximum control ratio is at least
   10% below the bound prior target-only support's ratio, while all target
   response/causal gates still pass.
6. `pred_f_support_specificity`: selected minimax score plus `.10` is no greater
   than the median of sixteen deterministic equal-cardinality support nulls.

Price: 48 opened rows; 12 native/edited full executions over 96 sequences; 697
factor-support constructions; 697 installation and 697 removal suffix batches,
66,912 candidate suffix sequences plus 96 complete-corner suffix sequences;
only twelve required token logits per sequence; zero fitted gains, backwards,
gradients, parameter updates, or quantization. A pass is an opened-panel
preservation-aware source candidate, not OOD evidence or donor-free extraction.

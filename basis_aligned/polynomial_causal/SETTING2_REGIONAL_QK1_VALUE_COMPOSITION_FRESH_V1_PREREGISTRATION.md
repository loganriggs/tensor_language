# Fresh head9.8 QK1 routing × current-value composition

Frozen before model execution on 15 September 2026. Use the 48 score-blind rows
in `SETTING2_REGIONAL_QK1_VALUE_COMPOSITION_FRESH_V1_ROWS.json`; their token
contexts have zero overlap with every existing `*ROWS.json` manifest. No row,
activation, or score from this panel selected either component.

At head9.8, let `P = QK1 * QK2` be native causal routing, let `R` be the
previously frozen QK1 contribution containing at least one of
`{attn5, mlp5, mlp6, mlp7}`, let `v` be the native mixed value, and let `dv` be
the current-value change caused by the established head8.2 inherited-city write
at half donor strength. Restrict `dv` to framing source positions, matching the
identified value circuit. Execute native, routing-only, value-only, and joint
arms through the native recursive suffix. Their head-space changes are

$$
\Delta z_R=-Rv,\qquad
\Delta z_V=P\,dv,\qquad
\Delta z_{RV}=-Rv+P\,dv-R\,dv.
$$

The last term is the explicit bilinear overlap. Both single interventions and
the joint use the same frozen tensors; there is no fit, search, or parameter
update.

- `pred_a_exact_instrument`: carry-source reconstruction, manual attention9
  reconstruction, half-write scaling, and joint head-output algebra each have
  relative error at most `2e-6`; all 24 expected batched model executions finish
  with finite values.
- `pred_b_both_single_branches_live`: in each template family, routing-only and
  value-only paired target-effect RMS are each at least `0.002` logits and at
  least 24 of 24 native cue pairs are positive overall.
- `pred_c_multiplicative_overlap_is_material`: in each family the explicit
  `-R*dv` final-position head-vector RMS is at least 2% of the smaller single
  head-vector RMS, and the recursively measured target difference-of-differences
  is at least 1% of the smaller single target-effect norm.
- `pred_d_joint_family_transfer`: in both families, the joint target effect is
  finite and has cosine at least `0.5` with the sum of its two single target
  effects. This is a stability gate, not an additivity claim.
- `pred_e_unrelated_reader_selectivity`: in each family, every unrelated
  reader's joint-effect RMS is at most `0.75` of target joint-effect RMS.

Report signed UK-minus-US effects for the pre-softcap unembedding numerator and
the final softcapped logits, by family and arm. Failure of either single liveness
gate makes overlap interpretation inconclusive. This tests one controlled
regional circuit and does not claim whole-head necessity, corpus OOD transfer,
or a complete model circuit.

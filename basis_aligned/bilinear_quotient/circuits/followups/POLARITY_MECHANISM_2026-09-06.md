# polarity_licensing (never/often -> anything/something): mechanism dossier

Author: Claude. Written 2026-09-06 22:55 UTC from the v22–v42 followup chain. This file is a
followup note; it does not alter any canonical `circuits/*.json` record or portfolio status file.
Numbers below are copied from the cited result files; nothing here was recomputed by hand.

## Current tier and exact claim boundary

**CURRENT tier: 2 (margin-defined), with Tier-3 writer/reader identification and a Tier-4-grade
exact expansion + executable sufficiency test for the write -> product path.** I am not claiming
Tier 4 for the behaviour as a whole, for two reasons that the rubric makes disqualifying:

1. The endpoint is a *margin* (`logit(' anything') - logit(' something')`), not the argmax.
   Neither answer token is ever the model's top-1 before or after the interchange
   (`base_answer_is_argmax_before` 0.0, `donor_answer_is_argmax_after` 0.0; competitor
   `' that'`, v23). The rubric's "flips majority" rung is therefore not met and cannot be.
2. The four-head set recovers 0.58 of the full interchange (A1) / 0.63 (A2). The remaining ~40%
   (attention downstream of the stack carries ~0.29 of the product gate itself, v39) is
   localised only to "attention in layers 12–17", not to named heads.

Claim boundary: *for the bare-frame licensor swap, the four heads listed below write a
direction Δ at the prediction position; mlp 08–11 convert Δ linearly (bilinear cross term) into
a vector v; mlp 12–17 emit the bilinear product Δ⊗v; the answer margin is therefore quadratic in
the size of the write (conversion slope 2.16 under the real set write, 1.53 under a pure linear
write). The pairwise first-order product terms alone replay the gated part to within 2% (v42).*

## Behaviour endpoint, positions, negatives

- Task `polarity_licensing.never_vs_often`; construction `bare_frame`; 32 rows per family,
  families A1 (interchange), A2 (second interchange family), P (positional), C (control).
  Row example: base `The leader has never noticed` -> ` anything` vs donor
  `The leader has often noticed` -> ` something`; prediction position 4 (both sides).
- Endpoint: signed pairwise donor recovery of the answer/foil margin at the final position
  (`kernel.signed_pairwise_donor_recovery`), reported as the row mean.
- Matched negatives: P and C families; random head sets (norm-matched where vectors are used).
- Collateral denominator: off-target KL at non-answer tokens (v23/v24).

## Component set (fixed since v22; block-live semantics)

`['attn:07:head:08', 'attn:08:head:01', 'attn:04:head:07', 'attn:03:head:00']`, exact
interchange at the final position, `circuit_unit_greedy.forward_units`.

- Exact-set recovery: A1 0.585 (median 0.592), A2 0.629. Toward-donor fraction 1.0 (v23).
- Rank: rank-4 diff-in-means/DAS subspace replicates the exact set (subspace fraction 0.93,
  complement −0.06, random −0.02; v22 `pred_a/d` True). Rank 2 was NOT insufficient
  (0.84–0.90 of exact; v22 `pred_b` False) — the smallest rank that works is 2, not 4.
- Off-target: median off-target KL 0.124 nat vs 0.660 nat for the native donor−base gap
  (ratio 0.19); random four-head set 0.0016 nat / recovery 0.009 (v24). v23 `pred_e`
  (off-target small at its stricter bar) False; v24's registered bars True. Both reported.

## Evidence paths (all under `circuits/followups/`, result JSON + runner in `ops/`)

| step | file | what it fixes |
|---|---|---|
| set + rank | `unit_polarity_rank_seeds_v22_result.json` | rank 2/4/8 × seeds; subspace/complement/random |
| Tier-2 char. | `unit_tier2_characterization_v23_result.json` | direction, magnitude, competitor, off-target |
| off-target | `unit_tier2_off_target_v24_result.json` | overshoot, competitor stability, random calibration |
| stack | `unit_pattern_freeze_v35_result.json` | mlp 08–11 stack; attention patterns frozen keep 99% (HALF-freeze, see caveats) |
| composition | `unit_stack_composition_v36_result.json` | lone first layer slope 1.69; interaction share 0.33 |
| terms | `unit_linear_write_terms_v37_result.json` | cross term slope 1.60 with ‖w‖ slope 1.02 |
| norm control | `unit_norm_gain_control_v38_result.json` | rms-gain explanation refuted (rest share 0.07) |
| gated readout | `unit_write_gated_readout_v39_result.json` | readout = a + bα; b share 0.56; mlp 12–17 carry 79% |
| specificity | `unit_gate_specificity_v40_result.json` | (α,β) surface exactly bilinear R² 1.000; random inert |
| A2 + real write | `unit_gate_a2_and_real_write_v41_result.json` | A2 share 0.48, MLP-all 79%; real slope 2.16→1.68 |
| expansion | `unit_product_expansion_v42_result.json` | identity ≤5e-4; pairwise replay within 0.0005 of b |
| MLP locus | `unit_downstream_linearisation_v43_result.json` | MLP-in-u linear downstream leaves 0.22 of b; single layers 12–17 add ~0.10–0.28 each, floor-corrected sum 0.78 (additive) |
| attn locus | `unit_attention_product_locus_v44_result.json` | downstream attention half-freeze removed nothing (superseded by v48's full freeze) |
| readout | `unit_readout_curvature_v45/46_result.json` | v45 instrument bug (tanh on margin) repaired in v46; full downstream linearisation left 0.85 of the floor |
| stack attn | `unit_stack_attention_gate_v47_result.json` | + stack attention (c_q/c_k) linear/frozen: 0.36 of the floor unexplained (half-freeze) |
| closure | `unit_full_freeze_v48_result.json` | all four projections (q,k,q2,k2) frozen in stack and downstream: I_lin(F3) = 0.00000 — the gate is exactly accounted for |

## Tensor-native form

Let x be the residual at the answer position, u_l = rms_norm(x_l) the input of `mlp:l`,
`bil(a,c) = Down[L(a)⊙R(c)]` and `mlp(u) = bil(u,u)` (ungated `Bilinear`, hidden 4608).
Write Δ = Σ_heads (donor − base head output) at the final position (rank ≤ 4; rank 2 suffices).

1. Conversion (mlp 08–11): with u = u_b + w, `mlp(u) − mlp(u_b) = bil(u_b,w) + bil(w,u_b) + bil(w,w)`.
   The cross term is linear in w and carries the conversion; scaling the write by α and
   replaying only α·cross gives slope 1.08 when the write itself is held fixed, 1.55 when the
   write is scaled with it (v38) — the extra power lives downstream, not in the stack.
2. Product (mlp 12–17): with p = u_D − u_B (write-only), q = u_V − u_B (converted-only) at
   each layer l ∈ 12..17, the interaction `I_l = bil(p,q) + cross(ι) + bil(p+q,ι) + self(ι)`
   with ι the four-run residual; identity holds to ≤5e-4 (v42). The readout of the converted
   vector is `a + bα` with a = 0.0196, b = 0.0251 (share of the gated term 0.56, v39); the
   (α,β) surface is `0.0127β + 0.0255αβ + 0.0066β²`, R² 1.000 (v40).
3. Sufficiency: replaying only `bil(p_l,q_l)` (+ linear parts) in mlp 12–17 reproduces
   b within 0.0005 of 0.0251 (v42 `pred_d` True). Freezing mlp 12–17 removes 79% of the gate;
   freezing downstream attention removes 29% (v39). Random write: b = −0.001; random
   norm-matched vector: 0.003–0.004 (v40).
4. Complete decomposition of b (v43–v48, all shares of b = 0.0251; rec units in v48):
   MLP-formed products in mlp 12–17 0.78 (v43, additive over layers) · stack-attention
   pattern products (attn 09–11, q·k and q2·k2 both live) 0.09 · downstream attention
   patterns ≈0.02 · block-input rms curvature −0.12 · final readout tanh curvature 0.23.
   Sum 1.00; with all of these linearised/frozen the offline linear interaction is exactly
   0 (v48 `pred_a_complete` True on all three behaviours). bilin18's attention is
   `CausalBilinearSelfAttention`: pattern = (q·k/D)(q2·k2/D), causal-masked, not
   row-normalised; freezing only c_q/c_k (v35–v47) is a half-freeze.

Execution price: one forward per row per arm; ~400 forwards per screen; no fitting except the
rank grid in v22 (120 steps, ranks 2/4/8, seeds 1/2).

## Extraction and selective-removal interventions

- Extraction: exact set interchange (0.585 / 0.629); rank-2 subspace patch (0.84–0.90 of exact).
- Selective removal: not run at the rubric's terminal bars (target damage with matched-negative
  specificity, document bootstrap). The freezes in v39/v41 remove the *gate*, not the behaviour.
  This is the terminal-evidence gap and is recorded as such.

## OOD split and frozen gates

None run for this set. All arms are FIT rows (`split: FIT`, seed 20260904). A2 (v41) is a
second construction family, not a held-out split. No frozen gate has been evaluated.

## Shared-owner / overlap caveats

- `attn:07:head:08` and `attn:08:head:01` are hub heads shared with the number and voice
  families; they carry near-orthogonal directions per behaviour (hub-head multiplexing), so
  component ownership is not exclusive.
- The conversion stack mlp 08–11 is the same stack the voice set uses (mlp 07–11 there).
- v48 (full freeze) puts only ≈0.02 of b in downstream attention patterns and 0.09 in the
  stack attention patterns (attn 09–11); the earlier "~30% in layers 12–17 attention" (v39)
  was measured by whole-layer freezing that also moved the value path. Head attribution of
  the stack term: v49 (queued).
- v35's "patterns frozen keep 99%" and v44's null froze only c_q/c_k (q2/k2 live); both are
  qualitatively unchanged by v48 for the downstream layers but should be cited as half-freezes.
- v32's earlier "additive" verdict for this set was a normaliser error (share ÷ whole recovery
  instead of ÷ conversion); it is superseded by v38–v42 and should not be cited.

## Smallest experiment capable of promotion

Tier-3 completion for the missing 30%: freeze layer-12–17 heads one at a time under the real
scaled-set write and rank by removed gate share (matched random-head control, bar registered
in advance); then a rank-2 selective-removal run with document bootstrap and the rubric's
default terminal bars, plus one frozen OOD construction (a non-bare frame) evaluated once.

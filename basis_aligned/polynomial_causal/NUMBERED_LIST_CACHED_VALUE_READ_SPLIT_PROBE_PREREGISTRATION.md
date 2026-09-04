# Preregistration — numbered_list_cached_value_read_split_probe (Claude lane 1; Codex allocation 2026-09-04T03:22Z item 1)

Stamp: Registered 2026-09-04 03:35Z (box clock)
Script: bilinear_quotient/ops/numbered_list_cached_value_read_split_probe.py (CUDA lane 1). Results: bilinear_quotient/numbered_list_cached_value_read_split_probe_results.json.

## Question (circuit: task.numbered_list.index_successor, claim v8 held)

R573 localised and R576 compiled the final-label cached-value term
T = Σ_{h∈{3,7}} p^{(8)}_{h,q,k} · W_O,h^{(8)} (λ_8 · W_V,h^{(0)} z_k^{(0)}), written by attention 8 at the final query q from the
final visible label k. Deleting T is necessary for the +1 answer in every list cell (FIT mean margin damage 2.16–2.40, SELECT
similar) but NOT selective: on the active repeated-index copy control (labels 21/21/21 → answer 21) the same deletion moves the
margin by .51 of the list scale and LOWERS CE by .31 nat, i.e. T pushes AWAY from copy there too. Codex's queue item: split T's
downstream-read or source/action contributions using the active repeated-list control; do not weaken the frozen thresholds or
repeat whole-term deletion.

This rung splits the DOWNSTREAM READ of T edge by edge. T is a single vector at one position of a residual stream that is
affine in T (the block skip x ← λ0·x + λ1·x0 scales a block-8 write by the product of the later λ0's); every later component
reads the residual through its own RMS-norm. Path patching therefore has an exact form: carry T alongside the native residual,
scaled identically, and subtract it from the INPUT of a chosen set of readers while leaving it in the residual for everyone
else. The readers of T are: the final norm → unembedding (edge DIRECT); mlp8; attn9…attn17; mlp9…mlp17 (19 component reads).

Mechanistic hypothesis under test: T is an embedding-derived COPY of the final label (its value is λ_8·W_V^{(0)} of the label's
layer-0 state, which knows only the token identity); the "+1" is COMPUTED from T by downstream bilinear readers, and it is those
readers — not T's direct unembedding — that carry both the list-successor effect and the anti-copy collateral on repeated lists.
The alternative (null story) is that W_O^{(8)}W_V^{(0)} already maps label → successor direction and the unembedding reads it
directly, i.e. T is a compiled successor table and the downstream blocks are passive.

## Frozen inputs (SHA256 pinned in the script's HASHES; checked at dry run and at run)

- increment_two_hypothesis_rows_rung567.json (rows; FIT and SELECT only; families: list_two_line_state_shift,
  list_three_line_state_shift, list_surface_preserved, list_middle_index_break, list_step_two_conflict = the five list
  necessity families; list_repeated_index_control = the active copy control). FINAL_TEST and OOD are never read.
- numeric_factor_removal_positions_rung575.json (query_position, source_position per row endpoint).
- numbered_list_cached_value_weight_removal_rung576_results.json (R576's per-cell FIT numbers, used ONLY as the
  instrument reference for pred_a; nothing is refit from it).
- This preregistration; the checkpoint blob.
- Code imported, not copied, from Codex's frozen modules: rung576 compiled_cached / projected_terms / candidate_ids / margin /
  ce / batch_endpoint / chunks, rung573 replay_attention. Native replay and the compiled T are the same objects R576 used.

## Arms (all on FIT and SELECT; each row endpoint scored at its own query position)

NATIVE; FULL (T removed from every edge — identical in exact arithmetic to R576's whole-term deletion); DIRECT (T removed
from the final-norm read only); READS (T removed from all 19 component reads, kept for the final norm); COMP_x for each of the
19 component reads singly; BLOCK_j for j = 8…17 (block 8 = mlp8 only; otherwise attn_j and mlp_j jointly); TOP2_JOINT (the
two components with the largest FIT pooled mean margin damage among the 19 COMP arms, removed jointly — chosen on FIT, scored on
SELECT). 34 arms. Price ≤ 34 × 576 row-endpoints ≈ 19.6 k short-row forwards, zero backwards, zero fitted parameters.

## Measurements

m(x,a) = z_a − max_{b∈N, b≠a} z_b over R576's registered numeric candidate pool N (rung576 candidate_ids); margin damage
d_m = m_NATIVE − m_arm (POSITIVE = the arm hurts the correct answer); CE change d_CE = CE_arm − CE_NATIVE (NEGATIVE = helps).
Cell = family × endpoint (FIT n = 32, SELECT n = 16). Pooled list = mean of d_m over the 10 list-necessity cells of a split.
Bootstrap lower = 2.5 % quantile of 2,000 resampled means (seed 2808). Share(arm) = pooled d_m(arm) / max(pooled d_m(FULL),
0.5) — FULL is known positive (R576 bootstrap lower ≥ 2.03 in every cell), the floor only guards a broken instrument.
Direct lens: ℓ_T(v) = ⟨T, W_U[v]⟩ / rms(x_final) at the row's query, pre-softcap (the linear direct-path logit contribution);
copy-over-successor fraction = fraction of list-target row endpoints (FIT and SELECT, both endpoints, five families) with
ℓ_T(final label) > ℓ_T(correct answer). Zero forwards beyond NATIVE.

## Predictions (bars stated with the wobble they include; each with worked example and operand signs)

BARS = {"exact_tol": 1e-8, "a_tol": 0.02, "b_reads_share": 0.6, "b_direct_share": 0.4, "c_top2_share": 0.5, "c_reads_floor": 0.5,
"d_help_floor": -0.05, "d_frac": 0.5, "e_copy_frac": 0.75, "floor": 0.002}
NULLS = {"b_direct_share": 0.6, "c_top2_share": 0.3, "d_top2_ce": -0.05, "e_copy_frac": 0.25}

- pred_a_instrument_full_reproduces_r576: my NATIVE logits reproduce the facade native forward to relative squared error
  ≤ 1e-8 on the first FIT chunk, AND in all 12 FIT cells |mean d_m(FULL) − R576 mean_margin_damage| ≤ .02 AND on the two FIT
  repeated-index cells |mean d_CE(FULL) − R576 mean_ce_increase| ≤ .02 (R576 evaluated FIT only — its evaluated_splits is
  ["FIT"]; this rung is the first to open SELECT for these list families, and SELECT carries preds b–d as registered). Worked: R576 two-line/base FIT 2.4012 → FULL
  reads 2.40 ± 1e-3 (fp32 wobble only, deterministic rows). Null story: a wrong λ0 scaling or a wrong position gives ≥ .1.
- pred_b_downstream_readers_carry_the_successor_effect: on SELECT, Share(READS) ≥ .6 AND Share(DIRECT) ≤ .4. Signs: FULL > 0
  (≈ 2.2); hypothesis: READS ≈ 1.5–2.2 (share .7–1.0), DIRECT ≈ 0 or NEGATIVE (a copy push that, removed, HELPS the +1 answer) →
  TRUE. Null (compiled successor table): DIRECT ≈ 1.8 (share ≥ .6), READS ≈ .3 → FALSE; null met if Share(DIRECT) ≥ .6. If both
  shares exceed .6 (strongly non-additive), pred_b is FALSE and the sum is reported.
- pred_c_top2_readers_concentrate_the_effect: on SELECT, d_m(COMP_top1) + d_m(COMP_top2) ≥ .5 × max(pooled d_m(READS), .5)
  AND each of the two has a positive bootstrap-lower pooled mean d_m on SELECT. Signs: single-component removals may be
  negative; the two are chosen by FIT signed mean. Worked: READS 1.6; hypothesis mlp9 .6 + mlp10 .3 = .9 ≥ .8 → TRUE. Null
  (diffuse reading): 19 components each ≈ .08, top-2 ≈ .2, share .13 → FALSE; null met if top-2 share ≤ .3. If READS < .5 the
  floor makes the bar ≥ .25 absolute and pred_c is expected FALSE (the effect is then not downstream at all).
- pred_d_same_readers_carry_the_repeated_list_collateral: on the SELECT repeated-index control (both endpoints pooled), mean
  d_CE(FULL) ≤ −.05 (T's removal still HELPS copy on SELECT) AND mean d_CE(TOP2_JOINT) ≤ .5 × mean d_CE(FULL). Signs: both
  negative under the hypothesis. Worked: FULL −.31 (FIT value) → bar −.155; hypothesis TOP2_JOINT ≈ −.20 → TRUE. Null (the
  anti-copy push arrives by the direct edge or other readers): TOP2_JOINT ≈ 0 → FALSE; null met if mean d_CE(TOP2_JOINT) ≥ −.05.
  If d_CE(FULL) > −.05 on SELECT, pred_d is FALSE and marked not-applicable (the collateral did not replicate).
- pred_e_t_direct_lens_reads_copy_not_successor: copy-over-successor fraction ≥ .75 (T's own direct unembedding favours the
  final label over its successor). Worked: hypothesis: T ≈ embedding copy of "22" → ℓ_T(22) > ℓ_T(23) on ~all rows → .95 →
  TRUE. Null (successor table): ℓ_T(23) > ℓ_T(22) on most rows → fraction ≈ .05; null met if ≤ .25.

Scoring is exactly as written; failures are preserved; a correction that would flip a conclusion needs an independent physical
control. Nothing here installs into the §312 frontier; the ledger's explained fraction is unchanged by this rung.

## What a held result licenses (proposal for Codex's record; I do not edit circuits/*.json)

If a, b, c, e hold: the mechanism statement "attention 8 (H3, H7) copies the final label's identity to the final position; the
named top-2 downstream bilinear readers compute the successor from it" enters the record as a downstream-read split with
component-level sites, and d tells whether the same readers explain the failed R576 selectivity (T is a context-blind label copy;
the +1 push it triggers is what a repeated list has to overcome). If b fails with its null met, T is itself the successor object and
the next split is source/action on W_O^{(8)}W_V^{(0)}. Item 2 (numeric-sequence carrier split) follows either way.

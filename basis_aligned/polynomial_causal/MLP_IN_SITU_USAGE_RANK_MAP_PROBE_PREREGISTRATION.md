# In-situ MLP usage-rank map, all 18 blocks (real activations, CPU) — preregistration (Claude CPU lane)

**Registered:** 2026-09-03 15:00 UTC. **Owner:** Claude. **Script:** `ops/mlp_in_situ_usage_rank_map_probe.py`. CPU only.
Price is NOT zero forwards: one exact full-model CPU forward over 2 x 192 docs x 257 tokens (tt_model semantics) to
collect per-block states. No GPU, no backwards, no parameters. Written before any number below was computed.

## Why

§2673/§2675/§2676 gave the EXACT weight-space rank map (token-context operator family effective rank 438-749 in every
block; MLP0 context branch 929), with the stated caveat that the token-embedding second moment is a proxy input for
blocks >= 1. R538 (Codex) just found that NONE of the MLP product states 8-14 is a common live site for pending-opener
(.1-.2 logit movement vs 3-4.7 at residual / L13H8 sites). The natural question both raise: how many dimensions does
each MLP block actually USE on real text — the in-situ effective rank of its product state and of its output write —
and does the weight-space map predict it?

## Object

Sampled positions 1..255 step 4 (64 per doc) x 192 docs = 12,288 per corpus (natural fit_natural.pt; code ood_code.pt).
Per block l = 0..17, centred sample covariances of: the MLP output write m_l = Down(g_l) (1152; the constant Down_bias
drops out under centring), the product state g_l = (L xhat)*(R xhat) (4608; float32 accumulation), and the attention
write a_l (1152). Effective rank = exp(entropy of the normalised eigenvalue spectrum); rank_90 = smallest k holding 90%
of the trace; mean write energies ||a_l||^2, ||m_l||^2 per position.

## Predictions (scored as written; natural corpus scored, code reported)

- **pred_a_instrument** — my manual CPU forward's mean per-token CE on 4 natural docs equals `jacclust.tt_model.GPT`
  (same fp32 weights, same CPU) within 1e-4 nat; checkpoint sha256 680d6c26...; >= 12,000 samples per corpus; every
  covariance PSD (min eig >= -1e-6 max, float32 accumulation).
- **pred_b_no_low_rank_mlp_output_in_situ** — min over the 18 blocks of the natural in-situ effective rank of m_l is
  >= 100. Null: any block <= 50.
- **pred_c_weight_map_predicts_in_situ_usage** — Spearman rank correlation across the 18 blocks between §2675's frozen
  `eff_rank_entropy` (all_mlp_operator_family_rank_results.json, e237ca67...) and the natural in-situ effective rank
  of m_l is >= .5. Null: <= 0. I do not know this answer; the weight-space object (operator family span) and the
  in-situ object (activation covariance) are different, and a null here would mean the exact map does not order
  real usage.
Report-only: g_l and a_l effective ranks and rank_90 per block; attention/MLP write-energy ratio per block; code corpus;
the natural CE of the real model on these rows (sanity, ~3 nat expected).

## Price

2 x 192 full-model CPU forwards (fp32; ~1e14 flop), 18 x (4608^2 + 2 x 1152^2) float32 covariance accumulations from
12,288 samples per corpus. Estimated 6-12 min CPU, ~6 GB RAM.

## What each outcome licenses

No circuit claim. b TRUE: no MLP block writes through a low-dimensional channel on real text — consistent with the
exact map and with R538's MLP-site null (a diffuse high-rank write is a poor single interchange site). b FALSE: an
in-situ low-rank block exists and becomes the first candidate for a "smaller than an MLP block" DAS site — recorded
as new information, not a contradiction of the weight-space map (different objects). c: whether the weight map can be
used to rank blocks for finer-grain work without forwards.

**Correction (15:06 UTC, post-registration, pre-result).** First run exited 1 before any prediction was evaluated:
the instrument slice fed 257 input tokens against 256 targets (`nat[:4, :T]` vs `nat[:4, 1:T+1]`, rows have 257
columns). Fixed to `rows[:, :T-1]` / `rows[:, 1:T]` (256 inputs, 256 targets) in both the instrument and the
collection loop; smoke-tested on 2 RANDOM-token docs (manual CE 12.7747 == module CE 12.7747). Predictions, bars,
nulls, and price unchanged; script sha256 now bf1ec4ecd88d5145 (prereg hash frozen in the script updated to this corrected file). No registered data were read by the failed run.

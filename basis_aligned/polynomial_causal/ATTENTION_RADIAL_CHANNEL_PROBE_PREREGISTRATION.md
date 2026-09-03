# Attention radial-channel probe — preregistration

Registered 2026-09-03 20:04Z (box clock), before the script exists. Lane 1 (CUDA; the §2704 GPU/CPU agreement is 1e-5, so
device is no longer a caveat). SIGN CONVENTION (§2135): every number is CE ADDED ABOVE THE REAL MODEL on held-out docs 96–159 —
LOWER IS BETTER. Descriptive; nothing installs into the §312 frontier.

## Motivation (from §2704, radial_gauge_map_probe_gpu, sha 248a5af4…)
Dropping the RADIAL component (r = w·x̂ along the pre-write residual direction) of the attention-1 write costs **+5.28 nat** and of
the attention-5 write **+3.29 nat**, while DOUBLING the same component costs only .047 / .191. Every other attention site is
≤ .14 for either arm. attn5's write is the program's known "price cliff" (top gap list). The attn1 write is rank-≈22 (§2696 eff
rank 21.8; k=8 truncation costs .066) yet its radial part is catastrophic to remove. Hypothesis: at these two sites the write's
function is largely a per-token RADIAL SCALAR (a norm gate on the residual), which would be a one-number-per-token description of
the price cliff. Null hypothesis: the tangential part carries the function and the radial part is an inseparable by-product.

## Design
Same forward / split as radial_gauge_map_probe_gpu (x̂ = pre-write unit residual; r = w·x̂; w_perp = w − r x̂), eval docs 96–159,
chunk 8, one site at a time. New arms:
- **RADIAL_ONLY**: w' = r x̂ (tangential part dropped) — all 36 sites.
- **RADIAL_MEAN**: w' = r̄_site x̂ + w_perp, r̄_site = mean of r over all token positions of FIT docs 0–95 (label-free; no eval
  data used) — all 36 sites.
- **DROP_RADIAL** at attn1 and attn5 only (reproduction of §2704).
Signed radial statistics recorded per site on the eval docs: mean(r/|x|), fraction of tokens with r < 0, quantiles .1/.5/.9.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of §2704-GPU 3.1125031; DROP_RADIAL(attn1) reproduces 5.2819 and
  DROP_RADIAL(attn5) 3.2929 within .015 each; RADIAL_ONLY + DROP_RADIAL identities not required (different arms).
- **pred_b_radial_scalar_carries_the_cliff**: RADIAL_ONLY(attn1) ≤ 1.0 nat AND RADIAL_ONLY(attn5) ≤ 1.0 nat (each ≤ 0.3 × its
  DROP_RADIAL cost). Null: RADIAL_ONLY ≥ DROP_RADIAL at either site (tangential is the functional part).
- **pred_c_per_token_gate_not_constant**: RADIAL_MEAN(attn1) ≥ 0.5 AND RADIAL_MEAN(attn5) ≥ 0.5 (the scalar must vary per
  token; a constant rescale does not do the job). Null: both ≤ 0.1 (a constant norm factor suffices — an even simpler story).
- **pred_d_norm_shrinking**: mean over eval tokens of r/|x| < −0.10 at attn1 and at attn5 (the write shrinks the residual;
  the drop/double asymmetry of §2704 is consistent with a negative radial component). Null: mean r/|x| > 0 at either.
- **pred_e_specific_to_attn1_attn5**: among the other 16 attention sites, RADIAL_ONLY ≥ DROP_RADIAL (§2704 values, frozen) at
  ≥ 14 of 16 — the radial channel is a feature of these two sites, not of attention writes generally. Null: ≤ 10 of 16.

## Price
64·(36 + 36 + 2 + 1) + 96 (fit r̄) + 8 = 4,904 GPU doc-forwards ≈ 45–60 s on lane 1. Output attention_radial_channel_probe_results.json.
Frozen: this file, radial_gauge_map_probe_gpu_results.json (248a5af49328eb21f69acb5d8a8de3c7c9eef380ea7ae4ebf9b61e3e7c0fa063),
site_write_pca_truncation_ce_map_probe_results.json (48bd52ec…), checkpoint, fit_natural.pt.

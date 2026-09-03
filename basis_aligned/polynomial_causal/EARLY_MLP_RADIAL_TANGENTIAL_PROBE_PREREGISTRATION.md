# Early MLPs 0-3: is the write's price in the residual RESCALE (radial) or in its DIRECTION (tangential)? (Claude, CPU)

Registered 2026-09-03 18:56 UTC (box `date -u`), BEFORE running. Script: `ops/early_mlp_radial_tangential_probe.py`.
Results: `early_mlp_radial_tangential_probe_results.json`. Price: CPU only, 0 GPU forwards; ~1,200 full CPU document forwards
(96 fit-doc forwards collecting tangential covariances; 64 baseline; per site 4 identity + 4 x 64 arm forwards) — est. 9-12 min
alone at 16 threads. Source: §2700 (early price is a fat-head effect; ~256-d subspaces suffice) + §2699 (early writes largely
radial w.r.t. the post-write residual, a reference that is trivial when |w| >> |x|) — HOURLY_STRATEGIC_REVIEW_2026-09-03_1848 #4.
SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out natural docs 96-159 (baseline 3.11250,
the §2696/§2700 set; bases fitted on docs 0-95) — LOWER = better. No installation into §312.

## Objects and formulas (arms named)
At MLP block l in {0,1,2,3}, position t: x = the PRE-write residual (after the block's attention write, before the MLP write),
xh = x/|x|; w = the MLP write. Decomposition w = r xh + w_perp, r = w.xh (RADIAL = a pure rescale of the current residual),
w_perp = w - r xh (TANGENTIAL). Tangential PCA: mu_perp, U_perp = mean and descending eigenvectors of the covariance of w_perp
over fit docs 0-95, all positions. Arms (patch replaces w, Down_bias untouched):
- IDENTITY: w' = r xh + w_perp (instrument; must be exact).
- DROP_RADIAL: w' = w_perp.
- RADIAL_ONLY: w' = r xh.
- RAD_EXACT_TAN_k: w' = r xh + mu_perp + U_k U_k^T (w_perp - mu_perp), k in {64, 128}.
Disclosed per site: radial energy fraction mean_t r_t^2/|w_t|^2, tangential covariance effective rank and rank-90, and the
§2700 frozen plain-PCA k=64/128 prices for comparison (read from the frozen json; not recomputed).

## Preregistered predictions (scored exactly as written)
- pred_a_instrument: (i) unpatched CE reproduces §2700's frozen `baseline_ce_eval` (3.1124951) within 1e-4; (ii) IDENTITY at mlp1
  adds |CE| <= 1e-4 on 4 eval docs.
- pred_b_radial_is_functional_mlp1: DROP_RADIAL at mlp1 adds >= .30 — the residual rescale carried by mlp1 is functional, not
  gauge (the next block's attention input is rms-normed, but the residual's scale sets how later writes mix). Null: <= .05.
- pred_c_tangential_cheaper_given_radial_mlp1: RAD_EXACT_TAN_64 at mlp1 adds <= .15, i.e. keeping the single radial scalar exactly
  more than halves the plain k=64 price (.3567, §2700). Null: >= .30 (the radial scalar buys nothing over plain PCA).
- pred_d_radial_only_insufficient_mlp1: RADIAL_ONLY at mlp1 adds >= .50 (the direction carries most of the function). Null: <= .10.
Disclosed, not scored: all four arms for mlp0, mlp2, mlp3; RAD_EXACT_TAN_128 for all four sites; radial energy fractions;
tangential effective ranks.

## Null model / what a failure means
pred_b null: the early radial component is inert (a true gauge freedom through the rest of the network) — then the early
compilable object is tangential only and one scalar per position can be dropped. pred_c null: the residual rescale is not what
makes plain PCA expensive at k=64; the fat head is tangential. pred_d null: mlp1 is essentially a per-position rescaler — a
one-scalar program. Failure combinations are each informative; none reopens a closed item.
Frozen: checkpoint 680d6c26…, fit_natural.pt 666a3201…, §2700 results json (hash frozen in the script). No GPU.
Expected (priors): b TRUE, c uncertain (registered as the falsifiable claim), d TRUE.

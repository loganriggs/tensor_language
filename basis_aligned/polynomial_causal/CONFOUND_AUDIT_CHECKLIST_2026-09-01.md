# Confound-audit checklist for the 08:40 compare phase (prepared 02:58Z, Claude)

Standing rules distilled from tonight's failures — each scorecard row gets checked against all seven.

1. R²/fit-quality is NOT a structural discriminator (298B: wrong prior fit to 0.9999992).
   Require: intervention transfer, OOD behavior, or priced compression beating the dense baseline.
2. CONFIG IDENTITY: verify the defining construction (index sets, active lists, priors), not summary
   stats (the 290 contiguous-vs-mixed lesson).
3. PRICES are standalone dependency bills (total scalars + bytes), never deltas from unaudited anchors
   (the 295 lesson: "180M" configs actually stored 596M).
4. HIDDEN COMPONENTS: enumerate every table/hook the artifact executes (the a1v/a0 lesson — one legacy
   table was 92% of the "irreducible floor" and threaded globally).
5. TRIPWIRE VALIDITY: no scalar-proximity inert checks where the null predicts the anchor (§2359);
   live config tripwires on every variant (§2371); off-knobs are as unverifiable as on-knobs (§2383).
6. POSITIVE AND NEGATIVE CONTROLS: any recovery/structure claim needs a planted positive control AND a
   false-positive-rate control; a trained-student control if the claim is about learned weights (298's
   student arm: functional structure invisible in gradient-trained supports).
7. SIGNED VECTORS for intervention/effect comparisons; unsigned member means introduced a fake two-regime
   law once already (§2365-§2374 arc).

Cross-references for the scorecard's "identifiability" column: km-codebook saturation (§2326/§2327),
router-state negatives (298), support-index information cost (explanation_0239).

## Draft per-direction audit (pre-filled 03:30Z from the five screens landed; final pass at 08:40)

| Direction | Screens | Checks passed | Open confounds | Draft verdict |
|---|---|---|---|---|
| 1. MLP0 structure | 298/298B | pos+neg controls ✓, student control ✓ | R²-cannot-discriminate (298B) applies to ANY future positive | KILL (weight-support route); functional question moved to D5 |
| 2. Shared MLP atoms | 299 | pos ~1.0 / neg ~0.02 controls ✓ | none | KILL (coefficient grain; agrees with §2312) |
| 3. Vocab factorization | 300/300B | two-corpus ✓, label-free ✓, priced ✓ | TAIL TRANSFER (rare-token damage rises under freq weighting); post-result metric — bars must re-freeze | LIVE with conditions (hybrid head/tail pricing; certificate battery unchecked) |
| 4. Causal-response coords | 301 | two-corpus ✓, priced ✓ | none | KILL as proposed; SEED: activation-PCA r256 (MLP0 76% @ +0.021) — composition + full-bill checks required before any extrapolation |
| 5. Predictive state | 302/302B | toy control ✓, transfer ✓, REPAIRED live shuffle ✓ | R²-level state claim withdrawn under live null; no price constructed | PARK (behavioral state + head-13.8 effect survive; representation-level claim dissolved) |
| 6. Error contracts | 303 | prospective split ✓, control power-law ✓ | bound width 3.41x = vacuous for gating | PARK (use as ranking prior only; Spearman 0.92) |

Standing cross-refs: composition super-additivity (§2329/§2330) applies to ANY multi-module extrapolation;
full standalone bills (rule 3) for every candidate; certificate battery is the tail-sensitivity detector.

## Final compare-phase pass (03:47Z, run early to match Codex's early compare)
Scorecard SIX_DIRECTION_COMPARISON_2026-09-01.md: CONFOUND AUDIT PASS. Rankings match the independent
draft table; adoption gates subsume all seven checklist rules. Post-compare addendum: rung 304's null
(rare-row residual dead; tail is diffuse, selections near-random-overlap) reverts the top route to
shared-512 without tail patch — certificate battery (gate 2) is now decisive for direction 3.

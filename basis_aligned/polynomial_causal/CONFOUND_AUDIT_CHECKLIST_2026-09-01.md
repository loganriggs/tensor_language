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

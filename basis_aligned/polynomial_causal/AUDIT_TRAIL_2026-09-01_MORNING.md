# Red-team audit trail — 2026-09-01 morning arc (rungs 318–330)
Staged for the morning synthesis. Every rung scored against its frozen bars as registered; sign convention §2135 throughout: census/OOD/fresh numbers are CE ADDED ABOVE NATIVE — LOWER IS BETTER. All receipts verified by Claude within minutes of landing; no bar was relaxed post hoc anywhere in the arc.

| Rung | Claim | Outcome (as written) | Verdict / law fed |
|---|---|---|---|
| 318 | All-layer weight-SVD input screen (p512/p768 × 18 layers) | pred_a,c HELD; pred_b FAIL; NULL FIRED (L16 +1.21) | Banded law: 13 mid-stack layers qualify; L15–17 explode. Energy capture (~0.81 all layers) predicts nothing |
| 318B | Late-depth control (p1152/p1024 at 15–17) | pred_a HELD (p1152 exact ≤4e-8); b,c FAIL; NULL FIRED | Cliff at spectral tail, not capacity ramp; instrument clean |
| 319 | Banded composition {0,7,14}@p768 | pred_a FAIL (25<35 certs); b,c HELD | Tax 1.2×; cert collapse = one-dim damage law. Mapped, not adopted |
| 320 | Final pairs {0,7} vs {0,14} | pred_a FAIL by ONE cert (37<38); b,c HELD | Exchange-rate map completed: ~1.5–1.9 certs per +.001 census. Route closed by stop rule |
| 321 | Context-metric late screen (RRR under contextual covariance) | ALL HELD | CONTEXT-METRIC LAW: L15–17 repaired 100–300× solo — Frobenius cliff was a metric artifact |
| 322 | Late triple composition {15,16,17}@ctx-p768 | ALL FAIL; null not fired | Bar was below tax floor (additive .0315 vs bar .020) — registered impossibility. Measured tax 1.30× |
| 323 | Sequential closed-loop refit | ALL FAIL; NULL FIRED | REFIT NULL: ratio 1.0033, 0 certs recovered — tax is irreducible interaction, not stale covariance |
| 324 | MLP0 context-metric screen (p512/p640) | ALL HELD | Context metric beats weight SVD 3–7× at MLP0 too; pre-hoc tax band posted (+.0095–.0128) |
| 325 | Physical composition p512/p640 | ALL HELD | Landed inside pre-hoc band; taxes 1.34×/1.32× — TAX ~1.3× CONSTANT (third measurement) |
| 326 | Shifted OOD both variants (after import-crash repair, 0 GPU lost) | ALL HELD | Census repro bit-exact; p640 transport gap .0003 (best ever) |
| 327 | Two-variant signed a16 gate | ALL HELD | THIRD ADOPTION: p640 = first strict-dominance replacement; p512 gated Pareto sibling |
| 328 | Lower-rank frontier p448/p384/p256 | ALL HELD (two exact-bar landings) | Rank frontier mapped; equal-bill equivalence observed (p256 ≡ failed triples at same bill) |
| 329 | Lower-rank OOD | pred_a FAIL (p384/p256 row-max); b,c HELD | Tail-row blowup at low rank = diffuse-tails in OOD; p448 alone advances; cert repro bit-exact |
| 330 | p448 signed gate | ALL HELD | FOURTH GATED POINT. Signed signature constant across all gated points (cos ~.993, rho ~.996) |

## Gated frontier (end of arc)
539,595,062/+.0047/54 ↔ 535,613,750/+.00826/52 (dominant, adopted rung 327) ↔ 534,286,646/+.01073/48 ↔ 533,623,094/+.01266/43

## Laws established or sharpened (with provenance)
1. **Context-metric law** (321, 324): input-map low-rankness is metric-dependent; contextual covariance ≫ Frobenius. Weight-energy capture is blind to functional load (318, 318B).
2. **Tax ~1.3× constant** (322: 1.30×, 325: 1.32×/1.34×; 319: ~1.2×): composition damage ≈ 1.3× additive for this family. Pre-hoc bands built on it landed dead-on (325).
3. **Refit null** (323): the tax is irreducible interaction — closed-loop refit recovers nothing.
4. **Cert slope / one-dim damage** (319, 320, 328): ~1.3–2.6 certs per +.001 census on every route; equal bills land at equal (census, certs) regardless of construction (328) — proposed "equal-bill equivalence" invariant, unregistered.
5. **Diffuse tails in OOD** (329): low-rank cuts transport on means but fail on rare rows; row-max is the binding OOD gate for aggressive compression.

## Open items for the synthesis
- Equal-bill equivalence: one registered test would make law #4's strong form citable.
- Context-metric ladder over the 13-layer mid-stack band (ranked #1 in my 0530 review) — untouched savings 30–70M scalars, ladder needed because one-shot additive ~.052 → ~.072 census.
- Context metric on the QK fine band {120..127} — same mechanism question, new site.
- Rung 330 awaits its ledger § (ledger at §2427).

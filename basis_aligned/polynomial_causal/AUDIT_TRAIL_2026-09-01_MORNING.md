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

## Addendum — rungs 331–347 (written 06:56 UTC)
Convention unchanged: CE added above native, LOWER IS BETTER. All scored as registered; zero relaxed bars continues.

| Rung | Claim | Outcome | Law fed |
|---|---|---|---|
| 331–333 | Context-QK96 (no hand fine band) + split/OOD + signed | ALL HELD ×3 | Fine band was a Frobenius artifact; first 62/62 config; dominance adoption over mixed104 parent |
| 334–335 | Dual-context QK96+p448 composition + signed | ALL HELD; tax 1.02× | Cross-family ~free (my 1.3× prediction WRONG — exposed two-regime tax) |
| 336–338 | QK88 screen + signed | ALL HELD | QK ladder gentle; retired p640-combo pre-gate |
| 337 | Cross-family discriminator (p512/p640) | ALL HELD | INTRA-FAMILY TAX LAW confirmed (~1.3× intra / ~1.02× cross, 4 measurements) |
| 339/341 | QK80 + signed | ALL HELD | Retired p512-combo AND adopted p448-dual by dominance |
| 342/344 | QK72 + signed | ALL HELD | Cert plateau at 54: cert slope is axis-dependent, not constant |
| 340/346 | Gauge commutant screens (PCA32; full exact fold) | Instrument HELD; real NULL FIRED ×2 | No combinatorial block structure at MLP0 in any gauge — modularity is spectral, not partitional |
| 343 | Third family: value r96 | pred_a FAIL (honest) | Value maps real but ~5–15× pricier/scalar than QK |
| 345/347 | QK64 + signed | ALL HELD | Convex ladder (~1.6×/rung); signed signature eroding monotonely (flagged) |
| ops | WikiText-2 OOD stream EXHAUSTED (286,177 tokens) | 2 pre-model crashes, 0 GPU lost | Corpus switch to WT-103 TRAIN (fingerprinted); overlap caveat → proposed audit rule #4 (skip steps ≥30,840) |

**Gated frontier (10 points earned, 5 survive as Pareto):** 535.1M/+.0012/62 ↔ 530.6M/+.0022/58 ↔ 526.1M/+.0033/54 ↔ 521.6M/+.0052/54 ↔ 517.1M/+.0082/50 — a pure context-QK ladder. In flight: QK56 (512.6M) on the new frozen corpus.

# Induction — the third mechanism (distributed-cooperative with identifiable heads)

**One line:** strong induction (synthetic score ~11.8) carried by a front chain with
identifiable-but-redundant heads; independent of content, but content is amplified when it fires.

## Established facts
- **Circuit:** attn0 writes prev-token/copy-source (§841/§877); L5 gates — ablating L5 collapses
  induction (synthetic 2nd-copy 1.29 vs 13.09; natural inductable 0.69→3.64, control L11 nothing;
  §877-878). L5's induction is head **L5H5** (pattern share 0.248, the model's highest; §1083),
  distinct from the 5.7 sink. Identifiable heads: **L2h5 top** (+0.123), L5h5, L8h3/h4, L3h8
  (§954) — but redundant: top head alone +0.12 vs collective front-6 +5.2 (§952-953); all-81
  single-head sum 0.63. Copying is NOT a localizable head-set (§649).
- **Strength & scaling (§880-885):** induction nearly free at inductable positions (CE 0.69,
  1% of loss budget); strength scales with size (18L 11.8 ≫ 12L 4.3) and normalization
  (sqrd12 8.6 > bilin12 4.3).
- **Generative validation (§1025-1026):** AB…A→B +7-8 nats, position-specific,
  architecture-general.
- **Coupling with content (§1027-1032, six controls):** ASYMMETRIC — induction independent of
  content (0.97); content amplified ~1.5-2× when induction fires, via induction attention
  carrying nearby content (spatial) + a global boost; real at logit level.
- **Routing (§983-984):** induction is one of the two range-robust routing modes (with recency);
  front/mid-peaked (peak L5, gone by L11+).
- On prose CE, induction heads cost little (L5H5 zero-abl 0.009; §1083) — induction binds on
  repeated structure, not average text.
- **Read-grain map, copy regime (§1204-1205, W=64 read-masks on verbatim-repeat rows):**
  copying is a SERIAL chain across bands — front (1.91) and mid1 (2.64) each alone
  catastrophic, band sum 1.47× the joint 3.20 (opposite of prose pooling's redundant crowd
  §1186-87). The distance-128 READ belongs to **L2** (0.53; head 2.5 alone 0.62) — the L5
  gate barely needs range (whole layer 0.12; H5 0.086, though within L5 H5 IS the long-range
  reader, complement 0.023). Division of labor sharpened: **L2 reads the distance, L5 gates
  locally** (both §877-78 and this coexist: ablation kills the gate's output role, read-mask
  shows the fetch already happened upstream). The named pair jointly = only 26% of the
  circuit's long-range carriage — but see §1206-1208: the FULL station set does localize.
- **Reader stations (§1206-1208):** bilin18's distance reads concentrate in FOUR heads —
  **2.5, 3.8 (L3's entire read: alone 1.076, complement 0.004), 8.3+8.4 (redundant pair,
  87% of L8)** — jointly 69% of the whole-model copy-regime cost (3.20 @W64). §649's
  "non-localizable" is scope-corrected: that was zero-ablation on prose (output redundancy);
  read-range in the copy regime DOES localize. Family: swiglu18 pays the SAME total (3.206)
  with the same few-station structure but stations deeper (L5 1.03, L8 0.43, L4 0.40) —
  price+concentration are family law, station depth is the architectural fingerprint.
  Third member bilin12: SAME price again (3.231; §1212) despite induction 4.3 vs 11.8 —
  the ~3.2-nat copy read price @W64 is text-set, not machine-set; stations L5/L2; its late
  band NOT local (17% — depth compression pushes copy reads into the readout zone). Quad
  heads are NOT prose-dormant: 22% of the natural 0.176 read budget (§1211). Arc §1204-12
  closed.

## Benchmark status
Mechanistically mapped (MED-HIGH); no compact stand-in exists because the circuit is
distributed-cooperative (like everything here, §956).

## Gotchas
- Pattern salience ≠ causal contribution (§649). Prose CE underrates induction — evaluate on
  inductable positions or synthetic repeats.
- §877's "L5 = induction + content" is two different heads (H5 induction, H7 constant; §1083).

## Open
- Reconcile reader-heads with census name-circuit source-builders (FINDINGS legacy Open E) —
  low priority.

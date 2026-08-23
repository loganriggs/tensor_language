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

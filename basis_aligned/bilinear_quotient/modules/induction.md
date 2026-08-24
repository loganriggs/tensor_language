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
  heads are NOT prose-dormant: 22% of the natural 0.176 read budget (§1211; swiglu 16%,
  same ~0.037 absolute §1216).
- **MECHANISM (§1215-1218, arc capstone):** one algorithm, two implementations. bilin18:
  front stations 2.5/3.8 are SOURCE-MATCHERS (attend the query's own copy at offset 128;
  0.30-0.33 of far mass) and mid stations 8.3/8.4 (+redundant 5.5) are SUCCESSOR-FETCHERS
  (o=127) — matching is itself a long-range read, which is why the front band is
  catastrophic. swiglu18: NO source-matcher — all four stations pure fetchers (o=127,
  0.54-0.73); matching via local key-composition (textbook), hence its mild front band.
  Deciding case bilin12 (bilinear scores + row normalization): HAS matchers (L2 pair H1/H3
  at o=128; L5 pair H1/H5 at o=127) — **the score function (bilinear vs softmax) decides
  the implementation; the ~3.2-nat price is indifferent.** Station sharpness: bilin18
  single heads > bilin12 pairs > crowds (bilin12 late §1214, swiglu L8). L3H1 = auxiliary
  of the 3.8 matcher, toxic without it (§1213), diffuse offsets. Arc §1204-18 closed.
- **SIGNS + WEIGHTS (§1237-1240):** match-identity travels the embedding path, payload-identity
  the v1 broadcast (§1236-37 scramble dissociation). The criterion is WEIGHTS-READABLE: raw
  wte codes through 2.5/3.8's q/k pipelines separate same-token pairs perfectly — INVERTED
  (AUC 0.00; layer-mates 0.50). Algebra: mirrored signed conjunctions (one + branch, one −
  branch each; product negative 99.8-100%); in vivo the matchers deliver −(matched value)
  (−0.30/−0.42, 99-100% consistent); fetchers sign-split (8.4 +, 8.3 −). Signs LOAD-BEARING:
  flip 3.8 costs 1.43 > mask 1.08; both matchers flipped 7.81 (worst garbage-in recorded);
  differential-pair reading falsified (joint fetcher flip 5.21). Explainer page: circuits_explained.html.
- **THE MATCH-EVIDENCE AXIS (§1248-1253):** mlp3's re-encoding = one super-stable direction
  (split-half 0.97-0.995 all family). NECESSARY: removal 1.00/1.18/1.75 nats (bilin18/swiglu/
  bilin12; random null ~0; prose 28-71:1 spared) — three-family law, representation CONVERGES
  though matching algorithms differ. SUFFICIENT (bilin18): one-direction restore = 95.3% of
  matcher-mask damage. READ at blocks 4-5 (removal there 1.12; fetch band 0.28, 9+ nothing).
  SEMANTICS: per-row match LEVEL (72%) + positional pattern (23%); wrong pattern < bare level.
  Block-5 cleanup of the spent raw vector = GENERAL off-manifold normalization (§1247
  correction, not a dedicated service).

## Benchmark status
Front end (all four heads 2.5+3.8+8.3+8.4): **0.78 weights-computed stand-in** (§1257-60:
wte codes → matchers' own q/k pipelines → max far score → 2-param map → verdict axis, PLUS
the found source's successor v1 code → fetchers' c_proj slices × measured signed coeffs
(+0.190/−0.119). Matchers alone: 0.92-0.97 incl. unseen offsets (§1258). §1259's "payload
must run attention" RETRACTED in §1260 — the failure was ignoring the sign structure.
Remaining tail: the distributed 41% beyond the quad (§1209).

## Gotchas
- Pattern salience ≠ causal contribution (§649). Prose CE underrates induction — evaluate on
  inductable positions or synthetic repeats.
- §877's "L5 = induction + content" is two different heads (H5 induction, H7 constant; §1083).

## Open
- Open E RECONCILED (§1228): the §239 source-builders (attn0/1) are the pipeline's
  identity-writers — matchers' o=128 reads collapse 58-71% when their source-half writes are
  zeroed (placebo 1/1000). Full pipeline: attn0/1 write identity → 2.5/3.8 match → 8.3/8.4
  fetch → mid consumes. On repeat text both ends are symmetric (§239's 18%-site-local
  asymmetry is a natural-text property). Nothing further pressing.

## Annotation service resolution (2026-08-24, §1289-98)
- The §1228 "builders = attn0/1" layer-grain claim resolves at head grain: **heads 1.1 and 1.8**
  (layer 1), either alone sufficient at match sources (restore 70%/84%, pair 96.5%; leave-one-alive
  instrument, keep-none anchor null). CAVEAT §1299: the keep-none 4.33-nat baseline is ~70%
  GENERIC front-band damage (random-position floor 3.08); source-specific excess ≈ 1.25 nats.
  Rankings within the baseline stand; the pair may be layer-1's load-bearing heads generally. L0 = weaker collective backstop.
- Two DIFFERENT algorithms, one interchangeable output: 1.1 = identity mark (local read, 86%
  within ±8); 1.8 = context signature (76% far). Dissociated on planted context-free bigrams:
  1.1 restores 110%, 1.8 33%; adding 1.8 there HURTS (§1297). Mirror arm NULL: no pivot-free
  fuzzy induction (verbatim 7-token context + novel pivot → base = chance §1298) — matching is
  strictly token-triggered; the signature is not an independent matching pathway.
- Pair is sufficient-NOT-necessary (only-pair ablation ≈ free §1296); annotation redundancy
  spans the front band (§1287 superadditivity). Corrupted marks poison via mlp4's re-encoding
  (blocking mlp4 restores 94% §1288); same consumers: matchers, 13.8 (delimiters), 10.5 (questions).
- Head-partition law for this circuit (§1290-94): the payload is FULL-RANK identity — no
  weights-derivable subspace of a head is "the induction part" (route = 31%, 16-dim = null,
  per-position raw-code = broadcast term only). Partition variables (verdict axis), not pipes.
  Keeper instrument: per-position identity mask, 60:1 shuffle null. Long tail = identity crowd.

## Stem-matcher discovery (goal-3 loop, §1307-08)
- The matcher criterion mined for collisions returns MORPHOLOGY: show/shows, make/making,
  story/stories (192 pairs >= 0.5x identical median). The matcher is a **stem matcher**;
  induction = copy-across-word-family. Certified on natural text: matcher ablation costs
  0.186 nats at variant-supported targets (conc 4.5, n=1569), 78% of identical-supported
  damage — inside the weights-predicted 30-80% band; control pair 0.8%. First complete
  weights -> generalization-prediction -> causal-certification chain (user goal 3).

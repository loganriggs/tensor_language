# Deep-middle MLPs (L5-14) — the content machine (the settled frontier)

**One line:** one shared, drifting, load-bearing, genuinely high-rank content computation —
context×context multiplication over a semantically-organized topic/register manifold; universal
across the model family; register-adaptive within a model. §1049-1081 settled this; do not re-plow.

## Established facts (the §1049-1081 conclusion, plus mechanism history)
- **One object, not ten:** shared top-64 deviation subspace across L6-14 (pairwise 0.577 vs 0.054
  null; drift 0.65-0.82 adjacent → 0.32 at 6↔14; §1049). Low-rank bottleneck anywhere starves the
  band (greedy corrupted-stream fit fails; §1050-1051). Born gradually L3-5 (§1052); cumulative
  residual object, not authored by middle attention (§1053).
- **What it computes:** context×context bilinear multiplication (pooled content × itself ~70% of
  variance, ~all loss; token×token negligible; §1041); genuinely HIGH-RANK for the loss (needs
  ~full 1152 context dims for 90%; confirmed 3 ways §1000/§1038/§1042). Per-layer stakes tiny
  (mean-abl 0.04-0.05 each; §1084) — importance is collective. Held-out tok share L8 = 0.13,
  dev 0.53 (§1090): the function is context, full stop.
- **What the content IS:** high-dim semantic topic/register manifold, top axes interpretable,
  top-10 PCs ~12% var (§1055); genuine topic not surface (~2%; §1064); multi-scale, mostly
  running local context (§1065). Built from the PROCESSED stream (R² 0.38@L1→0.88@L7; §1073),
  not a raw-word bag (§1072); ultimately a pooled bag of block-0's static per-token c_v values
  (§1076) gathered by early attention (§1074).
- **Causal:** ablation catastrophic + privileged across the whole spectrum (§1056); activation
  patching transports topic (62×/22× random @K16/64; §1059-1060; replicated to the 4th decimal on
  fresh rows §1150). Transport is POSITION-BOUND (§1150, in-protocol): position-shuffled source
  coords lose ALL excess over the random-subspace null (0.565 < 0.669), per-sequence mean broadcast
  = zero excess (0.653 ≈ 0.669); rank ladder c8 0.453 / c16 0.578 / c64 0.768 / c256 0.899 —
  only ≥64 dims beat the equal-energy random patch outright. Grain questions must be asked at
  THIS locus with THIS readout (§1149 method law); supports content/topical-word
  prediction (K=16: rare targets 7.6× vs frequent, above 6.0× difficulty baseline; §1068 —
  §1067's "broad" claim was a K=256 destructive-regime artifact).
- **Universal:** same info across independently-trained models (CCA 0.95-0.97; §1061), across
  architecture (swiglu18→bilin18 96% §1062) and width (bilin12; §1063/§1066 93%). Front-linear /
  middle-nonlinear loss split universal (§1058).
- **Register (OOD code; §1079-1081):** content subspace register-SPECIFIC (prose↔code 0.19),
  grammar 2× more general (0.41 vs 0.20); on code the model shifts to local/per-token prediction
  (content band cost-frac 0.69→0.28, grammar 2.00→0.65, value-residual up 1.01→1.10).

## Benchmark status
~**0.10** as variables (§906/§940) — bounded by REAL high-dimensionality, not instruments
(FINDINGS item 5). Content passthrough triples whole-model recovery (0.12→0.39; §1071).

**Feature-circuit thread (§1128-1129, closed):** writers DIFFER per atom (attention-authored:
attn5/7/9/10/11 poolers; MLP-authored: mlp3/6/7 transition — §1074 refined) but the API is a
POPULATION CODE: single atoms fail deletion (§1104-style), dose-steering (§1128, 0/8, generic
content boost), AND value-interchange (§1129, 0/8; wrong-dir null sometimes beats own). Smallest
causal unit = the package's value vector (whole-pattern patching works, §1059-60/§1105). Do not
attempt single-feature interventions on the content API.

## Gotchas
- Ablations at K=256 enter a destructive regime (§1067-1068): use low K for specificity claims.
- Patching must match the removal point/scale (§1066 bug).
- The manifold is a continuum — discrete topic labels don't replicate, the geometry does (§873-874).

## Open
- Sparse features RESOLVED (§1113-1115): the content code = STABLE SPARSE SKELETON (k≈8, 70%
  of coord variance, stability 0.76-0.81, 3.6× PCA at low k) + DENSE TAIL (30% var, stability
  dies 0.37, §763's instability = the dense limit). **CAUSAL (§1115): the skeleton carries 78%
  of the per-layer content read's CE; the tail 12% (variance≠CE inverted).** Atoms entangled
  with each other (not one-by-one handles, §1113) but the SET is the function. Stream must stay
  full-rank (many cumulative readers, §1051); each reader's draw is skeleton-dominated.
  FINAL PICTURE (§1118): **skeleton = the API, tail = the scratch space** — reads sparse
  everywhere (tail 10-15% incl the 0.90-nat logit path §1117, which cross-checks §1082's 0.885);
  construction MIXED (stream-level: skeleton-removal 97%, tail-removal 46% — the tail works in
  transit though read nowhere). Resolves §1042/§1051/§1055/§1056/§1060 tensions in one frame:
  high rank is irreducible for SIMULATING construction, not for READING the result.
  CONSTRUCTION-SIMULATION BOUNDARY (§1122-1127, wave closed): end-band content = 53% linear
  transport of L5 state (per-step maps LINEAR, chaining lossless — §1123's compounding reading
  corrected in §1124) + 17% attention injections + 30% CREATED IN FLIGHT in L5-9 (not carried at
  any input width incl RAW-512 §1127; not coordinate-quadratic §1125). Linear/feature simulation
  caps at R²≈0.70-0.72; the remainder must be RUN. Do not re-attempt fixed-basis simulators.
  BAND-WIDE (§1116): scale-invariant — skel 13% / tail 72% of fullrem 0.34 across all ten deep
  MLPs. RELOCATION: deep MLPs are MINOR content consumers (0.34 nats total read vs stream-level
  +8.4 §1056) — main consumers downstream (readout local read §997 + attention re-pooling); deep
  MLPs maintain/multiply the object (write-side) more than they consume it. Skeleton atoms named
  (§1116): topic (livestock), register (political-analysis, technical-expository), discourse
  (partitive "Some of"), entity-density; 6/8 readable. THREAD CONCLUDED.
- Register axis CLOSED (§1079-1112): grammar>content generality 1.6-2.1x all pairs (no fixed
  constant, §1111); content band prose-specialized; value-residual most register-robust.

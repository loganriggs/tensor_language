# THREE-HOURLY MATHEMATICAL REVIEW — 2026-09-01 07:10 UTC
Context: the morning arc (rungs 318–351) established the context-metric law, the two-regime composition tax (intra ~1.3× / cross ~1.02×), a certificate cliff closing the QK ladder at r56, and two gauge-quotient negatives (no combinatorial block structure at MLP0 in any basis). Sign convention §2135: damage numbers are CE ADDED ABOVE NATIVE — LOWER IS BETTER.

## Top three mathematical moves (ranked)

### 1. EXECUTED — Certificate margin factor model (one shape, family-specific gain)
**Object:** the 62-vector of per-certificate member damages m_i(config) in every receipt, vs battery thresholds .5·ref_i.
**Math:** treat {m_i} as a matrix over configs; test rank-1 structure m_i(cfg) = s(cfg)·k_i (factor model / one-dimensional damage made quantitative). Cert count is then a first-passage count #{s·k_i < .5 ref_i} — a deterministic function of the scalar s.
**Result (13 configs × 62 certs, receipts on disk, CPU only):**
- Pure census-slaved rank-1 (s ≡ census) FAILS across families: R² = −3.73; it predicts 4 certs for the late-triple configs which actually hold 28.
- Shared-shape with free per-config scale: **R² = 0.81, mean |cert-count error| = 1.23** across all 13 configs spanning +.0047…+.0396 census and four construction families (parent, MLP0-rank ladders in two metrics, pairs/triples, late-layer).
- The scale s correlates with census at only 0.872 — the missing variance is FAMILY: late-layer damage buys ~5× less certificate harm per unit census (their CE lands on non-certified behavior).
**Refined law:** the 62 certificate damages lie on ONE ray (shape k_i is universal); configs differ by a scalar intensity whose census-gain is family-specific. LOFO confirms: MLP0-family fit transfers to pairs (±5 certs) but not to the late family.
**Operational consequences:** (a) cert bars can be set pre-hoc from a config's earliest member-damage measurement via s, to ±1–2 certs — no more straddled bars; (b) k_i is an explicit measured "damage axis" — the certificate-gradient direction as a vector, connecting to the §2363-era gradient-subspace measurements; (c) the r48 cliff (43→29) is the first-passage prediction of a near-threshold margin mass — visible in advance from margins.
**Cheapest falsifier (proposed, GPU-cheap):** compute member_abs_dce for ONE new off-ray construction (e.g., a value-family config) and check shape transfer; if R²-shape < .5 there, the ray is QK/MLP-specific, not universal.

### 2. Induced-metric Eckart–Young / water-filling rank allocation
**Object:** every replaced map W with input covariance C; the program's context-RRR is exactly the optimal rank-r approximation of W·C^{1/2} (generalized Eckart–Young). **Consequence beyond reconstruction:** given a total scalar budget across sites, squared-error-optimal allocation is water-filling on the singular tails of {W_j C_j^{1/2}} — predicting the optimal rank SPLIT between QK, MLP0, value families without ladder search. **Assumption that may fail:** CE damage ∝ activation MSE with site-independent constant (the family-specific gain found in #1 says the constant differs by site — so water-filling needs the measured per-family gains as weights). **Falsifier:** compute tail energies for the measured ladder (CPU, weights+fit cache on disk) and regress measured census surcharges on them; a fit licenses allocation by computation instead of search.

### 3. Prequential/MDL pricing of the staircase
**Object:** the gated frontier as a code: total description length = bill·(bits/scalar) + N_tokens·(census damage in nats). This makes "scalars saved vs CE added" a SINGLE objective with an exchange rate set by the deployment corpus size; the staircase's Pareto points become MDL-comparable, and the r56-vs-r64 choice is a corpus-size statement (crossover N where 4.5M scalars = .0043 nats/token). **Consequence:** artifact selection stops being taste. **Falsifier/cost:** pure arithmetic on existing receipts; the assumption at risk is bits/scalar (quantization not yet measured — a registered fp16/int8 cast rung would pin it).

## Pruned this cycle
- Gauge quotients/commutant search: closed empirically at MLP0 in both scopes (340/346) — do not revisit without a new invariant.
- Tail-reweighted covariance objectives: falsified (350).
- Hankel/automata state extraction: prior delimiter Hankel line closed; no new interface since.

## Execution note
Move #1 was executed this cycle (analysis above; script inline in this file's git blob). Moves #2/#3 are CPU-cheap and handed to the board as candidates for Codex's direction.

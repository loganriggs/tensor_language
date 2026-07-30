# Unsupervised circuit discovery — techniques per circuit TYPE (bilin18)
Logan's organizing insight: different circuit TYPES need different discovery + verification tools. This
catalogs the tool for each type. Core loop for all: (1) an unsupervised RANKING signature over
decomposition paths; (2) an out-of-sample TRIGGER confirmation; (3) a CAUSAL verification move (the
direct-to-logits proxy is unreliable — §56 — so effects need ablation/patching). Held-back FW[448:600],
paired standard errors, mean-ablation in-distribution zero point.

| circuit TYPE | discovery signature (ranking) | trigger side | CAUSAL verify move | status |
|---|---|---|---|---|
| **class-boost head** (trigger→boost a token class) | trigger-purity × output-boost-purity | attended-source / current-token class concentration | mean-ablate → the boosted CLASS logits drop | §56 (5 verified) |
| **copy / value-router** (output = attended token's identity, not a fixed class) | COPY-purity: head output aligns with the ATTENDED-SOURCE token's (or successor's) unembedding | attended-source token | mean-ablate → the source-token (or successor) logit drops specifically; value-swap flips it (§41) | qk_unsup_copy.py (running) |
| **suppression / inhibition** (push tokens DOWN) | SUPPRESSION-purity: concentration of most-NEGATIVE logit contributions | context where a token is wrong | mean-ablate → the suppressed tokens RISE (opposite of boost) | qk_unsup_suppress.py (running) |
| **QK-steering composition** (head A reshapes WHERE head B attends) | path-patch: ablate A → change in B's attention pattern; QK-side > OV-side | A's trigger | edge path-patch (B reads residual minus A) → B's attention reorganizes; specificity control (same A into other B = weak) | §57 (h.L4.0→h.L6.7) |
| **feed-forward feature-building** (MLP builds features heads read) | path-patch: ablate MLP → change in downstream head's read | — | edge path-patch MLP→head | §57 (mlp.L1→h.L6.7) |
| **successor / table lookup** (per-element memorized table) | — (found via behavior or copy-successor signature) | element identity | HELD-OUT element test (calibrated elements pass, held-out fail) | §35/§51 |
| **prior / in-context copy** (NOT computation) | — | — | STATIC-PRIOR control: mean-ablate ALL attention → behavior survives = prior; demo-swap for in-context copy | §40 |
| **artifact** (byte-fragment U+FFFD, attention-sink, high-norm SVD dir) | high proxy-purity but... | degenerate single-token | mean-ablate → ΔCE ≈ 0 (non-load-bearing) | flagged throughout |

## Key cross-cutting techniques
- **Trigger fingerprints are RELIABLE; output proxies are NOT** (§56): the attended-source/current-token
  trigger held out-of-sample in every case, but the direct-to-logits effect proxy was wrong in magnitude,
  sign, AND case (lowercase vs capital). Always read the OUTPUT from causal ablation, not the linear proxy.
- **Specificity control** (§57): same upstream into a different downstream must be much weaker (33× for
  L4.0→L6.7) — rules out "big-vector" magnitude artifacts.
- **Co-occurrence vs composition** (§57): the highest raw dependency scores were pure co-occurrence (edge
  path-patch inert); only a direct-edge patch distinguishes routing from correlated triggers.
- **Steering vs enabling** (§57): a genuine edge can REDIRECT a downstream head (attention reshaped) without
  ENABLING it (aggregate contribution unchanged) — report which.
- [TO EXTEND as copy / suppression / cluster tools return: their signatures + what NEW verification each needs.]

## Under-served circuit TYPES (from §58 auto-cluster) — new tools needed
| type | why current tools miss it | NEW tool needed | status |
|---|---|---|---|
| copy / induction | class-boost scores a FIXED direction, not "boost whatever you attended" | attended-source-token-in-output metric | **DONE §60** |
| suppression / anti-copy | effect-purity ranks only POSITIVE logits | signed / most-raised-on-ablation ranking | **DONE §59**: 2 late-FF class-inhibitors (mlp.L17.d1/L16.d0); NO anti-repetition head; suppression is diffuse/class-level/late-FF |
| positional / structural | routes by relative position / line-structure, not a content class | position-vs-content probe: per-head variance decomposition of raw bilinear pat into an OFFSET template (q-k) vs a KEY-CLASS template, with the DECISIVE metric = content-RESIDUAL (class variance beyond offset); + argmax structural targets (prev-token, back-k, pos-0 sink, attend-last-newline); causal bucketing of paired dCE by distance-since-newline | **DONE §62** (qk_unsup_positional.py): of 162 heads, 54 genuinely POSITIONAL (44 fixed-offset + 7 offset-envelope + 2 pos-0 sink + 1 line-structure), 0 CONTENT-by-class, 108 mixed/diffuse. Content-residual ~0 for all positional heads. Causal: fixed-offset heads load-bearing (prev-token h.L0.3 dCE +0.074±0.003, self h.L1.1 +0.030±0.002) with damage UNIFORM across line-structure (corr(dCE,dist-since-newline)≈0); the one line-structure head (h.L2.4 attend-last-newline) is causally NULL in isolation (−0.0003), as are §58-flagged h.L1.2/h.L2.1 (~0.001, redundant/diffuse). [RED-TEAM attack 4: the distance-since-newline metric is UNDERPOWERED against a saturating signal — h.L5.7 shows a 2.7× monotone rise the Pearson metric scores as 0.0 — so the strong negative is RETRACTED; the tool licenses the positive fixed-offset attributions, not a claim that no distance-to-boundary circuit exists.] |
| redundant / distributed | clean trigger but single-ablation dCE≈0 (duplicated across heads) | greedy JOINT/subset ablation + redundancy ratio (joint/Σsolo) + minimal-subset + same-size RANDOM-set control | **DONE §61**: copy family ratio 3.86 (distributed circuit, minimal 4-head subset 87%); diffuse newline cluster ratio 1.1 (genuinely null) |
| byte-fragment / orthographic trigger | trigger is a sub-word byte/char pattern (suffix, prefix, digit, punctuation, caps), invisible to content-class fingerprints | orthographic-predicate library scored on DECODED trigger strings (purity×lift), OUT-OF-SAMPLE purity guard on a disjoint slice (the built-in artifact pre-filter), then conditional causal contrast (damage on pattern-matching vs non-matching positions) | **DONE §63** (qk_unsup_bytefrag.py): 3 genuine circuits — digit-attending heads h.L8.7 (out-of-sample purity 0.90, causal 11-20× concentrated on digit positions) & h.L8.3 (0.97), punctuation head h.L13.8 (purity 1.00 in AND out, causal effect ENTIRELY on punctuation positions). Out-of-sample guard rejected rare-suffix/prefix/n-gram overfits (purity → 0.000) and MLP L9.d1 (pure detector, causally null = trigger-genuine/output-diffuse) |
| trigger-vs-output DECOUPLING (remap circuits) | fires on class A but boosts a DIFFERENT class B (article→noun, boundary→capital); the direct-to-logits proxy over-ranks these AND is known-unreliable | build trigger-class + output-class histograms per path, rank by decoupling×trigger-purity×output-purity as a CANDIDATE GENERATOR, then causally verify the OUTPUT side (does ablation suppress class-B specifically at active class-A positions vs an inactive-class-A control, paired standard errors) | **DONE §64** (qk_unsup_decouple.py): 67 candidates, top 6 causally tested → 3 GENUINE remaps, 3 PROXY-ARTIFACTS (one sign-INVERTED, z=−17.6). Strongest: mlp.L15.d2 punctuation→capital (drop 0.0068±0.0009, z 7.7, full control, load-bearing dCE +0.024). Twin directions split (mlp.L15.d1 vs L16.d1 both "newline→capital", mlp.L0.d3 vs L1.d3 both "determiner→word") — one genuine, one artifact each; only output-side causal test separates them |

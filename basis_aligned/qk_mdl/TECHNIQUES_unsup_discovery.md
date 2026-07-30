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
| copy / induction | class-boost scores a FIXED direction, not "boost whatever you attended" | attended-source-token-in-output metric | qk_unsup_copy.py (running) |
| suppression / anti-copy | effect-purity ranks only POSITIVE logits | signed / most-raised-on-ablation ranking | qk_unsup_suppress.py (running) |
| positional / structural | routes by relative position / line-structure, not a content class | position-vs-content probe (structural-attention purity, separate from vocab boost) | TODO |
| redundant / distributed | clean trigger but single-ablation dCE≈0 (duplicated across heads) — proxy OVERSELLS | greedy JOINT / subset ablation to separate null-from-redundant | TODO |
| byte-fragment artifact | masquerades as a circuit, wastes verification budget | a PRE-FILTER (route U+FFFD / lone-byte to artifact bucket before causal test), not a detector | TODO |
| trigger-genuine / output-diffuse | conflates "real feature detector" with "real algorithm" | DECOUPLE trigger-verification from output-verification (report VALID-DETECTOR w/o a clean output) | TODO |

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
  patching transports topic (62×/22× random @K16/64; §1059-1060); supports content/topical-word
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

## Gotchas
- Ablations at K=256 enter a destructive regime (§1067-1068): use low K for specificity claims.
- Patching must match the removal point/scale (§1066 bug).
- The manifold is a continuum — discrete topic labels don't replicate, the geometry does (§873-874).

## Open
- Nameable sparse features on the manifold (dictionary > SVD; FINDINGS item 16-era) — a naming,
  not mechanism, thread. Register axis continuations (`FINDINGS` Open C).

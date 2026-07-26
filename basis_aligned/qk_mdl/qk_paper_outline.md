# Paper outline (Logan request, 2026-07-26)

Working titles:
- "From weights to a memory circuit: exact decomposition and causal dissection of
  the first two layers of a bilinear-attention transformer"
- "Parametric memory has an address and a payload: a causally verified retrieval
  circuit in a softmax-free transformer"

## Sections and their headline claims
1. Introduction — softmax-free bilinear attention makes a 546M-parameter model
   exactly foldable; exactness first, MDL pricing throughout, causal interventions
   where exactness ends; ends at a verified memory circuit + surgical fact eraser.
2. The exact fold — layer-0 pattern = closed-form token-identity lookup (zero
   approximation); layer-1 port captures ~99% (+0.027 of +2.70). [Fig 1: fold schematic]
3. Archetype inventories — symmetric CP on 7/9 heads, permutation nulls, stability,
   cross-corpus; causal loads per archetype; heads 0/4 asymmetric long-tail pair;
   capacity Pareto frontier (k=2 sweet spot). [Fig 2: atlas excerpt + frontier]
4. Understanding metric U-v1 — faithfulness x compression, full-freight weight
   charging, anchors (verbatim 0, pointing ~0, law 1). [Table 1: scored ledger]
5. Named basis — 96 embedding code + 144 archetype activations = 51% of context
   gap at ~9 Mb; rules+exceptions toy predicts the knee shape. [Fig 3: explicit-
   object frontier with toy overlay]
6. THE MEMORY PIPELINE (headline) — 4 stages, causally verified, population scale,
   general (95% of positions strong-key): in-place enrichment (front-loaded MLPs,
   block 1 foremost; mid-stack redundant band); query-side addressing (distributed
   attention L3-8, complete by L8, independent key routes); late fetch (L13+,
   necessity 0.99); readout. Worked examples: Lindsay->Lohan, Beneath a Granite->
   Sky, Henrietta-period->Hamilton. [Fig 4: pipeline diagram with causal numbers;
   Fig 5: recovery-vs-depth curves, case studies + population]
7. Address vs payload; store geometry — deltas share no causally-potent subspace
   (linear-in-rank; 80-span 0.29 vs own 0.98 = per-fact orthogonal, hash-like);
   context-bound (transplant 0.04 vs control 1.00); position-addressed (relocation
   0.008 at one token; duplicates inert; correct slot 0.98). Query computes the
   address; key slot holds the payload. [Fig 6: rank-recovery with inert shared
   variance]
8. Single-fact eraser — one vector, one position, one layer: target -2.95 nats,
   collateral 0.006 (~500x), zero cross-fact interference, direction-specific,
   no weight changes. [Fig 7: drop-vs-collateral scatter + relocation inset]
9. Discussion — MDL frontier meaning of the orthogonal store (tail arithmetic);
   write-site vs read-site editing targets; transfer to softmax models; layer 2 open.

Methods appendix: patching battery spec (sites, directions, recovery/damage
normalization); pre-registered adversarial-advocate protocol (consensus twice
wrong, controls caught it); planted-model positive controls; frozen MDL conventions.

Scope rule: supporting negatives (datastores, self-distillation, rival advocates)
appear only as one-line evidence inside sections 5-7; no null gets its own section.

Figure sources: F1/F4 from qk_features.html sections 2 and 8; F2 from artifact
archetype cards + capacity sweep jsons; F3 from ladder/named-basis results; F5 from
qk_depth_sweep.json + qk_enrich_scale.json + qk_pipeline_generality.json; F6 from
qk_enrich_rank.json; F7 from qk_fact_eraser.json + qk_fact_reloc.json.

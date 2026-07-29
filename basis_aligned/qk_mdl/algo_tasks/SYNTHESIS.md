# Algorithmic task circuits: three-agent synthesis (2026-07-29)

Three independent agents decomposed three verified behaviors (bracket/quote closure, numbered-list
increment, successor sequences) via patching, DAS, and Ethan's data-conditioned weight reduction.
Full reports in bracket/, increment/, successor/. Cross-cutting findings:

## 1. The v1-router principle (now four independent occurrences)
Every task circuit's late attention heads turn out to be ROUTERS of the layer-0 value cache, via
the v-lerp (lamb(L8)=4.0 -> head value = -3*v_L8 + 4*v_L0): closure (L13H8 routes block-0 values;
own c_v irrelevant), increment (L8H7/H3, v1 term 0.89 vs own 0.02), successor (L8H3, zeroing L0
c_v drops task to 34%, zeroing L8 c_v keeps 98.7%), and induction (the match/copy arcs). QK decides
WHERE, layer 0 decides WHAT. An architectural principle of bilin18 forced by the v-lerp design.

## 2. Increment and successor converged on the SAME circuit
Two agents, blind to each other, different task framings: increment found L8H7+L8H3 -> MLPs 8-14;
successor found L8H3 (+L8H7 in top-10) -> MLPs 8-12/14. Increment IS digit-succession; the model
implements one succession machine. And L8H3 is atlas-induction rank 4/180: succession reuses an
induction-family head. One algorithm, three-plus lookup tables.

## 3. One algorithm, family-specific tables (activation AND weight level)
Successor: per-family DAS subspaces overlap geometrically (principal cosines 0.7-0.98) but do NOT
transfer functionally (weekday-trained flips weekdays 80%, months 0%); weekday-fit rank-16 W'
retains weekdays 99.9%, months 42%. Matches the program-level finding that lexical content lives
in keyed tables while the routing structure is shared.

## 4. Atlas sanity-check verdict (the original purpose)
All three task circuits are CONTAINED in atlas-important components but NOT resolvable from
single-knockout importance: Spearman 0.26-0.35 vs nearest atlas tasks, near-zero top-10 overlap;
redundancy hides task-critical heads (closure's L13H3 at atlas rank 145). Reading: the atlas
measures GENERIC competence (dominated by early-MLP enrichment); task patching isolates the
DIFFERENTIAL circuit. Consistent, different granularity -- the full decomposition survives the
sanity check, with the caveat that mean-ablation knockout under-resolves redundant task machinery.

## 5. Ethan's method verdict (three targets)
Data-conditioned W' = SVD_r(WX) X^+ beats data-free SVD on task-per-rank in all three circuits:
closure rank 2 vs 16; increment rank 16 vs 32-256; successor rank 16 vs 128. General-CE damage
depends on the target matrix class: QK read slices are near-free (<=+0.0025); the shared layer-0
value stream is destructive (+0.82) -- apply the method to read matrices, not shared value carriers.
DAS dimensionalities: closure ~1-4; increment ~4; successor ~16 (family tables).

## 6. Method lessons from agent failures (all reported honestly)
Pre-registered DAS sites failed twice because the v1 cache bypasses mid-stack residuals (the
failure itself localized the payload); a near-null weight target produced a vacuous rank-1
"success" until zero-ablation triage picked the real carrier; position-preserving corruptions make
structural patterns invisible to patching.

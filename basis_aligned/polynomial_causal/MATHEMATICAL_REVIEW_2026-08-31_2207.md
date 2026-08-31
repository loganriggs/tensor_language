# MATHEMATICAL REVIEW — 2026-08-31 22:07Z

Convention (§2135): all damage numbers are CE added above the real model; LOWER IS BETTER.

## Flagged audit: what "explained fraction" means after the grammar change
The strict ledger still quotes 5.348%/10.923%, 4.727 nat, 0/68 — built for the era of module-by-module
fitted replacements. That accounting is now OBSOLETE in structure, not just stale in value: the current
best registered config (r64, census +0.0852, 7/62 certificates) replaces NOTHING with fitted objects.
Its accounting is: 148 of 162 attention heads' four QK pattern maps SVD-compressed 2× (weights-only,
exact values/OV); 0 of 36 MLPs compressed at certificate grade (fixed-subset pruning collapses, §2303);
embeddings/lm_head untouched. The honest coverage statement: the program has one benignly compressible
object class (pattern maps, rank law r*≈√128) and one structural discovery about the rest (MLP usage is
per-token sparse and rotating: 1/4 density costs +0.016, but no fixed subset works). The 0/68 strict line
should be retired in favor of the certificate ledger (now 7/62) — recorded here, not edited into old §s.

## Where the mathematics now points
The night's data: (i) per-token top-k curve +0.016/+0.064/+0.184 at 1/4, 1/8, 1/16 density — conditional
computation, not static sparsity; (ii) the neuron basis is gauge (user directive) and the free
eigenfeature re-decomposition already shows a quarantined 0.77/0.65 advantage; (iii) exactness
certification itself needs precision-honest metrics (§2307's split verdict).

## Top three moves
1. **Symmetric Tucker/HOSVD of the invariant tensor** (multilinear rank; Q_d = interaction matrices,
   shared input basis W from Σ Q_dQ_dᵀ, per-direction cores WᵀQ_dW). The gauge-quotient canonical form
   with SHARING across output directions — exactly the "shared dictionary" object. Assumptions that may
   fail: usage may need per-direction bases (then CP/eigen wins). Measurable beyond reconstruction:
   matched-value comparison vs neurons AND eigenfeatures on m16's own circuits. Cheapest falsifier:
   rung 212 (EXECUTED, queued).
2. **Precision-certified exactness** (backward-error analysis for the certificate pipeline): a
   certificate is only as good as its exactness tripwires; fp32 4608-term cancellations put a ~1e-2
   floor under scale-free max metrics. Move: fp64 control + scale-honest bars as standing instrument
   rules. Falsifier: rung 211 (EXECUTED, queued) — if fp64 residual persists, it was a bug, not precision.
3. **Conditional-computation routing (sketch-gated top-k)**: convert the sparse rotating usage into
   executable cost — predict each token's active-unit set from a low-rank sketch of x (rank ~64) instead
   of computing all 4608 products. The mathematical object is a shallow arithmetic circuit with a
   selection gate; the measurable is census damage of sketch-selected vs oracle top-k at matched density.
   BLOCKED-BY-DESIGN until 211/212 land: route in whichever basis wins. Next rung after they land.

Pruned: Hankel/automata (no new sequential-state object; attention already handled by the rank law), MDL
reweighting (duplicates census accounting), any global/per-circuit additive bias (illegal frame mixing,
§2249/§2252), block-grain rank allocation (dead, §2302), further uniform-rank endpoints (curve bent, §2301).

## Executed
Rungs 211 + 212 queued (GPU, preregistered above); the audit paragraph above is the CPU-side deliverable.

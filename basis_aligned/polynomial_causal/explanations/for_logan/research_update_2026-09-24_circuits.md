# Circuits on bilin18 at the transport × multiplication level: 100 output directions share three transport heads and a core of ~40 block-17 units (24 September 2026, 01:30 UTC)

Claude (Fable). Lane opened on Logan's direction (23 Sep): distributed circuits are fine; find ~100, look for reusable clusters, include unembedding directions, run the finite-scale check, and make the description weight-space so it is not context-bound. Project `pr_dct_checks/` (plans `plans/CIRCUITS_PLAN_V*.md`, receipts `results/circuits_*.json`). Rungs are preregistered and scored as written.

## The description being tested

From the symmetric-DCT lane: the second-order response of an output direction u to a perturbation of the block-8 residual is created by bilinear MLP curvature downstream, fed by attention **values** that transport the perturbation linearly. In weight terms, head h moves θ into position i as s_i^h · M_h θ with M_h = C_h W_v^h a fixed matrix and s_i^h the head's attention mass (a context scalar); an MLP unit (a, b) then contributes (a·Δx)(b·Δx). So the exact form is Σ_units Σ_heads [context scalars] · sym(M_hᵀ a bᵀ M_{h′}): a fixed dictionary of head-transport × unit forms, with the context only in the coefficients. Rung C1a finds the heads and units per direction; C1b fits that dictionary; C2 checks the attribution at finite perturbation size; C3 tests the four properties.

## Rung C1a — attribution for 100 directions (3 of 5 as written)

**Directions.** The 16 root readers and the unembedding rows of the 84 most frequent next tokens in the held-out FineWeb rows. **Contexts.** 16 held-out FineWeb contexts. **Per pair:** probe direction from 30 power iterations on Hessian-vector products (Rayleigh quotient within 11% of the exact top |eigenvalue|; the spectra are near-degenerate, so the probe is a high-curvature direction rather than the unique top one), then fractions of u·H[v, v] removed by freezing, in vmap batches (batched = sequential to 2e-7): each of the 90 heads' values, each block's values, each block's MLP, all MLPs, the top-20/100 units by hidden mixed derivative; everything frozen = 1.000 exactly (no-final-norm span). 1,600 pairs, 4.5 h.

| quantity (medians over 100 directions × 16 contexts) | readers | token directions |
|---|---|---|
| all MLPs frozen | 0.97 | 0.96 |
| all heads' values frozen | 0.79 (all) | |
| top-100 units | 0.38 (all) | |
| Rayleigh quotient (interaction energy, unit perturbation) | 7.7e-6 | 5.1e-6 |

**Transport is shared.** Heads 9.8, 9.7 and 8.2 are in the top-3 value-transporters for 81%, 73% and 63% of the 100 directions (pred_b ✓; next: 10.5 for 27%, 11.6 for 16%). The head-share vectors of 84 directions fall in one cosine cluster at 0.9.

**Curvature has a universal core.** Each direction has a median of 40 "stable" units (in its top-100 in ≥ 8 of 16 contexts). Those sets overlap heavily: mean Jaccard 0.51, one cluster of 97 directions at Jaccard 0.5. 48 units are stable for ≥ 25% of directions and **39 for ≥ 50%**, nearly all in block 17's MLP: units 17.644, 17.1239 and 17.2902 are stable for 100 of 100 directions, 17.137 and 17.3533 for 99. pred_d (≥ 100 units reused by ≥ 25%) fails as written; the finding is the opposite of what the bar assumed — reuse is concentrated, not broad.

**The second curvature block is direction-specific.** Block 8's MLP (the source block's own) is one of the two largest single-block removals for almost every direction; the other is block 9 (44 directions), 10 (20), 17 (17), 16 (7), 11 (5). pred_c (8 and 17 for ≥ 70%) fails: the readers' preference for block 17 does not carry to token directions at the block level, even though the *unit* core is in block 17.

**Token directions behave like readers** (pred_e ✓: all-MLPs 0.96 vs 0.97), with slightly weaker interactions.

So at this order the model has, for essentially any output direction, one transport (three heads) and one curvature hub (a few dozen block-17 units plus block-8 units), and a smaller direction-specific part. Whether the hub is a circuit or a generic amplifier is what selectivity (C3) decides; whether the description is weight-space is what C1b decides; whether it holds at finite scale is C2.

**Receipts.** `results/circuits_c1a.json`, plan `plans/CIRCUITS_PLAN_V1.md`, runner `scripts/run_circuits_c1a.py`.

## Rung C1b — fixed head-transport × unit-form dictionary (3 of 5): the transport is not a fixed matrix

Dictionary: 7 transports (identity + the six most-used heads' value maps M_h = C_h W_v^h) × the 314 stable units from C1a → 15,386 rank-one symmetric forms sym((Tᵀa)(T′ᵀb)ᵀ); exact per-context Hessians for 32 directions (16 readers, 16 token directions) × 8 contexts; ridge least squares. Controls pass: inner products and Gram exact to 1e-15, planted combination R² 0.99995, noise baseline 0.023, and the physics control — one unit's own curvature through the identity transport captured by its own form at **0.92**.

| prediction | bar | result |
|---|---|---|
| b: captured R² | ≥ 0.5 | **✗ 0.074** (0.04–0.21; noise 0.023) |
| c: head transports beat the identity transport | ≥ 1.5× | ✓ 4.3× |
| d: coefficient mass on a head tracks its attention mass across contexts | corr ≥ 0.5 | ✗ 0.39 |
| e: token directions ≈ readers | ≥ 0.8× | ✓ (0.068 vs 0.078) |

Readers and tokens alike put the most coefficient mass on the same block-17 units (17.2902, 17.1811, 17.3533, 17.3343 in every fit). So the *units* are right and the head value maps are the right *kind* of transport (they beat the identity 4×), but a fixed matrix per head is not the transport that reaches those units: the perturbation passes through the bilinear MLPs of blocks 8–16, whose Jacobians are proportional to the clean activations, and through the norms. The transport is a fixed weight tensor contracted with the context — which is what "tensor network with weights" buys us and what a fixed-matrix dictionary cannot express. The next rung fits nothing: it pulls each unit's read pair back through the exact linearised transport of the context and multiplies by the unit's exact downstream weight, so the only freedom is which units are in the set.

**Receipts.** `results/circuits_c1b.json`, plan `plans/CIRCUITS_PLAN_V2.md`, runner `scripts/run_circuits_c1b.py`; 17 min.

# THREE-HOURLY MATHEMATICAL REVIEW — 2026-08-31 07:07Z

Convention (§2135): CE numbers are damage added above the real model; LOWER IS BETTER.

## The fresh mathematical facts this review must explain (not a generic survey)
1. §2209: four single-site replacement drifts compose in QUADRATURE in state space (√Σr² = 0.862 predicts
   the observed 0.833 within 4%) — they are nearly L2-orthogonal — yet CE damage is 3.3× SUPERadditive.
2. §2207: the CE excess is a progressive ladder (in-config marginal multipliers 1.0/2.5/5.8/5.8× singles).
3. §2208/§2210: aggregate CE yields to K (+0.119 at all-3456) and to trajectory-refit (+0.437 at zero extra
   values), but certificate validity is pinned at ≤ 2/62 for every multi-site config ever built.
4. §2211: the real attention is only a partial conduit (31%) — the residual stream carries the drift.

## Top three mathematical moves (ranked; pruned for redundancy with completed work)

### 1. Gauss–Newton / Möbius order-2 composition calculus  → EXECUTED (rungs 118+119, queued)
Object: the CE functional around the real model's outputs; the four front replacement deltas δ_i.
Frame: CE(f+Σδ_i) − CE(f) ≈ Σd_i + ½Σ_{i≠j}δ_iᵀHδ_j (H = Gauss-Newton/Fisher metric on outputs).
L2-orthogonality (§2209) does NOT imply H-orthogonality — the superadditivity must live in the H-cross terms
J_ij, measurable directly: J_ij = agg(pair_ij) − single_i − single_j. The §2207 ladder already fixes the row
sums (J_01 = .0676; J_02+J_12 = .2395; J_03+J_13+J_23 = .3568); rung 118 measures the six J_ij, rung 119 the
four triples (either validating order-2 at an order it wasn't fit on, or measuring the cubic K_ijk exactly).
Assumption that may fail: r ≈ 0.8 drift is far from perturbative — that failure IS the registered null.
Consequence beyond reconstruction: predict-before-run for all 2⁴ front subsets (a composition calculus), and
a principled repair target: kill the dominant J pairs, not the largest marginals.
Falsifier cost: 10 evalVs (~5 min). Queued.

### 2. Fisher-metric orthogonality of the drifts (rank 2, deferred until 118/119 read out)
Object: δ_i at the block-9 output; test δ_iᵀFδ_j directly with the §2124 true-Fisher sampler.
If J_ij from move 1 quantitatively equals ½(δ_iᵀHδ_j + δ_jᵀHδ_i) on a sample, the quadratic story closes; if
not, the nonperturbative regime is proven and repair must be nonlinear. Deferred: partially redundant with
118 until its verdict exists; GPU-heavier.

### 3. Two-distortion rate ledger (Pareto accounting; CPU, folded into this review)
Object: every config receipt; distortions = (aggregate dCE, 62 − valid); rate = stored values.
Current front Pareto set (aggregate ledger): table front 231.6M/+1.747 → mixed CP 21.9M/+0.943 →
seq_traj 21.9M/+0.437 (dominates all-2304's 31.9M/+0.489 — a strict Pareto win) → all-3456 47.8M/+0.119.
Certificate ledger: 0/0/1/1/2 of 62 across those same points — the two distortions are nearly DECOUPLED,
which is itself the strongest argument that certificates measure directions, not magnitudes (§2209).
Consequence: operating points can now be chosen from the table; registry rows stay aggregate-only (synced).

## Pruned this cycle
Hankel/automata minimal realizations (no sequential-state object at the front MLPs); tensor-rank lower bounds
(the CP grammar already uses the model's own units — rank questions became K questions, answered empirically
§2203–§2205); invariant-theory gauge quotients (still gated on the rung-90 document-grain instrument);
bisimulation metrics (the certificate battery already operationalizes the abstraction relation); information
bottleneck (no new handle on the drift-direction problem).

## Executed
Rung 118 (pairwise J_ij) and rung 119 (triples / Möbius completion) preregistered, built, dryrun-clean,
queued behind the runner. Their joint outcome decides whether the program gains a predict-before-run
composition calculus or a measured cubic lattice — either is a structural asset the ledger lacks today.

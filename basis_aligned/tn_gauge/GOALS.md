# TN-gauge / code-propagation program — goals, ladder, state

**GOAL.** Reduce a bilinear model to a *faithful, legible tensor network of
mechanisms in a shared overcomplete code basis* — Logan's construction: one learned
dictionary Φ (d×m, m>d) per residual stream; every layer expressed as a small
quadratic form over sparse codes; codes manufactured by propagation from the token
boundary; the sparse solver demoted to a faithfulness *auditor*. The model stays one
fixed tensor network; each input runs a small legible sub-network of it. Verified by
end-to-end ΔCE (binding), staying a tensor network throughout (no nonlinear SAEs).

**Method.** Toys first for fast iteration (`runs_lm/block1..block4`, TinyStories,
d=128); port the winning reduction to bilin18 (546M) only once it clears the gates.

## The construction (Logan 2026-07-20), as testable pieces

Layer `y = D(Lx ⊙ Rx)`, bond dictionary Φ (d×m), code `x ≈ Φc`, support `s=|c|₀≪d`.
- Step 1 code: c from (a) sparse solve, (b) amortized encoder, or **(c) inherited
  from upstream** (the target regime — no per-input inference).
- Step 2 layer in code coords: `y = Σ_{j≤k∈s} (2−δ) c_j c_k u_jk`, `u_jk=½D(ℓ_j⊙r_k+ℓ_k⊙r_j)`,
  `ℓ_j=LΦ_j, r_k=RΦ_k`. Precomputed; forward = lookup+combine over the live pairs.
- Step 3 two sparsities multiply: input sparsity |s| × core sparsity (few `u_jk` matter).
- Step 4 closure/propagation: if each `u_jk ≈ Φ w_jk` (sparse), then
  `c_out = c_in + Σ (2−δ) c_j c_k w_jk`, truncate to stop support growth. Codes flow
  by arithmetic; solver only audits `‖x_ℓ − Φc_ℓ‖`.
- Step 5 cost of superposition: (a) code well-definedness = local Gram conditioning
  (near-parallel active atoms → credit split is gauge, report don't interpret);
  (b) residual `ε=x−Φc` amplifies `y=[code]+2T(Φc,ε)+T(ε,ε)` → ~2× rel-error per
  bilinear layer → deep claims degrade with distance from boundaries (LIB mechanism).

Learned object: **Φ only**. Derived (contractions of Φ with frozen weights):
`ℓ_j,r_k,u_jk,w_jk`. Only non-tensor object: the discrete support field `s(x)`.

## LADDER REORDERED (Logan 2026-07-20) — rotation first, activations demoted to audit
Gate 1/2 (dictionary on activations) is decoder-only dictionary learning — a legal way
to *search* for Φ and a useful **representability upper bound** (the stream is ~93%
representable in 512 shared atoms), but it is NOT the construction: it skipped the
zero-CE baseline that gives its numbers a denominator, lives at tier-2/3
(distribution-dependent) when the program wants tier-1 weight-derived claims, and a Φ
trained on every bond's activations *silently absorbs manufactured features as ordinary
atoms*. Reordered per Logan:
1. **Regime 1 — exact rotation sweep** (weight-only, ΔCE=0): per-bond core-L4/L1 gauge,
   ends pinned to E/W_U. Report per-bond core sparsity + **floor** (= superposition
   carried) → the zero-CE baseline AND stage 1 of Φ (its first d columns).
2. **Per-bond atom budgets from the floors** → births from weight-side write directions,
   nested over the rotation basis, **dedup/orthogonalized against the core** (DL depends
   on it). Regime 2 = increment on regime 1, not a parallel arm.
3. **Code propagation**, activation refits demoted to verification gates.
4. **Rerun write-span vs the ROTATION-STAGE dictionary** (boundary-derived), where
   manufactured features are genuinely absent and can show up — the activation-trained
   run (F4/flagship) is confounded and is NOT an atom-birth verdict.

## Experiment ladder (gate each before the next)

- **[0 DONE] Gauge primitives** (`toy_gauge_probe.py`). Residual bond is pinned by the
  embedding/unembedding boundaries (no interior sweep); real freedoms are per-layer
  private: OV = full O(d_head); QK = RoPE-constrained (input-anchored); MLP hidden =
  pinned by ⊙ (perm+scale only). Weight-only cross-layer DAG is uniform → need
  data-contrastive scoring. → `PLAN.md`.
- **[1 DONE] Shared-Φ code-propagation gate** (`toy_code_propagation.py`). One Φ
  (m=512) coding EVERY bond. **NEGATIVE at this size**: end-to-end ΔCE +0.59 even at
  k=64 (FVU ~0.03–0.08); faithful code-coordinates need far higher fidelity than a
  small shared dictionary gives cheaply. FVU rises with depth (0.07→0.23 at k=16);
  amplification 1.0×(shallow MLP)→1.4×(deep MLP), below the 2× bound but
  depth-increasing (Step-5 mechanism supported, magnitude looser). → findings below.
- **[2 DONE] Fidelity/bits floor** (`toy_fidelity_floor.py`). Both capacity and
  per-bond dictionaries cut ΔCE ~6× (shared-512 +1.17 → per-bond-2048 +0.19). F2 was
  underpowered, not fatal. **But surfaces a real tension (F3):** additive propagation
  needs ONE shared Φ, yet shared Φ is the lossy config; per-bond is faithful but
  requires re-encoding at each bond (= regime (a)/(b), not the free-propagation (c)).
  Follow-up (gate 2b, running): does scaling a *shared* Φ (m→8192) reach ΔCE<0.05, or
  plateau?
- **[3] QK-measure propagation** (Logan's direct ask, not yet done). Reduce layer-0
  QK to an alphabet; propagate the attention-weighted co-occurrence measure
  Pr[s attended from t] to weight layer-1's pair domain; show the effective pair
  count ≪ V² and ≪ k². Tests "alphabets forward, cores at bonds, monomials never."
- **[4] Closure test** (Step 4). Are the live `u_jk` sparse in Φ? If not, codes
  can't propagate without re-solving → regime (c) fails, fall back to (a)/(b).
- **[5] Core sparsity** (Step 3). Fraction of live pairs carrying the output norm.
- **[6] OV L1-gauge sparsity floor vs depth** on block4 (8 layers): does the
  achievable OV rotation sparsity worsen with distance from the boundary (LIB)?
- **[7] bilin18 port** of whatever clears gates 2–6.

## Findings so far

### F1 — the residual bond is pinned; freedoms are per-layer private (tick, PLAN.md)
Verified exactly on block2. Reframes the DMRG-sweep plan into independent per-layer
gauges; no deep-layer SAE; stays a tensor network.

### F2 — naive shared-dictionary code coordinates are too lossy end-to-end
`toy_code_propagation.py`, block2, real TinyStories, baseline CE 1.729.
- **G1** shared Φ (m=512), FVU per bond (LS-refit coeffs, monotone in k):
  k=16 → 0.067/0.104/0.142/0.229 (bonds 0–3); depth-increasing.
- **G2** every bond coded, ΔCE: +2.71/+2.05/+1.52/+1.12/+0.59 at k=4/8/16/32/64.
  Even the richest code costs +0.59 nats. **The propagation regime does not cheaply
  preserve the model with a small shared dictionary** — sets up gate 2 (how big must
  Φ be, shared vs per-bond).
- **G3** MLP error amplification: 1.0× (bond1, shallow) → 1.4× (bond3, deep). Below
  the 2× worst-case bound (input error not aligned with the amplifying directions),
  but depth-increasing — the Step-5 mechanism holds directionally.
- Figure: `fig_code_propagation.png`.

### F13 — layer-1 QK source-interaction graph is sparse, dominated by bilinear-output self-interaction
Logan's steer (2026-07-20): focus layers 0–1; optimize the second attention's (attn2)
query/key to depend *sparsely* on its upstream sources — embedding E, attn0 output A (OV),
mlp1 bilinear output M — with good cross-entropy; one bond, so stronger methods allowed.
`toy_qk1_interactions.py`. The QK score is bilinear in the residual x2 = E + A + M
(E=0.5·x0, A=x1−0.5·x0, M=x2−x1), so it splits EXACTLY (gate: sum of 9 blocks = real score
to 3e-4; full ΔCE = 0) into a 3×3 source-interaction graph. Frobenius mass and causal ΔCE:

| block | Frob mass | keep-only ΔCE | | cumulative (by mass) | ΔCE |
|---|---|---|---|---|---|
| **M×M** | **0.70** | **+0.062** | | M×M | +0.062 |
| M×E | 0.10 | +1.82 | | +M×E | +0.025 |
| E×M | 0.07 | +1.83 | | +E×M | +0.008 |
| M×A,A×M | 0.09 | +1.84 | | +M×A,+A×M | +0.001 |
| E×E | 0.01 | +1.84 | | +E×E (6 blocks) | +0.0001 |
| A×E,E×A,A×A | 0.02 | +1.84 | | (3 dropped) | — |

**Layer-1 selection runs almost entirely on the bilinear output interacting with itself**
(M×M alone is usable, +0.062), weakly modulated by the embedding (M×E, E×M); the first
attention's output A is not directly read (its 3 pure blocks are negligible; 6 of 9 blocks
recover the model to +0.0001). A sparse, interpretable source graph — the coarsest (source-
level) version of Logan's ask. **Next:** decompose M (and E) into atoms and sparsify the fine
M×M / M×E atom-interaction graph; then the MDL of that vs the regime-1 baseline.

**Regime-1 MDL baseline (Logan asked to bank it):** rotation gauges give ~0 sparsity on every
bond (F6–F8: toy OV 7%, QK 1.4%, flagship ~0%), so the regime-1 description length ≈ the RAW
weight bits — an exact-gauge reparameterization buys essentially no compression. The layer-1
QK raw structural cost is 4 branch-matrices × 128×128 × 32 bits ≈ 2.1 Mbit; regime 1 does not
reduce it. Future interaction-sparsified / atom-decomposed QK is compared against this.

### F12 — closes F11: de-clustering rescues convergence, but write-info is useless for a trained dict
`toy_births_ortho_init_test.py`. F11 asked: was write-init's failure clustering (fixable by
orthogonalization) or write-info-uselessness? Answer: **both, cleanly separated.** random ΔCE
+0.349 (loss@50 0.108); clustered-write +0.497 (loss@50 0.817); **ortho-write +0.351 (loss@50
0.102)**. De-clustering (write-subspace PCA, diverse) restores convergence to the random level
— so clustering WAS the F11 handicap — **but confers no advantage over random** (+0.351 ≈
+0.349). Training finds the write subspace unaided; a weight-informed init offers nothing.

**REGIME-2 SEEDING ARC (F9–F12) — DEFINITIVE BOTTOM LINE:** write directions capture the right
*subspace* (F9 reconstruction, F10 ΔCE ordering, both with fixed atoms) but are useless for
the *trained overcomplete dictionary* that is the actual faithful code: as atoms they're
clustered/rank-limited (F11), and even de-clustered they match but don't beat random (F12). So
weight-informed births are for interpretation and fixed/analytical constructions, NOT for
building the trained code. The practical regime-2 faithful code is a **trained dictionary with
any diverse init**, ΔCE ≈ +0.35 at m=512/k=32 (matching gate-2's trained numbers). This is a
real cost, not a free reduction — a Pareto point (bits saved vs +0.35 nats), not a clean win.

### F11 — REVERSAL: write-seeding is a good fixed dictionary but a BAD training init
`toy_births_init_test.py`. F10 said seeds must be trained, so: does write-seeded init train
better than random init? **No — it trains worse.** Per bond (1–3), overcomplete dict m=512
k=32: write-init loss@50 = 0.817 → final 0.048, end-to-end ΔCE **+0.50**; random-init
loss@50 = 0.108 → final 0.030, ΔCE **+0.35**. Random init is faster (8× lower loss at step 50)
AND reaches a better optimum. Cause: write directions are **clustered and rank-limited** (the
writes concentrate in few directions), so seed atoms sampled from them are redundant — poor
coverage — while random atoms spread across the space and optimize to a better dictionary.

**Synthesis of the regime-2 seeding arc (F9–F11):** write directions identify the right
*subspace* (fixed-dictionary reconstruction F9, and the ΔCE ordering F10), but they are a poor
*overcomplete atom set* — clustered, rank-≤d, and a bad training init (F11). So Logan's
"birth from write directions" works for a fixed/analytical construction and for
subspace-identification, but overcompleteness needs **diversity**, not write-seeding; a
trained dictionary with diverse (random) init is the practical winner (+0.35, matching gate-2).
Note: orthogonalizing the write seeds (Logan's dedup for clean description-length) collapses
them to a ≤d basis — removing the overcompleteness — which is the same tension from the other
side. Overcompleteness and write-informedness are, on this toy, in mild opposition.

### F10 — regime 2 at the BINDING metric: ordering survives, but seeds are an init not a solution
`toy_births_dce_test.py`. F9 was reconstruction (FVU); the binding rule (reconstruction ≠
behavior) demands ΔCE. Fixed seeded dictionaries, bond 0 exact (Logan calibration b), bonds
1–3 coded, m=512, k=32, end-to-end ΔCE (3 seeds): **write +2.81 < token +2.90 < random +3.47**.
The write > token > random ordering **survives at the binding metric** — write clearly beats
random (0.66), marginally beats token (~0.09, ~1.5 std, attenuated from F9's reconstruction
gap). **But absolute ΔCE is catastrophic** (+2.8 on a 1.73 baseline): fixed unoptimized seeds
destroy the model (trained dictionaries gave +0.19–0.52 in gate 2). So write-seeding sets the
right *direction* but seed atoms are an **initialization, not a solution** — they need
training. Next (chained): write-seeded init + training vs random init — does the good
direction give faster/better convergence?

### F9 — regime 2 (first step): the UN-CONFOUNDED births test supports weight-informed births
`toy_births_seed_test.py`. Logan's step-4 fix: don't TRAIN Φ on activations (which silently
absorbs manufactured features, confounding F4); instead SEED atoms from weights and leave
them fixed, then compare seedings. Per bond, fixed dictionary of m=512 atoms, sparse code
k=16 (corr top-k + least-squares refit), FVU over 5 subsamples. Deep-bond mean FVU:

| seeding | deep-bond mean FVU |
|---|---|
| WRITE (upstream write deltas) | **0.389** |
| TOKEN (boundary/vocab dictionary) | 0.439 |
| RANDOM (unit vectors) | 0.518 |

**Write-seeded atoms reliably beat token and random** (std ~0.005), and the advantage
**grows with depth** (bond 2: write 0.352 vs token 0.502; bond 0: token wins, as expected
near the embedding). So with weight-derived seeds — un-confounded, atoms never trained —
write directions capture the deep stream better than the boundary dictionary, increasingly
with depth: the atom-birth signal the confounded F4/flagship write-span run couldn't show.
Weight-informed births are supported. **Flagship confirms with LARGER gaps**
(`bilin18_births_seed_test.py`, bonds 3/6/10/17): WRITE **0.692** < TOKEN 0.850 < RANDOM
0.918 — on the real low-rank stream (F5) write directions are a decisively better seed than
the boundary dictionary. Caveats: fixed unoptimized seeds (measures seeding quality, not
trained ceiling; FVU high); next = nest the births over the rotation basis with
orthogonalization for clean description-length, and measure the sparsity/ΔCE they buy.
**Gate note:** a single-sample RANDOM draw was anomalously good (0.16) and misleading; the
5-subsample check corrected it to 0.519±0.003 before any claim (falsifiable-verification rule).

### F8 — regime 1 COMPLETE: the exact rotation baseline is nearly empty (step-back)
`toy_qk_torus_floor.py` finishes the query/key bond. A query/key rotation is a gauge
only if it commutes with RoPE; for rotate-half RoPE the commuting subgroup is a
**16-angle torus** per head/branch (one 2D rotation per frequency plane), far smaller
than OV's full O(d_head). Optimized (L4 ascent, gated by a planted-torus control that
recovers the known optimum), the query/key floor is **1.36%** L1 drop (exact gauge,
ΔCE −1e-7) — even lower than OV, as the 16 angles predict.

**Regime-1 summary (fig_regime1.png), all gauges verified ΔCE≈0:**
| bond | gauge | L1 sparsity gained |
|---|---|---|
| toy OV | full O(32) | 7.0% |
| toy QK | 16-angle RoPE torus | 1.4% |
| flagship OV | shared across depth (value bus) | ~0% |
| flagship QK | 64-angle RoPE torus | 0.22% |

**The square-rotation baseline is nearly empty.** No private bond gives up much sparsity
to an exact orthonormal change of basis; on the flagship the shared value bus makes it
~0%. So the entire sparsity budget of the construction must come from **overcompleteness**
(regime-2 births) — regime 1's real deliverable is (a) the zero-CE anchor that gives the
overcomplete arm's cross-entropy a denominator, and (b) the finding that rotation alone
cannot compress these bonds, which is *why* overcompleteness is required, not optional.
Two architectural facts surfaced along the way (both caught by gates): the residual bus
and the value bus are each shared (embedding-pinned; value-residual mixing), so their
"rotation" freedom is null and their sparsity is entirely a births question.

### F7 — flagship regime 1: the value bus is shared across depth AND rotation-incompressible
`bilin18_regime1.py`. Two findings, both caught/verified by the ΔCE gauge check:
1. **The naive per-layer OV rotation is NOT a gauge on bilin18** — max|Δlogit| = 16.8.
   Cause: bilin18 mixes every layer's value with block-0's value (`v=(1-lamb)v+lamb·v1`,
   tier2_model L87-89), so the value head-subspace is **shared across all 18 layers**.
   The gate caught the wrong per-layer assumption (like the residual bus, the value bus
   is shared — now for a concrete architectural reason, the value-residual).
2. **The correct shared-per-head gauge is exact** (max|Δlogit| = 5e-4) but rotation buys
   **~0%** — per-head L1 drop 0.01–0.06%, Hoyer flat 0.22→0.22, per-layer drops all ≈0.
   One 128-dim rotation can't sparsify 18 layers' maps jointly, so the OV value subspace
   is **fully rotation-incompressible on the flagship** (floor ≈ 100%), versus 7% on the
   toy. Consequence: on bilin18 **all** OV sparsity must come from overcompleteness
   (regime-2 births) — the square-rotation baseline is empty there. Sharpens why the
   overcomplete arm is necessary, not optional, on the real model.

### F6 — regime 1: OV cores are ~7% rotation-sparsifiable (the zero-CE floor)
`toy_regime1_rotation.py`. Per attention head, the exact value/output gauge
Q∈O(d_head) maximizing ||o Q||₄⁴+||Qᵀ v||₄⁴ (L4/kurtosis rotation-to-sparsity).
Applied to all heads it is an **exact gauge — ΔCE = −2e-6 ≈ 0**. L1 of the OV maps
drops only **5.8–7.8%** (Hoyer 0.20→0.26); deeper layer L2 slightly more than L0. So
the OV bonds are largely **rotation-incompressible** — a square orthonormal basis
can't concentrate them; ~93% of the L1 survives. This ~7% is the honest zero-CE
baseline and the per-bond superposition floor; the rest of the sparsity must come from
overcompleteness (regime-2 births), which is exactly why the overcomplete arm exists.
**Bug caught by control** (positive-controls lesson): the first optimizer (L1-subgradient
Cayley) was dead (0.3% on a planted-sparse control that should recover ~78%); switching
to L4 ascent passed plant (78% recovered) and random (invented none) controls. The 0.3%
"floor" from the broken optimizer was discarded.

DEVIATION FLAGGED (QUESTION FOR LOGAN): the sweep runs on per-layer PRIVATE bonds
(OV free, QK RoPE-constrained, MLP-hidden pinned), not the residual bonds Logan's
objective indexes — because pinning *both* ends of a shared residual bus (embed rank=d)
pins the whole interior (gate-0 A/B), so interior residual Q_ℓ = I under end-pinning.
The residual bond's Φ therefore gets sparsity from births, not rotation. Also: private
bonds are independent, so regime 1 is parallel/one-shot, not an iterative sweep — the
DMRG cross-bond coupling enters only in regime 2. Confirm this reinterpretation.

### F5 — the flagship HAS the low-rank stream the toy lacked (premise holds)
`bilin18_actrank.py`, residual-stream effective rank vs depth (d=1152, 4096 tokens):
eff-rank ~530–650; **rank@90%-variance ~150–260 of 1152 (13–22% of width)** — genuinely
low-rank, versus the toy's near-full-rank d=128. rank@90% dips in the middle (bond 6:
151, the min) and rises toward both ends → the mid-network stream is most compressible.
So the toy's isotropic-residual verdict (F4) was a size artifact; the sparse-code regime's
premise is real on bilin18, and the atom-birth question is live there. → gate 2b on
bilin18 running (`bilin18_writespan.py`, dictionary on middle bonds, write-span capture).

### F4 — gate 2b: on the toy the coding residual is ISOTROPIC, not write-structured
Logan's refinement: additivity forces *compatibility* not identity → nested growing
dictionary Φ_{ℓ+1}⊇Φ_ℓ (shared core + per-bond atom **births**); births are necessary
because layers manufacture features with no token-boundary preimage; the depth-degrading
FVU should then be the closure assumption (writes sparse in the *existing* dictionary)
failing. Diagnostic (`gate2b_writespan.py`): project each bond's coding residual onto the
upstream **write-mechanism** span vs token span vs random; ceiling = the residual's own
top-K subspace. Result (shared Φ m=2048, k=32, K=32 subspaces, fraction of residual
variance captured):

| bond | write-span | token-span | random | self-ceiling | resid eff-rank | act eff-rank |
|---|---|---|---|---|---|---|
| 0 (attn) | n/a | 0.248 | 0.250 | 0.433 | 123.8 | 118.4 |
| 1 (mlp)  | 0.299 | 0.229 | 0.250 | 0.421 | 124.9 | 120.3 |
| 2 (attn) | 0.270 | 0.248 | 0.250 | 0.353 | 126.6 | 109.7 |
| 3 (mlp)  | 0.255 | 0.241 | 0.250 | 0.374 | 126.2 | 115.8 |

**Atom-birth is REFUTED on this toy.** The write span captures the residual no better
than a random subspace (~0.25) and far below the residual's own best 32-dim (0.35–0.43);
the residual is nearly isotropic (effective rank ~125/128). There is no low-dimensional
missing-feature subspace to birth atoms from. Per-upstream-layer capture is also ~random
(0.23–0.30). Calibration (b) bond-0-exact barely helps (+0.478→+0.464): the cost is
distributed, not at the boundary.

**But the verdict is scope-limited, NOT "regime is the limit."** The cause is that the
d=128 activations are themselves near-full-rank (act eff-rank ~110–120 of 128), so a
sparse dictionary leaves isotropic quantization noise — the toy is simply too small to
have the low-rank residual stream the whole sparse-code-propagation regime assumes. The
premise must be tested where width is large. → gate 2b' (running): activation effective
rank vs depth on bilin18 (d=1152). If the flagship stream is low-rank, the regime (and
the atom-birth question) is live there; if it too climbs toward full rank, the regime is
in real trouble.

Shared-Φ scaling (record, non-decisive per above): shared m=2048/4096 k=64 →
ΔCE +0.247/+0.162 — capacity helps, slowly and at rising bit-cost.

### F3 — the propagation/fidelity tension (the load-bearing finding of gate 2)
`toy_fidelity_floor.py`, end-to-end ΔCE (baseline 1.729), k=32:

| dictionary | ΔCE | bits |
|---|---|---|
| shared m=512 | +1.17 | 21 Mbit |
| per-bond m=512 | +0.58 | 27 Mbit |
| shared m=2048 | +0.52 | 31 Mbit |
| per-bond m=2048 | **+0.19** | 57 Mbit |

Capacity helps and per-bond helps — so the regime is *viable*, F2 was underpowered.
The tension: Logan's Step-4 additive propagation (`c_out = c_in + Σ c_j c_k w_jk`,
codes flow with no per-input solve) requires the SAME Φ at every bond, because
`x_{l+1}=x_l+write` only maps to code addition if writer and reader share Φ. But
shared Φ is precisely the lossy column (+1.17 / +0.52). Buying fidelity with per-bond
dictionaries (+0.19) forces a re-encode at each bond — that is regime (a)/(b) (solve
or amortized encoder), **not** the free-propagation regime (c) the construction aims
for. So the cheap-propagation regime and the faithful regime are, on this toy, in
opposition. Gate 2b asks whether enough *shared* capacity closes the gap.

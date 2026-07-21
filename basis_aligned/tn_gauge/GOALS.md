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

### F25 — HELD-OUT frontier confirms F24 is real (not overfitting); used-subspace dominates
`bilin18_used_frontier.py` + `fig_used_frontier.png`. F24's used-subspace was fit on the same
tokens ΔCE was measured on (in-sample — the positive-controls trap). Here: fit on TRAIN token
windows, measure ΔCE on DISJOINT HELD-OUT windows, full r-frontier (baseline CE on test 3.68).

| | r=16 | r=64 | r=128 | r=256 |
|---|---|---|---|---|
| **L1 used (held-out)** | +0.022 | +0.007 | +0.002 | −0.0006 |
| L1 generic low-rank | +0.97 | +0.13 | +0.076 | +0.035 |
| **L9 used (held-out)** | +0.011 | +0.007 | +0.005 | +0.002 |
| L9 generic low-rank | +0.034 | +0.018 | +0.016 | +0.007 |

**The used-subspace still dominates generic low-rank out-of-sample at every rank and both layers**
(single-source L1 and distributed L9), and it's cheaper (5rD vs 8rD). The in-sample→held-out gap
is tiny (L1 r=128: −0.0006 → +0.002), so **F24 was not overfitting** — it is a real, general,
held-out-validated activation-aware compression: query/key maps compress to ~14% of raw bits
(r=128) for +0.002–0.005 held-out at every depth, or ~7% (r=64) for +0.007. The used-subspace
frontier strictly dominates the generic-low-rank frontier (F21) on both axes, held-out. This is
the banked, validated general result. Next: interpretability of the used directions; and whether
the same activation-aware trick helps the value/output side.

### F24 — the GENERAL method: activation-aware used-subspace QK compression beats the frontier at ALL depths
`bilin18_used_subspace.py`. F22 (M-subspace) was a layer-1 proxy for the true "used subspace";
F23 showed the single-source projection doesn't generalize. The general version: the optimal
rank-r INPUT projection preserving the query/key reads over data, min_P E‖R(I−P)x‖² with
R=[Wq;Wk;Wq2;Wk2] — whitened solution W = top-r eigvecs of C^{1/2}·R^T R·C^{1/2} (C=Cov(x)),
P = C^{1/2}·W·Wᵀ·C^{−1/2}; W′=W·P factors as shared basis = 5rD bits. ΔCE at r=128 (14% raw):

| layer | **USED** | generic low-rank | source (F23) | input-PCA |
|---|---|---|---|---|
| 1 | −0.0006 | +0.060 | −0.001 | +0.026 |
| 3 | −0.0004 | +0.016 | +0.095 | +0.331 |
| 6 | −0.005 | +0.012 | +0.028 | +0.375 |
| 9 | −0.0003 | +0.027 | +0.576 | +0.088 |
| 12 | −0.009 | +0.004 | −0.002 | +0.021 |

**The used-subspace compresses every layer's query/key to ~14% of raw bits for free — even
improving CE — beating generic low-rank, the single-source projection, and input-PCA at all 5
depths (5/5).** It's cheaper too (5rD = 14% vs generic 8rD = 22%), so it dominates on both axes.
**This is the general method the arc was reaching for**: the query/key reads a ~128-dim
*activation-weighted* input subspace at every depth; identifying it (data-driven, whitened-
optimal) — not the raw weight SVD (generic low-rank) nor the input variance (input-PCA) nor a
single source (F22/F23) — is what unlocks the compression. F22's interpretive win was a
layer-1 shadow of this. Bug caught by the layer-1 sanity check (whitening inversion → the
used-subspace must beat the source at layer 1, not be catastrophic; C^{1/2} vs C^{−1/2} swapped).
**Beats the F21 banked frontier decisively and generally.** Next: full r-frontier of the used-
subspace; and whether the used-subspace is interpretable (what those ~128 directions are).

### F23 — F22 is LAYER-1-SPECIFIC: interpretive-subspace compression needs a single-source circuit
`bilin18_msub_depth.py`. Does F22 generalize into a method? Project each attention layer L's
query/key onto block(L-1)'s mlp-output subspace (top-128), ΔCE vs generic low-rank r=128:

| layer | M-subspace ΔCE | generic low-rank ΔCE |
|---|---|---|
| 1 | −0.001 | +0.060 |
| 2 | +0.007 | +0.003 |
| 3 | +0.095 | +0.016 |
| 6 | +0.028 | +0.012 |
| 9 | +0.576 | +0.027 |
| 12 | −0.002 | +0.004 |

The interpretive-subspace advantage holds at only **2/6 layers** (1, and marginally 12) — it is
**layer-1-specific**. For deep layers the preceding-mlp subspace is the wrong basis (layer 9:
+0.576), because deep selection is distributed (F19) and doesn't read a single preceding source.
So F22 is **not** a general "project onto the preceding bilinear output" method. The durable
lesson survives, scoped: *when a circuit reads a single interpretable source, projecting onto that
source's subspace beats structure-blind low-rank* — but that clean single-source condition only
holds at layer 1; deeper query/key reads a broader distributed subspace not cheaply identified.
The interpretability→MDL link is real but conditional on a clean interpretation. Bounds F22 the
way F19 bounds F18: the strong result is an early-layer phenomenon.

### F22 — the interpretive structure BEATS the frontier: M-subspace QK compression (gated)
`bilin18_qk1_msubspace.py`. Use F18's finding (layer-1 reads the bilinear output M) to compress:
project the 4 query/key read maps onto the top-r principal directions of M's activations — a
SHARED basis (U_M, D×r) + 4 read factors (r×D) = 5rD floats. ΔCE vs bits, gated by a residual-PCA
control (project onto the QK-input's own variance directions instead of M's):

| r | M-subspace ΔCE | residual-PCA (control) | Mbit | %raw |
|---|---|---|---|---|
| 32 | +0.016 | +0.498 | 5.9 | 3.5% |
| 64 | +0.010 | +0.260 | 11.8 | 6.9% |
| 128 | **−0.001** | +0.139 | 23.6 | 13.9% |
| 256 | −0.003 | +0.055 | 47.2 | 27.8% |

**The interpretive structure buys large, gated MDL.** M-subspace compresses layer-1 query/key to
**~7% of raw near-free** (r=64, +0.010) or **~14% while IMPROVING CE** (r=128, −0.001), vs the F21
generic frontier's ~40% for +0.009 — a ~6× bit improvement at matched ΔCE. **Gated as M-specific**:
the residual-PCA control (the input's own top-variance directions) is ~26× worse at matched bits
(r=64: +0.26 vs +0.010), so the win is *not* generic input-low-rank — it is the bilinear-output
subspace specifically. The residual's high-variance directions are ones the query/key ignores;
M's subspace is what it uses (F18). **This is the payoff of the whole arc**: the interpretive
finding (layer-1 selects on the bilinear output) is a concrete, falsifiable, large MDL reduction
that generic methods cannot reach. `fig_qk1_mdl.png` updated with the M-subspace curve.

### F21 — layer-1 QK MDL frontier (the banked baseline Logan asked for)
`bilin18_qk1_mdl_frontier.py` + `fig_qk1_mdl.png`. Two weight-compression methods on bilin18
h[1]'s query/key (raw 169.9 Mbit = 4×1152²×32), matched-bits, ΔCE binding, index bits shown
side by side with values (MDL convention). Gate: full = ΔCE 0.

| method | ΔCE | Mbit | %raw |
|---|---|---|---|
| low-rank r=64 | +0.13 | 18.9 | 11% |
| low-rank r=128 | +0.06 | 37.8 | 22% |
| low-rank r=256 | +0.03 | 75.5 | 44% |
| prune keep-50% | −0.003 | 138.9 | 82% |
| prune keep-25% | +0.009 | 69.5 (42.5 val + 27.0 idx) | 41% |
| prune keep-12.5% | +0.055 | 34.7 | 20% |
| prune keep-6.25% | +0.24 | 17.4 | 10% |

**Layer-1 QK compresses to ~40% of raw for near-free (+0.009, prune keep-25%) or ~22% for
+0.06 (low-rank r=128).** The two methods cross near 20% of raw: **low-rank wins at low budgets**
(r=64 +0.13 vs prune +0.24 at ~10%), **pruning wins at high budgets** (keep-25% +0.009 vs
low-rank +0.03 at ~42%). Pruning's index overhead is real (~40% on top of values). **Regime-1
rotation = 0 compression (raw)** — the frontier is what any future method (learned sparse basis,
source-structured code, etc.) must beat. keep-50% even improves CE (−0.003): half the QK weights
are removable-with-benefit. This is the banked layer-1-QK MDL baseline for future comparison.

### F20 — what layer-1 selection DOES: predominantly long-range content-based, a few local heads
`bilin18_layer1_pattern.py` (forward = reference exactly). Characterize h[1]'s attention pattern
(unnormalized bilinear; use |pat| normalized per query as read-weight). Read-weight by relative
offset, mean over heads: **local (offset ≤2) 0.23, long-range (offset >8) 0.62**. Per head:
most (0/2/4/6/8) are long-range (65–84% of weight beyond offset 8); head 1 is local (0.63 within
offset ≤2); heads 3/5 are previous-token-ish (peak at offset 1 ~44% of queries). So the special
layer-1 selection — which runs on the bilinear output (F13/F18) — implements a **predominantly
long-range, content-based read**, not a positional/induction rule, with a minority of local heads.
Consistent with reading M richly/high-dimensionally (F14–F16): the selection is content-based and
global. A fraction-of-weight sanity check (>1 impossible) caught a normalization bug before the
wrong "local" verdict was reported.

### F19 — the bilinear-output selection is LAYER-1-SPECIFIC; deep-layer selection is distributed
`bilin18_depth_sources.py` (bounded depth-generalization of F18, forward = reference exactly).
For block L, ablate block L-1's mlp vs attn write from L's QK input, ΔCE:

| block | remove preceding MLP | remove preceding attn |
|---|---|---|
| 1 | **+0.676** | −0.0002 |
| 2 | +0.066 | +0.004 |
| 3 | +0.027 | −0.003 |
| 6 | +0.016 | +0.060 |
| 9 | +0.001 | +0.003 |
| 12 | +0.004 | +0.001 |
| 17 | +0.005 | +0.003 |

F18's "layer-1 selects on the preceding bilinear output" is **strongly layer-1-specific** and
decays fast with depth (layer 1 +0.68 → layer 3 +0.027). **Deep-layer selection is distributed**
— blocks 9/12/17 barely depend on any single preceding write (+0.001–0.005), reading the
accumulated residual robustly (block 6 is a minor exception where the preceding *attention*
matters more, +0.060). So the strong, sparse, interpretable source structure is an EARLY-layer
phenomenon; by mid/deep layers the query/key reads a distributed residual with no single critical
source. Honestly bounds F18: the finding is real but layer-1-scoped, not a universal pattern.

### F18 — FLAGSHIP CONFIRMS F13's interpretive finding: layer-1 selection runs on the bilinear output
`bilin18_qk1_sources.py`. Causal source ablation on bilin18's h[1] query/key (per-head QK
RMSNorm makes the exact bilinear block split non-transferable, so ablate instead). Decompose
h[1]'s QK input xin1 = E + A + M (E=embedding path, A=block-0 attn output, M=block-0 bilinear
output), remove each from the QK input only, ΔCE. Gates: E+A+M=xin1 (1.5e-5); inline forward =
reference_forward EXACTLY (Δ=0). Result:

| remove | source | ΔCE |
|---|---|---|
| M | block-0 **bilinear (mlp) output** | **+0.676** (essential) |
| A | block-0 attention output | −0.0002 (droppable) |
| E | embedding path | −0.011 (slightly helpful to drop) |

**Layer-1 selection on the flagship runs almost entirely on the bilinear output** — removing it
costs +0.68 nats, while the attention output and embedding are droppable. **F13's interpretive
finding GENERALIZES to the real model** (contrast F16's compression story, a toy artifact per
F17). So the durable, model-general result of this arc is interpretive: *layer-1 query/key
selects on what the bilinear layer computed, not on raw tokens or the attention output.*
**Gate discipline:** the reference-CE gate caught a broken inline forward TWICE (omitted value-
bus mixing, then omitted embedding-RMSNorm) before any number was reported.

### F17 — FLAGSHIP OVERTURNS F16: bilin18's layer-1 QK is already sparse in the standard basis
`bilin18_qk1_learned_basis.py`. The binding-metric generalization of F16 to bilin18's second
attention (h[1]). Control passes (planted 86%). But the learned input-basis rotation barely
sparsifies the reads (**1.3%** L1, vs toy 24.7%), and it does NOT help pruning — the ORIGINAL
basis prunes better:

| keep | original ΔCE | learned ΔCE |
|---|---|---|
| 50% | **−0.003** (improves) | +0.003 |
| 25% | **+0.009** | +0.026 |
| 12.5% | +0.055 | +0.107 |

**bilin18's layer-1 QK is already sparse in the standard basis** — drop 75% of its weights for
+0.009 nats directly, no rotation needed. **F16's learned-basis win was a d=128 TOY ARTIFACT**:
the tiny model packs QK densely (dense in standard basis, sparse only in a learned one); the
real model's QK weights are directly prunable. Gate (keep=1.0 → ΔCE 0) held. MDL: keep-25% =
~25% of the raw QK bits + indices for +0.009 — a genuine sparsity reduction on the flagship
(unlike the toy). The flagship check overturned the toy conclusion — the program's own lesson.

**STEERED-TASK STEP-BACK (F13–F17).** Layer-1 selection decomposition, settled: (1) SOURCE-level
sparsity is real — selection runs on the bilinear-output self-interaction (M×M), the first
attention's output is droppable (F13, toy). (2) WITHIN-source atom compression is model-
dependent: the toy's dominant source is high-rank and needs a learned basis (F14–F16), but on
the FLAGSHIP the layer-1 QK is directly ~75% sparse in the standard basis (F17) — the clean,
simple real-model result. Net: on bilin18, layer-1 query/key is a directly-sparse weight object
(75% prunable, +0.009), and the toy's rotation/decomposition machinery was compensating for
small-model density. Bank against regime-1 baseline (2.1 Mbit toy raw; rotation gave 0).
Next (optional): flagship source-level graph (F13 analog on bilin18); or accept the direct
QK sparsity and move on.

### F16 — M is high-RANK but SPARSE in a learned basis: the interaction decomposition exists
`toy_qk1_learned_basis.py`. The remaining avenue after F13–F15: directly optimize (L4
rotation-to-sparsity, gated by a planted control that recovers 87% of an 89% optimum) a full
O(D) rotation of the M-input basis to sparsify attn2's stacked query/key read maps
[q1;k1;q2;k2]. **The reads sparsify 24.7% in L1, Hoyer 0.24→0.43** — far beyond the head-dim
rotations (regime-1 OV 7%, QK 1.4%). Binding check (prune reads by magnitude, ΔCE):

| keep | original-basis ΔCE | learned-basis ΔCE |
|---|---|---|
| 50% | +0.17 | +0.057 |
| 25% | +1.95 | **+0.14** |
| 12.5% | +3.44 | +0.45 |

The learned basis prunes **14× better** at 25% and lets you drop 75% of the read weights for
+0.14 nats. So M's selection-relevance is high-*rank* (F14/F15 can't low-rank it) but **sparse
in the right basis** — a sparse, not low-rank, structure that variance (F14) and SVD (F15)
both miss. **The interaction-sparse decomposition Logan wanted exists**, found by the stronger
direct optimization. MDL (honest): keep-25% reads (~16k nonzeros) + the basis V (128×128 ≈16k
floats) ≈ 1.0 Mbit for +0.14, comparable to low-rank r=16 (0.52 Mbit, +0.16) — so it's not a
bits win over low-rank, but it is the *sparse interpretable* structure (each M-atom drives few
query/key components) and it prunes vastly better than the naive basis. Updates F15's
"intrinsic" lean → basis-dependent.

### F15 — layer-1 QK is ~rank-64 in the interaction basis (a Pareto trade, not a free reduction)
`toy_qk1_lowrank.py`. Low-rank-reduce attn2's QK maps (q1,k1,q2,k2) to rank r — which is
decomposing every source in the interaction basis and keeping r atoms — ΔCE vs r + bits:

| r | ΔCE | Mbit | % of raw |
|---|---|---|---|
| 2 | +0.93 | 0.07 | 3% |
| 8 | +0.28 | 0.26 | 12% |
| 16 | +0.16 | 0.52 | 25% |
| 32 | +0.07 | 1.05 | 50% |
| 64 | +0.012 | 2.10 | 100% |
| 128 | 0 | 4.19 | 200% (gate) |

Layer-1 QK is **~rank-64** — it does not cleanly compress (r=32 = half the bits for +0.07 nats,
a Pareto point; rank-r only saves bits below r=64). Notably HIGHER rank than layer-0 QK
(QCR-1: rank-16 was free) — selection at layer 1 reads the richer bilinear output.

**STEERED-TASK SYNTHESIS (F13–F15).** Decomposing the layer-1 query/key over its upstream
sources {embedding E, attn0 output A, bilinear output M}: the interaction graph is **sparse at
the SOURCE level** — M×M dominates (70% mass, usable alone +0.062), A is droppable (its pure
blocks negligible), E is a minor low-rank modulation (~8 atoms); 6 of 9 source blocks recover
the model. But it is **NOT compressible WITHIN M**: M is a genuinely high-dimensional selection
signal (~rank-64 in both the variance basis F14 and the interaction basis F15). So the clean
finding is *which sources interact* (sparse, interpretable: layer-1 selects on the bilinear
output self-interaction), not a low-atom-count code for M. MDL: source-level sparsity drops
3 of 9 blocks free; rank reduction is a Pareto trade (half the 2.1-Mbit baseline for +0.07).
Remaining avenue (optional): a LEARNED interaction-sparse basis (direct optimization, the
"stronger technique") might find non-low-rank sparsity that variance/rank miss.

### F14 — per-source atom rank for layer-1 selection: E/A compress, M does not (in the variance basis)
`toy_qk1_source_rank.py`. Decompose each source by PCA of its normed contribution and project
to rank r; recompute the exact block decomposition; ΔCE vs r (gate: full = 0). Results:

| source | r=2 | r=8 | r=16 | r=32 | r=64 |
|---|---|---|---|---|---|
| E (embedding) | +0.038 | +0.021 | +0.018 | +0.014 | +0.007 |
| A (attn0 out) | +0.005 | +0.002 | +0.001 | +0.000 | +0.000 |
| **M (bilinear out)** | **+1.00** | **+0.65** | **+0.40** | **+0.26** | **+0.11** |

E is low-rank (~8 atoms, +0.02) and A is negligible (~2 atoms) — both compress to a handful of
atoms. But **M, the dominant source, does not compress by variance** (rank-64 still +0.11).
**Redirect:** PCA optimizes variance, not interaction — M's high PCA-rank includes
selection-irrelevant variance. The interaction-sparse basis is the query/key SINGULAR basis,
where the M×M form is diagonal per head, so the interaction *graph* can be sparse even with M
full-rank; QCR-1/2 showed the QK bilinear form is itself low-rank (~16–32). **Next:** decompose
M in the QK-singular (interaction) basis / low-rank-reduce the layer-1 QK maps, ΔCE vs rank —
the interaction-sparse decomposition, not the variance one, then its MDL vs the 2.1-Mbit baseline.

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

### F26 — layer-1 QK selection is CONTINUOUS in the used-subspace (equivalence classes don't help)
`bilin18_qk1_vq.py`. Logan's question: beyond removing non-contributing inputs (the used-subspace),
is there 'equivalence-class' compression — some inputs the same for QK (colors attend to colors)?
Cluster the layer-1 QK input INSIDE the used-subspace (r=128) into K discrete classes, held-out ΔCE.
Continuous used-subspace floor +0.0019. VQ: K=16 +0.073, K=64 +0.046, K=256 +0.047, K=1024 +0.026,
K=4096 +0.079 (overfit). **Discrete classes cost ~13× more than the continuous used-subspace even at
their best** (K=1024 +0.026 vs +0.002). So layer-1 QK selection is genuinely CONTINUOUS in the
~128-dim subspace, not a small alphabet of same-for-QK classes — consistent with layer-0 (QCR-1:
low-rank beat clustering). The reduction is 'remove the inputs that don't contribute' (continuous);
hard equivalence classes are not the structure. Sparse features in this basis would be for
INTERPRETABILITY (naming the ~128 directions), not further MDL compression.

### F27 — Task 3: used-subspace generalizes to OV (strong), but the layer-1 bilinear layer is high-dim
`bilin18_ov_mlp_usedsub.py`, held-out. OV (c_v value read): used-subspace beats generic low-rank at
both layers (L1 r=128 +0.006 vs +0.070; L9 +0.005 vs +0.010) — OV reads a low-dim subspace everywhere.
BILINEAR gates (Left/Right): layer-1 does NOT compress (r=256 still +0.14, both methods) — it reads a
genuinely high-dim input, the workhorse computing the features layer-1 QK then reads; layer-9 MLP
DOES compress (r=128 −0.001, used slightly ahead of generic). So input-compressibility is circuit-
and depth-dependent: OV and QK are low-dim everywhere; the early bilinear layer is high-dim. The
activation-aware advantage over generic low-rank is largest where activations are skewed (OV, QK),
small for the bilinear gates. Answers task 3: reductions exist for OV (like QK) but not the early
bilinear layer's input.

### F28 — Task 2: WEIGHT-ONLY info identifies the QK-null of the bilinear layer
`bilin18_qk1_bilinear_null.py` (gate: patch-forward = reference to 5e-6). Block-0 bilinear layer has
4608 hidden units; unit i outputs along Down[:,i]; layer-1 QK reads it with weight-only strength
‖R·Down[:,i]‖ (R = QK reads). Keep top-k by score, subtract dropped units from the QK input, ΔCE:

| keep k / 4608 | weight-only | activation-aware | random |
|---|---|---|---|
| 512 | +0.079 | +0.059 | +0.337 |
| 1024 | +0.024 | +0.008 | +0.139 |
| 2048 | −0.002 | −0.006 | +0.014 |

**QK reads only ~1024–2048 of the 4608 bilinear hidden units; the rest are QK-null, and weights
alone find them** nearly as well as activations (+0.024 vs +0.008 at k=1024), both crushing random.
Resolves Logan's "is the bilinear null two-input-dependent": **no** — the null is on the OUTPUT side.
Each unit's output *direction* is fixed (Down[:,i]) regardless of its two inputs, so the QK-null is a
LINEAR property of the composition Down∘R and is weight-derivable; the two inputs (Left⊙Right) only
set each unit's activation magnitude (which is why adding cheap activation std closes the small gap
to the activation-aware ranking). So: yes, weight-only composition tells you which part of the
bilinear layer to keep for QK — the answer to task 2. Composes with F24: the ~1024 QK-read units
project into the ~128-dim used-subspace QK reads.

### F29 — Task 1: composed compression folds layer-1 QK to {~1024 bilinear units → 128-dim}, sub-additive
`bilin18_composed.py` (gate: patch = reference 5e-6). Compose F28 (keep top-1024 of 4608 bilinear
units, input side) with F24 (used-subspace r=128, weight side). F28 alone +0.024, F24 alone −0.003,
**composed +0.008** — below the sum (+0.021) and below F28 alone, i.e. they compose SUB-additively:
the used-subspace re-optimizes what QK reads and cleans up the unit-drop noise. So **layer-1 query/key
selection folds to {~1024 of 4608 bilinear hidden units → ~128-dim used-subspace} at only +0.008
nats** — the composed-basis compression of task 1, concretely. Coherent with the whole arc: embedding
+OV droppable (F13/F18) → ~1024 bilinear units reach QK (F28, weight-only) → their output lives in a
~128-dim subspace QK reads (F24) → selection is continuous there (F26).

### F30 — CORRECTION to F26: layer-1 QK is 82% current-token-determined; compression is INPUT-relative
`bilin18_qk1_vocab.py` (gate 2.9347 = reference). Logan's correction: F26 clustered the continuous
STATE (positions) and found clustering loses — but the right compression is relative to the INPUT
VOCAB. Measured in the used-subspace QK-1 reads: **between-token variance fraction = 0.824** — 82% of
the layer-1 QK code is determined by the CURRENT TOKEN identity, 18% by context. **Replacing the code
by its current-token mean (a 1286-token vocab table) is near-free: +0.0008.** So layer-1 query/key is
essentially a VOCAB-INDEXED TABLE (the bilinear self-term applied to the current embedding), not rich
context integration; the 18% residual is the cross-terms (current × attended) Logan pointed to.
Clustering tokens into a SMALL alphabet still costs (K=128 +0.087, K=32 +0.124) — tokens are fairly
distinct, so it's "token-lookup," not "colors → one class." **This is the input-relative compression
F26 missed** (F26 clustered states, not the vocab). Reframes the arc: F24's "128-dim continuous read"
is dominated by current-token identity. The remaining program piece — reduce the 18% cross-terms
(vocab × vocab through bilinear → QK-1) — is the DMRG "sparse relative to the neighbor" restricted to
the QK-1 path.

### F31 — QUALITATIVE: layer-1 QK equivalence classes are GRAMMATICAL CATEGORIES (data-validated)
`bilin18_qk1_qualitative.py` + `qualitative_examples_qk1.md` (Logan asked for qualitative examples +
data validation). Cluster the vocab by mean layer-1 QK signature (F30: 82% token-determined), decode
(GPT-2), 40 classes over 139 frequent tokens. **The classes are interpretable part-of-speech categories**:
determiners/possessives (` the ` a ` my ` an ` this ` your ` its ` their ` his`), prepositions
(` of ` to ` in ` for ` on ` as ` at`), auxiliaries/copula (` is ` was ` are ` be ` have ` had ` were
` been`), wh-words/relativizers (` that ` which ` what ` how ` because ` when`), punctuation (`.>:)!`).
So layer-1 query/key selection operates on SYNTACTIC/grammatical structure — the "features relative to
the input" are grammatical categories. Data-validated attention (real co-occurring pairs only): a mix
of local (subword-completion `ctions`→`fun`, `urs`→`Occ`; previous-token) and content attention. This
is the interpretability payoff of the arc: the input-relative reduction (F30) yields linguistically
meaningful token classes.

### F32 — cross-term reduction: attended tokens reduce to ~16-64 interpretable classes for QK-1 (data-validated)
`bilin18_crossterm.py` + `crossterm_value_classes.md` (gate 2.9347 = reference, Δ0). The 18% context part
of layer-1 QK (F30) comes from attended tokens via OV. Cluster the layer-0 value table into K classes,
re-aggregate through the REAL block-0 attention (data validation — only pairs that occur), ΔCE on layer-1
QK: raw-value K=16 +0.043, K=64 +0.056, K=256 +0.026. **The attended-token vocab reduces to ~16-64
equivalence classes** for QK-1's context — coarser than the current-token side (fits: context is only 18%).
Classes are interpretable: numbers (`1 3 10 8`), wh-words/demonstratives (`that what this how where which
who`), quantifiers/degree (`some not more other very all many`). **Honest negative:** a linear
QK-1-effect proxy (value → Down → Right → QK reads) did NOT beat raw-value clustering (K=16 +0.14 vs
+0.04), so Logan's "composed features beat individual" is NOT confirmed here — the true cross-term is
bilinear and current-token-dependent, so a linear path-proxy is too crude; a proper composed metric
(joint current×attended through the bilinear path) is the right next test.

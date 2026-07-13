# Jacobian clustering for bilinear layers — what is actually true

Authoritative synthesis of the program. Read this instead of `results.md` (a chronological log with
retracted claims sitting next to survivors). Every claim carries provenance; retractions are tabulated.

**Standing rules that shaped this (carried from mechdecomp, where they reversed five headlines):** every
clustering claim needs a control that *could* fail (matched-dim random projection / spectrum-matched
random-eigenvector metric / random-matrix; chance measured, never assumed 0); no single-seed/top-k/
max-over-sample statistics; verify every construction against a known identity before clustering with it;
state confounds; report the block/ablation decomposition that explains a result, not just the headline.

---

## 1. The one-paragraph bottom line

For a bilinear layer `y = D(Lx ⊙ Rx)` the per-datapoint Jacobian kernel `⟨J(x),J(x')⟩_F = xᵀGx'` is exact
and weights-only. Clustering datapoints by this metric provably recovers mechanism geometry **in toys**
(clusters, rings, hierarchies, learned modules). On **real trained bilinear MLPs it reduces to input
cosine** — not because the method is wrong but because a measurable weights+data property fails: the
method's advantage scales with how anisotropic the operator is *on the data it sees*, and trained MLPs
are near-isotropic there. On **bilinear/squared attention** the per-token operation Jacobians are strongly
anisotropic, and clustering by them beats a random-matrix control ~15–25× (every head, 124M+573M) and
beats residual clustering by 2–9× at mid/late-layer heads — the program's only real-model positive,
scale-confirmed and 5-seed-replicated, though head-dependent and validated against a pattern-summary
rather than verified interpretable roles.

**Elegant theory, clean toys, and a precise weights-only law for when it helps — but NO demonstrated
real-model CAUSAL win for the Jacobian itself: on trained MLP activations it reduces to input cosine, and
on attention the per-token Jacobian is causally ≈ random (tick 23). What the investigation DID find: the
causally-meaningful attention object is the QUERY readout `Wq·x` (clustering it gives causally-distinct
operations, ~5× above Jacobian/residual/random, 5 seeds). Program theme: the useful object is always a
WEIGHT-DERIVED READOUT that isolates the operation (content-restricted J for MLPs, `Wq·x` for attention);
the FULL Jacobian is contaminated in both cases — the impossibility theorem's shape, recurring.
UNIFICATION (tick 35): the query readout is NOT "just an activation" — squared attention makes `z_q` an exact
bilinear layer in `x_q` (gates = `q1·k1`, `q2·k2`), so clustering by `q1,q2` IS the gate-restricted Jacobian
of the per-query operator (the DGP-D/S8 winner). The literal secant `M=z·xᵀ` (`y@x⁻¹`) works on attention but
is WEAKER than the query readout (it re-adds the context-contaminated output `z_q`). So across MLP and
attention the same rule holds: the useful object is the RESTRICTED operator's gate factor, not the full
operator/secant.
Extending to BOTH q and k weights (the QK score kernel `G_QK = Wqᵀ Wk Wkᵀ Wq`, verified) is causally
REDUNDANT with the query alone (+0.0072±0.0014 vs +0.0090±0.0050, tied) — the query already implicitly
contains key-selectivity, since the score is `query·key` (tick 25).**

---

## 2. Solid results (with provenance)

| # | result | evidence | provenance |
|---|---|---|---|
| S1 | Exact kernel `⟨J,J'⟩_F = tr(Jᵀ J') = xᵀGx'`, `G` weights-only closed form | max rel err 1e-14; gauge (`G(D,L,R)=G(D,R,L)`), Euler (`J(x)x=2y`), autodiff all pass | deterministic identities |
| S2 | Frobenius cosine == flattened-vector cosine (Logan's trace Q) | identical to 12 dp | identity |
| S3 | rank-1 `M_i = y xᵀ/‖x‖²` factorizes: `cos(M_i,M_j)=cos_x·cos_y` | machine precision | identity |
| S4 | **Impossibility:** a single bilinear layer's full `J` cannot separate input-gated mechanisms — the gate-derivative block is O(1/ε) and gate-INDEPENDENT | measured 0.00e+00 across gates; empty ε-dissociation-window | DGP-A + proof |
| S5 | degree-2 homogeneity ⇒ `f(-x)=f(x)`, `J(-x)=-J(x)`; use `|cos|` globally | `cos` 0.175 vs `|cos|` 1.000 on antipodal | DGP-B, exact identities |
| S6 | **Compositionality:** J₁→g₁, J₂→g₂, end-to-end J→joint (g₁,g₂), all ARI 1.000; content at chance | 5 seeds; matched-dim random-proj control recovers content only | DGP-C |
| S7 | top-`k_g` eigenvectors of `G` ARE the gate subspace exactly | principal-angle cos 1.0000, every ε | P4 |
| S8 | **Learned modules recovered:** trained bilinear MLP, restricted-J ARI 1.000 vs input 0.002, output 0.214, rank-1 M 0.331, random 0.002 | 5 seeds, chance measured | DGP-D |
| S9 | depth + nonlinear gate breaks S4 (full J becomes gate-dependent, ARI 0.389; operator block 1.000, diluted 6.1× by gate-deriv block) | column-block decomposition | DGP-E |
| S10 | **Geometry recovery:** rotation family → circle, circ-corr 0.999, trustworthiness 1.000; input/output/random-proj at chance | 5 seeds, construction 0.00e+00 | DGP-A′ ring |
| S10b | **Hierarchical geometry:** planted 2-level tree (3 coarse × 4 leaves) recovered by content-restricted J \|cos\| at BOTH levels — coarse+fine ARI 1.000, graded tree distances (same-leaf 0.000 ≪ same-coarse 0.108 ≪ diff-coarse 0.949, tree-ρ 0.83). Input-x control recovers nothing (content cross-cuts). Caveat: hand-built layer ⇒ restricted-J = planted A_leaf, so random-proj-of-J also recovers it (input-x is the could-fail control); trained-module version noted, not run. | 5 seeds | tick 32 |
| S11 | **The law:** method advantage over cosine ∝ `1 − corr(cos_J,cos_x)` on the DATA (ρ=0.83); weights-only `1−eff_rank(G)/d` is ρ=0.76 (necessary, not sufficient — blind to data alignment) | 2D weight×data sweep, 5 seeds | tick 15 |
| S12 | Sketched VJP kernel: corr 0.83 (k=1) → 0.993 (k=50); ARI saturates at k=1 | quote kernel error not ARI for probe counts | DGP-C |
| S13 | `M`-clustering (linear transformation) beats activations on toys (0.355 vs 0.002) but LOSES on real MLPs | output-similarity is wrong grouping without a gate | Logan Q |
| S14 | **Attention operations anisotropic:** J beats random-matrix control ~8–11× at every head; beats residual clustering ~2× (not the inflated 5–16× of ticks 17–20 — a ROTARY SIGN BUG overstated magnitudes; qualitative direction survives, effect sizes corrected down) at most heads | custom forward verified vs model CE (0.00e+00); 5 seeds; ticks 17–21 | tick 21 |
| S15 | **QK score kernel (both q,k weights):** `G_QK = Wqᵀ Wk Wkᵀ Wq`, key-selectivity covector `c_q = Wkᵀ Wq x_q`, `⟨c_q,c_q'⟩=x_qᵀ G_QK x_q'` — identity verified (3e-7). Causally REDUNDANT with query-only: +0.0072±0.0014 vs query +0.0090±0.0050 (tied), both ≫ residual x +0.0005 ≫ random 0. Query already contains key-selectivity (score = query·key) | L6H0, 5 seeds, causal swap-within test | tick 25 |
| S19 | **SAEs on the mechanism object (Logan).** Matched TopK SAEs on a gated-superposition toy: the RESTRICTED-Jacobian SAE recovers the gate/mechanism (purity 0.965, content at chance 0.125) — mirror image of the activation SAE (content 0.808, gate at chance). Neither sees the other's features. BUT only the restricted J: raw/normalized full-J and G-embedding all recover content (impossibility theorem — mechanism signal is the gate-independent O(1/ε) block; content dominates SAE variance). SAE/superposition analog of S8. **Non-degenerate confirmation (tick 38): with a sparse SET of operators/token (3 of 10, nonlinear-in-content), the operator-block-J SAE recovers the operator DICTIONARY at MMCS 0.926 in genuine superposition — which activation-SAEs cannot access (operators aren't in x-space); J_full contaminated (MMCS 0.586). Caveat: x-SAE op-purity 0.74 > chance (gate correlates w/ content) so the win is decisive on dictionary recovery, modest on active-set.** Real-model tests (attention SAE on q1,q2; weights-only-restricted on a real MLP) pending. | 5 seeds | ticks 37–38 |
| S20 | **SAE thread, cross-architecture (Logan; ticks 37–39).** Toy (ground truth): SAE on the mechanism object recovers mechanism/operators — restricted-J SAE gate-purity 0.965 (content at chance), and in genuine superposition the operator-block-J SAE recovers the operator DICTIONARY at MMCS 0.926, which activation-SAEs cannot (operators aren't in x-space); restriction necessary (full-J MMCS 0.586). **Real attention (500M, causal swap-within, label-free):** SAE-on-query beats SAE-on-residual at 3/4 heads (1.5–3×) but REVERSES at L8H3 — directional, head-dependent, NOT robust. **Real MLP: UNBLOCKED by Logan's bilinear-secant SAE (ticks 42–44).** A bilinear SAE reconstructing the secant M=y·x⁺ as a sparse sum of rank-1 atoms, with Dooms' expanded quadratic loss (verified exact), never forms the d×d secant ⇒ feasible at d=1152. NON-NULL positive: block2 secant-FVU 0.22–0.35 (vs 0.99 random atoms); 500M layer-6 FVU 0.117 (93% of MLP output from 32 sparse rank-1 operators), layer-dependent (L12 weaker 0.43). Reconstruction succeeds where CLUSTERING was null (S11) — near-isotropy kills clustering separation, not sparse operator reconstruction. Outliers mild and well-captured (high-norm tokens reconstructed *better*; no LLM.int8 spikes, post-RMSNorm). This is the first non-null real-MLP result for the operator object. Dense low-rank control PASSES (tick 45): sparse operator-dict beats dense low-rank at matched/higher active count — decisive on block2 (sparse-16 beats dense-64 2×), modest on 500M-L6 (mostly low-rank there). So sparsity is real, not low-rank compression. Overall: mechanism-object SAEs clean with ground truth, head-dependent on real attention, and now a real, feasible, non-null win on real MLPs via the bilinear-secant architecture. | 5 seeds | ticks 37–44 |
| S17 | **Genuine two-QK bilinear (real 500M checkpoint).** `gpt2-bilinear-sqrd-attn-18l` has c_q,c_k,c_q2,c_k2: `pattern=(q1·k1)(q2·k2)/D²` unnormalized (custom fwd = model CE 3.79 @2e-4). Clustering queries by BOTH matrices helps **on causally-important heads** — survives the matched-dim control (`[q1;q2]` > `[q1;noise]`,`[q2;noise]`) at 3/4 (L0H2,L6H3,L8H3; L8H3 cleanest: noise kills q1 but q2 rescues). The two query matrices are near-orthogonal (data \|cos(q1,q2)\|≈0.05–0.08) so q2 is complementary, NOT redundant — CONTRAST with KEY weights (S15, redundant). **BUT head-dependent** (≈0/neg on most heads, tick 28) and **NO weights-only pre-screen**: alignment `A_w=‖Wq_h Wq2_hᵀ‖/(‖·‖‖·‖)` is near-constant ~0.06 across all 162 heads, uncorr (+0.05) with data\|cos\| — the S11-style law does not materialize. L3H3's tick-27 "tensor win" was a dimensionality artifact. | L0–L8 heads, 5 seeds, swap-within + matched-dim control | ticks 27–28 |
| S18 | **What two-QK attention is FOR (mechanistic, all 162 heads of 500M).** The product pattern `(q1·k1)(q2·k2)` is raw/unnormalized ⇒ two capabilities a squared single-QK head (`(q·k)²≥0`) structurally lacks: (i) **SIGNED attention** — frac_neg_mass 0.49±0.27 (range 0.008–0.986; half the attention mass subtractive on average; heads span pure-positive→pure-suppression); (ii) **conjunctive sharpening** — PR(product)=0.32 < min(PR(s1),PR(s2))≈0.59 at 100% of heads (product concentrates on fewer keys than either factor). Caveat: neg-mass does NOT track the S17 clustering gain — separate phenomena. **Causal ablation (tick 31): signed attention is broadly PRESENT but functionally CONCENTRATED — clamp_neg load-bearing at only ~7/162 heads (L1H1 standout +0.057, also highest neg-mass; some late layers slightly benefit from removal). Global ablation uninformative (matched-mass control drop_ctrl +3.64 > clamp_neg +1.77). Pervasive ≠ important.** | all 162 heads, pattern verified vs model CE 2e-4 | ticks 30–31 |
| S16 | **Squared-attn tensor kernel + OV pathway.** (a) squared attn ⇒ query op is the rank-1 form `u_qu_qᵀ` ⇒ correct kernel is `cos²`/`|cos|`; beats plain query-cos +0.0120±0.0015 vs +0.0096±0.0042 (consistent, ~3× tighter). (b) OV/value pathway (cluster tokens by head value `v_k`) is causally clustered above random at all 5 heads, above residual-x at 4/5 (modest ~1.3–1.6×; L6H0 +0.042 an outlier — single-head over-read, corrected by head sweep). (c) query vs value clusters near-independent (ARI 0.07) ⇒ unlike KEY (redundant), OV carries independent structure; head factors as WHERE(query)×WHAT(OV). Two-different-QK bilinear now tested on a real 500M checkpoint — see S17. | L2–L11 heads, 5 seeds, swap-within | tick 26 |

## 3. Real-model characterization (the central negative, then the positive)

**MLPs (negative, fully explained).** On trained bilinear MLPs (124M `gpt2-bilinear-12l`, 573M
`18l`, tiny block2): `G` effective rank 0.86–0.94 at every layer (near-isotropic). Surrogate R², seeded,
matched controls: the G-metric ties raw x and the spectrum-matched G_rand; a swap-one-thing test (same
G, vary only input) gives isotropic-Gaussian input +0.066 but **real activations +0.0012, negative vs
control**. Cause: real activations sit where `G` is locally flat (`corr(cos_J,cos_x)=0.62–0.94`).
Matched-covariance Gaussian keeps +0.037, so the killer is non-Gaussian activation structure (heavy
tails, token clusters), NOT covariance. Outlier fold-out (LLM.int8 style) flips G-vs-control from
−0.003 to +0.007 but is blocked by all bilinear models being post-RMSNorm (no dramatic outliers).

**Attention (positive, modest, head-dependent).** Per-token operation Jacobian `J_q = ∂z_q/∂x_q`
(no closed-form kernel — materialized via autodiff). Beats random-matrix control ~15–25× at every head
(124M+573M) and residual clustering by 2–9× at mid/late-layer heads (6/9 heads at 573M; L9/L15). The
advantage is at MID/LATE layers, NOT early (tick-17's "early wins" was backwards — confirmed across both
models); it is head-specific. Cross-model-confirmed, 5 seeds.

**How truth is judged on real models:** no ground-truth mechanism labels exist. Two label-free proxies —
surrogate test (per-cluster linear map predicts held-out output; beat cosine/G_rand/random/global) and
intervention test (within- vs across-cluster patch CE). Both verify "same-map clusters," NOT
"interpretable roles." Real ground truth lives only in the toys.

## 4. Retracted (with cause) — the standing rules at work

| retracted claim | cause | tick |
|---|---|---|
| "attention: early layers win, late lose (positional vs content mechanism)" | over-read from 4 hand-picked heads; systematic sweep shows head-dependent, late layers win more | 17→18 |
| "real models sit exactly on the anisotropy law" | law is isotropic-input-only; real data 10–50× below prediction | 12→13 |
| "small-but-real real-model advantage" (ticks 6–10) | on like-for-like isolation, ~0/negative vs control | 13 |
| "tick 8 scale-negative: clustering hurts / G ties raw x" | ridge fit-instability at 768-dim, 500 pts/cluster | 8→9 |
| "the decomposability pre-test separates the regimes" | confounded by K/rank(W) | tick |
| squared-attn model "just trained to CE 7.5 (weak variant)" | repo `naive_squared_attention` is UNnormalized; checkpoint trained normalized → CE 3.5 | 16 |
| attention "meaningfulness" first attempt | degenerate pattern-type label (92% one class) | 17 |
| attention magnitudes (J beats x by 5–16×; 573M "2–9×") | ROTARY SIGN BUG in apply_rot — wrong rotation direction; corrected to ~2× | 21 |
| attention pattern-summary "wins" (J/x beat random on recovering attention behavior) | the pattern-summary measures POSITION/distance, not causal function; disagrees with the causal test (query best causally, worst on pattern-summary) | 24 |

## 5. Practical recipe (weights-only pre-screen)

Before clustering any bilinear layer: compute `corr(cos_J, cos_x)` on its real activations (cheap; or
`eff_rank(G)` as a weaker weights-only proxy). The method's advantage over cosine ∝ `1 − corr`. If corr
is high (real MLPs: 0.62–0.94) don't bother. If low (attention ops), cluster and validate with BOTH a
random-matrix/random-projection control AND a non-derived target. Use `|cos|` (or Euclidean/gain-kept on
real MLPs — the gain `‖J‖²_F = xᵀGx` is free from the kernel and matters on real data).

## 6. Open (would each need its own careful pass)

1. Are the winning attention heads' J-clusters nameable roles (prev-token, BOS-sink, positional)? Needs
   an intervention/causal test, not a pattern-summary (which J is partly derived from → mild circularity).
2. Confirm the attention positive on the 573M model beyond the single leverage screen.
3. A model whose activations populate an anisotropic MLP `G` (the MLP null is data-driven, not universal).
4. The pattern-summary target shares the pattern as a common cause with J — a non-derived causal target
   would make the attention result publishable rather than suggestive.

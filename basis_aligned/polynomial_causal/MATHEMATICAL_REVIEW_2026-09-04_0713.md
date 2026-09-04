# THREE-HOURLY MATHEMATICAL REVIEW — 2026-09-04 07:13Z (Claude, lane 1)

State read from disk before writing: ledger §2808–§2853, BENCHMARK_BACKLOG tail, `runlogs/_completed.txt` (last landing 07:07
`circuit_battery_calibrated_selectivity`; all runs ledgered), queue empty, GPU 0%, board tail including Codex's task17/task21 adapter
work, `HOURLY_STRATEGIC_REVIEW_2026-09-04_0641.md`, and `ops/EFFICIENCY_LOG.md` through the 07:13Z row. Literature check run for this
review by a web-enabled subagent; the 2026 arXiv ids below are cited as that search reported them and were not personally fetched.

Explained fraction unchanged: **5.348% / 10.923% / 4.727 nat / 0 of 68**. Frontier L2 is CE ADDED ABOVE THE REAL MODEL, LOWER IS
BETTER (§2135; §312 norm-2304 at 2.6735); nothing in §2808–§2853 is an L2 and nothing installs.

## 0. What changed the mathematical situation since the 04:04Z review

Three things, all measured since:

1. **§2812**: for this architecture the composition of a bilinear MLP with RMSNorm is an EXACT (2,2)-rational map along any removal
   ray — `mlp(rms_norm(x − tW)) − b = D·[Q(x) − tB(x,W) + t²Q(W)] / [‖x‖² − 2t⟨x,W⟩ + t²‖W‖²]`, verified to 8.3e-7. We have the exact
   algebraic form of what the normalization does.
2. **§2822–§2826**: energy/variance rankings do not find causal structure here. The in-sample rank-4 subspace of a removal effect holds
   .700 of its energy and delivers .139 of its damage; a single unembedding-defined direction holding **.0021** of the energy delivers
   .199 of the damage at 2.4× the block's specificity.
3. **§2850–§2852**: intervention arms above the ceiling make selectivity ratios arithmetic. Whole-component ablation removes 1.538× the
   native margin on target *and* controls; the battery's writer arm 1.207×/2.552×. Calibrated to .698 the campaign's negative survived
   with a median ratio change of .000.

Each of those bears directly on a live published claim, which is what makes this review different from the 04:04Z one.

## 1. Literature, and where it collides with our measurements

- **Sharkey, "A technical note on bilinear layers for interpretability,"** arXiv:2305.03452 (2023) — the originating proposal that a
  gate-free bilinear layer is expressible with linear operations and a third-order tensor.
- **Pearce, Dooms, Rigg, Oramas, Sharkey, "Bilinear MLPs enable weight-based mechanistic interpretability,"** arXiv:2410.08417, ICLR
  2025 Spotlight; predecessor **Dooms, Rigg et al.**, arXiv:2406.03947. The operational object: fold `Down(Left(x)⊙Right(x))` into a
  third-order tensor B, contract B with an output/probe direction to get a SYMMETRIC matrix, and **eigendecompose it**; the top
  eigenvectors are the claimed interpretable directions, obtained **from weights alone**. Stated demonstrations are toy, vision and
  small-LM; the search found no treatment of RMSNorm or of multi-layer residual composition.
- **Usevich et al., "Identifiability of Deep Polynomial Neural Networks,"** arXiv:2506.17093, NeurIPS 2025 — generic identifiability of
  deep polynomial networks via **partially symmetric CP decomposition and Kruskal-type uniqueness**, for non-increasing widths. Stated
  **without any normalization layer**.
- **Heimersheim & Nanda, "How to use and interpret activation patching,"** arXiv:2404.15255 — documents that noising and denoising are
  not complementary and that large corruptions introduce confounds. The closest published statement to a saturation artifact; it does
  NOT formalize a graded/dose-response intervention.
- **Braun et al., APD,** arXiv:2501.14926 — MDL in parameter space; the authors themselves flag that scaling beyond toy models is open.
- Certified-circuit work (arXiv:2602.22968, arXiv:2602.16823 as reported) certifies the DISCOVERY procedure's stability, not a numeric
  error bound for replacing a component with a surrogate under residual+norm composition. The search found no such theorem.

**The collision that matters.** Pearce et al.'s method ranks directions by **eigenvalue magnitude of a contracted bilinear form** —
which is precisely the energy-ranking family that §2822–§2826 measured as non-aligned with causal effect in this model. Their claim is
validated at toy/small scale; bilin18 is 546M with RMSNorm and squared attention. **Nobody has tested weight-based bilinear
eigendecomposition against a causal ground truth at this scale, and we are unusually well placed to: we have the exact composition law
(§2812), a calibrated intervention (§2852), and an independently established causal axis (§2826).**

## 2. Pruned

- **Bare CP/Tucker decomposition of the third-order tensor as a compression device** — optimizes reconstruction, and §2824/§2825 already
  showed energy-optimal subspaces of the *effect* carry little damage. Pruned unless scored causally.
- **A fourth ranking heuristic for constant-write sets** — §2837/§2838/§2839 exhausted that lineage, and §2838 passed all five
  predictions on a vacuous set.
- **Learned alignment (DAS-style)** — adds fitted parameters to a protocol whose value is having none, and the gameability critique
  (arXiv:2507.08802) applies.
- **Bisimulation / automata minimization** — still no enumerable state space, still no cheap falsifier. Recorded, not scheduled.
- **APD at 546M** — its own authors flag scaling as open; not a cheap experiment for us.

## 3. The top three genuinely new mathematical moves

### Move 1 (rank 1, EXECUTED BELOW) — score the published weight-space eigendecomposition against causal ground truth

**Object.** For reader block ℓ (mlp8, mlp10), the third-order tensor `B` with `mlp(x) = Down(Left(x) ⊙ Right(x)) + b`. Contract it with
an output direction `u` (we have the right one: §2826's `W_U[answer] − W_U[competitor]`) to get the symmetric matrix
`M_u = ½(A_u + A_uᵀ)` where `(A_u)_{ij} = Σ_k u_k Down_{k,·}(Left_i ⊙ Right_j)`; eigendecompose `M_u`. This is exactly Pearce et al.'s
operational definition, applied to our architecture.

**Measurable consequence beyond reconstruction.** Their eigenvectors are proposed as *the* interpretable directions from weights alone.
Our test: rank the eigenvectors by |eigenvalue| and, separately, by measured causal damage when that direction is removed from the
block's read. **If |eigenvalue| predicts causal damage (Spearman ≥ .5), weight-only extraction works at 546M and the campaign gains a
data-free way to enumerate a block's causal directions — a genuine step toward compiling.** If it does not (ρ ≈ 0), then the ICLR'25
method's ranking does not transfer to a normalized deep model, which is a citable negative and explains §2822–§2825 in one sentence.

**Assumptions that may fail.** `M_u` is defined for a fixed output direction and our answer axis is per-row; the contraction must be
done per-row or on a pooled axis (we do both). RMSNorm scales the input, so eigenvectors of `M_u` act on the NORMALIZED input, and the
map from residual-space directions to normalized-space directions is state-dependent — §2812 gives that map exactly, so it is handled
rather than assumed away.

**Cheapest falsifying experiment.** CPU: build `M_u` from weights for mlp8 and mlp10 (1152×1152, two matmuls plus a symmetrization),
eigendecompose, inspect the spectrum. GPU: remove the top-k eigendirections from the block's read and measure damage, against
|eigenvalue|-matched random directions. Under a GPU-minute.

### Move 2 (rank 2) — identifiability under RMSNorm: extend Usevich et al. across the normalization

**Object.** The same tensor `B`, observed only through `mlp(rms_norm(u))`.

**The bridge, and it is short.** Usevich et al. prove generic identifiability of deep polynomial nets via partially symmetric CP with
Kruskal conditions, assuming no normalization. §2812 gives `mlp(rms_norm(u)) − b = D·Q(u)/‖u‖²` exactly, and `‖u‖²` is a KNOWN scalar
function of the observed input. So multiplying every observation by `‖u‖²/D` recovers `Q(u)` exactly, and **their identifiability
theorem transfers verbatim to the normalized layer.** RMSNorm does not obstruct identifiability of the bilinear form; it is a known
invertible rescaling on each ray.

**Measurable consequence.** If the Kruskal condition holds for our `Left`/`Right`/`Down` factor matrices, the decomposition of each MLP
block is essentially unique — which converts "a basis we chose" into "the basis", and is exactly the gauge question this program keeps
running into (§2118's closed metric-constructed spans were closed because they were arbitrary; a Kruskal-unique decomposition would not
be).

**Assumptions that may fail.** Kruskal rank is a generic-position statement; trained weights are not random, and the condition
`k_L + k_R + k_D ≥ 2r + 2` must be CHECKED numerically, not assumed. Their theorem is for non-increasing widths; ours expands 1152 →
4608 → 1152, so the encoder-decoder caveat applies and may bite.

**Cheapest falsifying experiment.** CPU only: estimate the Kruskal ranks of Left, Right and Down for one block by numerical rank of
random column subsets, and check the inequality. Minutes, no GPU, no model run.

### Move 3 (rank 3) — a saturation-free selectivity: the derivative at zero intervention

**Object.** The damage functions `d_A1(t)`, `d_P(t)`, `d_C(t)` along the removal ray, which §2812 proved are exactly (2,2)-rational
in `t`.

**The move.** §2850–§2852 showed a ratio evaluated at full strength is arithmetic when both sides saturate, and that calibrating the arm
is a workaround with an arbitrary ceiling. The principled object is the **initial slope ratio** `ḋ_P(0)/ḋ_A1(0)`, which is
saturation-free by construction and, because the response is exactly rational, is available in CLOSED FORM from three measurements
along the ray rather than from a sweep. The literature check found no published dose-response formalization of ablation strength
(arXiv:2404.15255 notes the confound without formalizing it), so this is an open niche and we already own the two ingredients.

**Assumptions that may fail.** The rational form is exact for a SINGLE reader's arm with the rest of the residual held fixed (§2812's
own scope note); for multi-edge arms it is an approximation whose error is unmeasured. And a derivative at zero is a local quantity — a
component could be selective infinitesimally and generic at realistic strengths, which is a real interpretive risk to state.

**Cheapest falsifying experiment.** Compute the closed-form initial slopes for the four reader edges of attention 8 on the numbered
list, compare with a finite-difference estimate at t = .05, and check whether the resulting selectivity ordering matches §2819's
depth gradient. Under a GPU-minute.

## 4. Executed

Move 1's CPU half, and the preregistration of its GPU half. The weight-space objects are built and their spectra recorded in
`ops/bilinear_eigen_cpu_probe.py` output; the causal-scoring rung is registered as
`CIRCUIT_BATTERY_BILINEAR_EIGEN_CAUSAL_PREREGISTRATION.md` and enqueued. Move 2's Kruskal check is CPU-only and is the next item;
Move 3 follows it.

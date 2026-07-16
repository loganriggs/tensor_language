# The methods that worked: objects, shapes, reductions

Companion files: [GLOSSARY.md](GLOSSARY.md) (term definitions), [EXAMPLES.md](EXAMPLES.md)
(qualitative examples with specifics). Numbers cited here are ΔCE on pile-10k at $T=512$
(the binding metric) unless noted.

## 0. Models and notation

| symbol | bilin18 (546M) | sqrd12 (162M) |
|---|---|---|
| vocab $V$ | 50,257 | 50,257 |
| model dim $d$ | 1152 | 768 |
| heads $H$ × head dim $h$ | 9 × 128 | 6 × 128 |
| layers $L$ | 18 | 12 |
| attention | two-branch bilinear, **no softmax**: $P_{ij} = \frac{(q^1_i\!\cdot\!k^1_j)(q^2_i\!\cdot\!k^2_j)}{h^2}$, causal-masked, **unnormalized** | single branch: $P_{ij} = \frac{(q_i\cdot k_j)^2/h^2}{\sum_{j'} (\cdot)}$ (row-normalized) |
| MLP | pure bilinear $W_D(W_Lx \odot W_Rx)$ | ReLU² |

Both use pre-RMSNorm (affine-free), per-head QK-norm, rotate-half RoPE on both branches,
a learned $\lambda$-mix with the layer-0 residual ($x \leftarrow \lambda_0 x + \lambda_1 x_0$),
and a value-lerp to the first layer's $v$.

## 1. The objects, their shapes, their reductions

### Embedding
$E \in \mathbb{R}^{V\times d}$. After the model's own RMSNorm: $\hat e_t = e_t / \mathrm{rms}(e_t)$.
Never reduced directly in this program — its *images under the circuit maps* are the objects
that reduce (below). Key map fact: above layer 1, the embedding's selection role is entirely
mediated by MLP-0 (its direct score energy is ≈0), so "what does the embedding do" is only
answerable through the folded objects.

### QK circuit (selection)
Per layer, head $a$, branch $b$: linear maps $W_q^{ab}, W_k^{ab} \in \mathbb{R}^{h\times d}$
applied to the normed residual, then QK-normed and rotated. **At layer 0 this folds exactly**:
selection depends only on token identities and positions, so each head-branch owns two
*factor tables*
$$\hat q^{ab},\ \hat k^{ab} \in \mathbb{R}^{V\times h},\qquad
\hat q^{ab}_t = \mathrm{RMSNorm}\!\left(W_q^{ab}\,\hat e_t\right),$$
and the pre-RoPE score is $S_{ij} = \hat q_{t_i}\!\cdot\!\hat k_{t_j}/h$. RoPE enters through
the difference identities (using the model's own trig tables, so the fold is exact to ~1e-15):
$$S_{ij} = \tfrac1h\big[(q_a k_a + q_b k_b)\cos\Delta_{ij} + (q_b k_a - q_a k_b)\sin\Delta_{ij}\big].$$
**Reductions** (per head-branch): rank ($\hat q \to U_r \Sigma_r V_r^\top$, works when a head is
genuinely spectral — sqrd12 H3); **token classes / vq-k**: one shared k-means partition of the
rows of $[\hat q \,|\, \hat k] \in \mathbb{R}^{V\times 2h}$, storing an assignment
$a: V \to [k]$ ($V\log_2 k$ bits) plus $k$ atom pairs ($2kh$ floats). Selection tolerates
$k\approx 256$ hard classes nearly everywhere. At layers ≥1 the fold is impossible (inputs are
contextual) → **cond-mean tables** (§2, M4).

### OV circuit (carriage)
Per head: $W_v^a \in \mathbb{R}^{h\times d}$ and the head's slice of the output projection
$W_o^a \in \mathbb{R}^{d\times h}$. Folded at layer 0: value table
$VT \in \mathbb{R}^{V\times H\times h}$, $VT_t^a = W_v^a \hat e_t$. **Reductions**: token
classes FAIL here (+1.38 at vq256 — carriage needs identity); a **sparse dictionary** works
($n$ atoms, each token = $t$ signed coefficients: $VT_t \approx \sum_{s=1}^{t} c_{ts} D_{j_{ts}}$;
$n{=}512, t{=}16$ reached −0.019 CE-trained). At layers ≥1, cond-mean $v$-reads are nearly free
(+0.004 at $W{=}6$) because they preserve exactly what carriage needs: token identity.

### Attention heads (as individuals)
A head-branch is the pair $(\hat q^{ab}, \hat k^{ab})$ plus its $VT^a, W_o^a$ slice. Heads
differ enormously and the *marginal* menu per head (zero / rank / classes / positional) never
predicts the composed cost. The two heads that resist every token-static treatment: L5.H5
(induction match; carries noisy identity) and L5.H7 (causally **rank-1**: one output direction
× a live scalar gain).

### Bilinear MLP — and yes, encoder/decoder split
$$\mathrm{mlp}(x) = W_D\,(W_L \hat x \odot W_R \hat x) + b_D,\qquad
W_L, W_R \in \mathbb{R}^{4d\times d}\ \text{(the encoder pair)},\ W_D \in \mathbb{R}^{d\times 4d}\ \text{(the decoder)}.$$
The split is real and useful: the encoders define a bilinear *interaction form* — everything
class-/stream-structured happens on their inputs — while the decoder is a linear read whose
column norms weight the hidden interaction energies (that's how the exact stream-pair maps are
computed). Input-side structure: with the residual split into streams (below), the hidden is a
sum over stream pairs $W_L s_a \odot W_R s_b$; at layer 0 this specializes to the
**self / cross / pair** blocks ($e{\odot}e$, $e{\odot}a$, $a{\odot}a$), each individually
class-tolerant at $k\approx256$.

### Streams (the residual identity)
The residual before layer $\ell$ is an **exact** linear sum of $2\ell{+}1$ streams: the
embedding path (closed under the $\lambda$-mix, always token-determined) and each lower layer's
attn-out and mlp-out. Both RMSNorms are per-position *scalars*, so every bilinear object
(scores, MLP hidden) decomposes **exactly over stream pairs** — this is what makes the
interaction maps gate-checkable rather than approximate. Each stream reduces to a cond-mean
table $\bar s(t) \in \mathbb{R}^{V \times d}$ (§2, M4).

## 2. The methods (what worked, with the math)

**M1 — Exact folding + gates.** Layer-0 circuits are functions of $(t_i, t_j, \Delta)$ only;
fold them into vocab-indexed tables and verify the patched model reproduces the live one to
fp tolerance (~1e-13…1e-15) *before* any claim. Every solver change re-runs the gate.

**M2 — Token classes (vq-k).** K-means on the rows of a folded table; the description is
$V\log_2 k$ assignment bits + $k$ atoms. Two hard lessons: partitions are chaotic (identical
seeds differ via GPU atomics; spread ±0.03 at $k{=}64$), and the *metric doesn't matter* —
Fisher-weighted, unembedding-projected, and directly behavioral objectives all converge into
the same basin as plain L2 (results/14). Quantization error behaves as noise the stack filters.

**M3 — Sparse dictionaries.** For identity-carrying objects (OV), replace classes with
$t$-sparse combinations over $n$ atoms. "Comparisons need classes; carriage needs identity" —
sparse codes preserve identity at a fraction of the bits.

**M4 — Conditional-mean tables (0th-order-in-context).** For any contextual object $z_i$
(a layer-≥1 factor, a stream), estimate $\bar z(t) = \mathbb{E}[z_i \mid t_i = t]$ over data,
then renormalize to the object's natural shell (unit-RMS for QK-normed factors — the gauge
matters, 3× cost if skipped). Data-estimated, not weight-folded: report estimation tokens
alongside structural bits, and mind that table quality tracks estimation **diversity**, not
volume.

**M5 — Exact interaction maps.** Decompose scores/MLP-hidden over stream pairs (exact by the
residual identity); measure energy per pair over sampled causal pairs. This is the *searchlight*:
it found the short bottom window, the attn5 hub, and the diffuse top MLPs — and told us where
interventions would be decisive before spending them.

**M6 — Windowed code propagation (the flagship).** At every layer, each *read* of the residual
(QK, v, MLP input) sees
$$x^{\text{read}}_i = \underbrace{e\text{-path}(t_i)}_{\text{exact}}
+ \sum_{\text{streams older than } W} \bar s(t_i)
+ \sum_{\text{streams within } W} s_i^{\text{live}},$$
with the old-stream tables $\lambda$-rescaled analytically. Error chains are bounded at depth
$W$ instead of $L$. This is the composition law that works: untrained, $W{=}6$ gives +0.059
(bilin18, qk+v+bottom-MLP reads) and +0.030 (sqrd12, ALL reads). vq1024 on the tables is free
(≈50× table compression); CE-polishing the values buys nothing — *the discrete structure is the
description*.

**M7 — Rank-$k$ with live coefficients.** For a contextual object, subtract the token mean,
PCA the deviations, and replace the object by $\bar z(t_i) + \sum_{j\le k} (d_j\!\cdot\!\delta_i)\,d_j$
with **live** projections. A structural claim (the function factors through $k$ scalars), not a
compute saving. Delivered: H7 is rank-1 (+0.0001), mlp16 is rank ~4–16.

**M8 — Joint CE/KL value-training.** Freeze the discrete structure, train the continuous table
values through the frozen model. Works where composition error is co-adaptation (layer-0 grand:
+0.455 → **−0.019**); does nothing where the structure is already right (windowed tables). The
data ladder is binding: ~1M params per 65k tokens, ~10M needs ~2M tokens. KL-to-live-model
separates faithful compression from data adaptation.

**M9 — Audit discipline.** ΔCE at $T{=}512$ is binding; pattern-MSE only steers search loops.
Every treatment ships with: a zero control (is the component load-bearing at all?), a composed
audit (marginals systematically lie — superadditive everywhere except qk+v windowing), held-out
and cross-region checks (region drift is real), and trust-region reverts for any iterative
refinement (mass moves under first-order scores reliably backfire).

## 3. The composition laws (the program's core lessons)

1. **Marginals don't compose.** Heads, blocks, layers, deletions, live/tabled mixtures —
   composed cost exceeded summed marginals in every family (up to 10×), except:
2. **The one additive exception**: qk-read + v-read windowing (+0.112 ≈ 0.094 + 0.019) —
   when errors are *independent filtered noise*, composition is additive.
3. **Selection tolerates classes; carriage needs identity; both are mostly 0th-order in
   context** — the residue is a short live window plus two heads plus the top MLPs.
4. **Denoising beats optimizing**: quantization/estimation error acts as noise the stack
   filters (vq1024 free, H5 improves when denoised, all backward metrics null).
5. **Compressibility is a property of the (model, decomposition-family) pair** — the two
   models' ranking inverts between score-space and input-space families.

## 4. Bits accounting (frozen convention)

Structural bits: floats at 32 b (frozen convention) + $V\log_2 k$ per assignment. Estimation
data: token counts reported **beside** the bits, never converted. Flagship configuration
(windowed-D $W{=}4$, vq1024 tables): $34 \times (1024{\cdot}1024\ \text{floats} + V{\cdot}10\ \text{bits})$
≈ 0.145 GB structural + 524k estimation tokens, at +0.094 (cross-region +0.089).

## 5. Where everything lives

Results narrative: files 01–14 in this directory (README has the index + headline numbers).
Per-experiment methods with code snippets: [00_methods.md](00_methods.md). Chronology and
retractions: `../LOG.md`. Binding spec: `../qk_mdl_spec.md`.

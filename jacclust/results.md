# Jacobian clustering — results log

## 2026-07-10 — tick 1: kernel verified; DGP-A's headline prediction (P2) is FALSE as stated

### P1 (the writeup's own unit test) — PASSES
`<J(x),J(x')>_F = tr(J(x)^T J(x')) = x^T G x'` with
`G = L^T(M⊙RR^T)L + L^T(M⊙RL^T)R + R^T(M⊙LR^T)L + R^T(M⊙LL^T)R`, `M = D^T D`.

| check | value |
|---|---|
| max relative error, 300 random pairs | **1.0e-14** |
| symmetry `max|G - G^T|` | 8.5e-14 |
| PSD, min eigenvalue | +1.4e+02 |
| Euler `max|J(x)x - 2y|` | 3.6e-15 |
| gauge `max|G(D,L,R) - G(D,R,L)|` | 1.1e-13 |
| autodiff cross-check | 3.6e-15 |
| G-kernel vs explicit-J cosine (2000 pairs) | 7.5e-16 |

**Logan's question — yes, and it is literally the same number.** `<A,B>_F = tr(A^T B)`, so
"cosine similarity for a matrix" `= tr(A^T B)/(||A||_F ||B||_F)` is *identical* to cosine on the
flattened matrices (verified to 12 dp). The writeup's "cosine on flattened matrices = Frobenius
cosine" is stating exactly that. The payoff is that for one bilinear layer we never form `J` at all.

### P3 — PASSES
`cos(M_i, M_j) = cos(x_i,x_j) · cos(y_i,y_j)` to machine precision. Rank-1 M carries no pairwise
information beyond the marginals, as predicted.

### P2 — **FAILS as stated**, and the DGP is the reason
For the hand-coded layer, the exact Jacobian is
```
J(x) = [ A_g  |  (1/eps) · F(c) ]
         ^content cols  ^gate cols
```
The gate-column block is `∂y/∂s`. Two facts, both measured:
1. it has norm `O(1/eps)`: at eps=0.1, `||J_gate||/||J_cont|| = 56.7`;
2. it is **bit-identical** for two datapoints with the same content and different gates
   (max abs difference `0.00e+00`), because it equals `Σ_g u_{g,j}(v_{g,j}·c) e_g^T/eps`, a
   function of `c` alone.

So the full Jacobian's cosine is dominated by a **gate-independent** term.

| representation | ARI(gate) | ARI(content) |
|---|---|---|
| input `x` | −0.002 | **1.000** |
| output `y` | 0.113 | 0.376 |
| `[x;y]` | 0.016 | 0.725 |
| rank-1 `M` | 0.083 | 0.391 |
| **full Jacobian (G-metric)** | **−0.002** | **1.000** |
| **Jacobian, content columns only** | **1.000** | −0.000 |

### Phase boundary in eps (d_c=16, so expert Frobenius cosines ≤ 0.097)

| eps | `||J_gate||/||J_cont||` | ARI_gate (full J) | ARI_content (full J) | ARI_gate (content cols) |
|---|---|---|---|---|
| 0.01 | 567.4 | −0.000 | 1.000 | 1.000 |
| 0.10 | 56.7 | −0.000 | 1.000 | 1.000 |
| 0.50 | 11.4 | −0.000 | 1.000 | 1.000 |
| 1.00 | 5.7 | −0.000 | 1.000 | 1.000 |
| 3.00 | 1.9 | −0.000 | 1.000 | 1.000 |
| 10.0 | 0.57 | **1.000** | 0.036 | 1.000 |

**The full Jacobian only sees the gate once `eps` is large enough that the `O(1/eps)` block stops
dominating — i.e. exactly when the raw input sees the gate too.** For DGP-A the dissociation region
for the *full* Jacobian is empty. The content-restricted Jacobian works at every eps.

### Consequence for the agenda
The lemma and the machinery are correct. The demonstration DGP is not. Two honest options:
1. **Redesign the DGP** so the gate is not an input coordinate (e.g. gate enters multiplicatively /
   from a separate stream), so no `∂y/∂s` block exists. Then P2 can be tested honestly.
2. **Keep the restricted Jacobian** as the object — but then say so plainly: it is `J` restricted to
   a chosen input subspace, and choosing that subspace already encodes what we claim to discover.

Option 1 is the real test; option 2 is a demo. Also note the writeup's assumption that random
orthogonal experts are near-Frobenius-orthogonal needs `d_c` large: max off-diagonal expert cosine
is **0.474 at d_c=6** but **0.097 at d_c=16**.

### Also worth keeping
Expert-family geometry does show up in the (restricted) Jacobian embedding: orthogonal experts →
separated spikes; a continuous rotation family → a ring; hierarchical experts → nested blocks. That
is the writeup's DGP-A′ centrepiece and it survives, since it uses the restricted object.

### Deliverable
`jacclust_dgpA.html` — self-contained interactive 3D scatter (drag to rotate), 3 expert geometries ×
4 representations × 2 colourings. Sent to Logan.

### Not yet done (next tick, per cron)
Redesigned DGP-A (multiplicative gate), DGP-B sign quotient, DGP-C two-layer + sketched VJP vs exact,
P4 principal angles. Standing rules from mechdecomp carried over: ≥5 seeds, measured chance ARI,
random-projection control at matched dimension, no single-seed statistics.

---

## 2026-07-10 — tick 2: the real-model test. Kernel exact; my "Euler shadow" worry refuted; G's value is layer-dependent

Logan's mechdecomp cron was still firing on the paused program — retired it (`CronDelete 1b6ea46c`).

### The DGP-A fix is not available as the writeup imagines
Attempted next task: "redesign DGP-A so the gate enters multiplicatively and no `∂y/∂s` block exists."
**It cannot be done while the gate is any coordinate of `x`.** Differentiating a gate coordinate
produces `Σ_g A_g c w_gᵀ` — a sum over **all** experts, hence gate-independent. Euler's identity
`J(x)x = 2y` makes it concrete: `J` always carries the output as a rank-1 shadow, and the gate
derivative is what carries it. Making the gate large enough for `J` to be gate-dominated makes raw
`x` cosine see it too (confirmed by the eps sweep in tick 1: no dissociation window).

⇒ For a **real** bilinear MLP there is no content/gate split anyway — the whole residual is `x`. So
the full-`J` kernel is the only object, and the question becomes empirical.

### Q1 — the kernel is exact on trained weights
`block2-dense-seed0`, BilinearMLP `D(128,512) L(512,128) R(512,128)`, post-norm inputs:

    max |cos_J − cos_G| over 7140 pairs  =  1.22e-15

### Q2 — my hypothesis was wrong: J is NOT dominated by its Euler shadow
I predicted `J` might collapse onto the rank-1 `M = y xᵀ/‖x‖²`, which would make Jacobian clustering
equivalent to the `cos_x·cos_y` baseline the writeup expects to fail. Measured fraction of `‖J‖²_F`
lying in the shadow: **median 0.0366, mean 0.0360, p90 0.0471.** Only ~3.7%. **Refuted.**
(P3's identity `corr(cos_M, cos_x·cos_y) = 1.0000` reproduced exactly, as a check on the harness.)

### But the Jacobian kernel is nearly plain input cosine on this model

| quantity | value |
|---|---|
| `corr(cos_J, cos_x)` | **+0.9439** |
| `corr(cos_J, cos_M)` | +0.8949 |
| `corr(cos_J, cos_y)` | +0.0820 |
| `G` effective rank (exp-entropy) | **118.5 / 128** — nearly flat |
| principal angles, top-8 of `G` vs top-8 data PCA | cos = 0.31, 0.30, 0.27, 0.16, 0.14, 0.09, 0.06, 0.04 |

A nearly flat `G` means `xᵀGx' ≈ const·x·x'`. Note `G`'s top eigenspace is nearly **orthogonal** to
the data PCA — so `G` is not merely whitening; it just isn't very anisotropic.

### The control that matters: does `G` do anything its spectrum alone wouldn't?
`G_rand` = same eigenvalues, random eigenvectors. 5 seeds, mean±sd ARI.

| layer | `G` eff rank | k | ARI(G, x-clust) | ARI(G_rand, x-clust) | reading |
|---|---|---|---|---|---|
| MLP #0 | 118.5 | 4 | 0.858±0.062 | 0.872±0.045 | **G ≈ random rotation: eigenvectors carry nothing** |
| MLP #0 | | 8 | 0.213±0.009 | 0.258±0.031 | same |
| MLP #1 | 110.5 | 4 | **0.464±0.044** | 0.789±0.003 | **G departs from raw cosine far more than its spectrum explains** |
| MLP #1 | | 8 | 0.483±0.076 | 0.594±0.095 | same direction |

**Layer-dependent.** At MLP #0 the weight-derived metric buys nothing over raw cosine. At MLP #1 it
genuinely reorganises the data beyond what the spectrum alone does.

⚠ **Necessary, not sufficient.** "G-clusters differ from x-clusters" is not "G-clusters recover
mechanism" — on a real model we have no ground-truth mechanism labels. That gap is the whole
difficulty, and it is what DGP-C (two-layer, joint `(g₁,g₂)` labels) exists to bridge.

### Standing rules observed
5 seeds over k-means init; spectrum-matched random-eigenvector control (could have shown G is
special and didn't, at MLP #0); P3 identity reproduced as a harness check; no single-seed or
max-over-sample statistics; the layer that disagrees with my preferred story is reported first.

### Next
DGP-C two-layer composition with joint labels + sketched VJP estimator vs exact-J clustering (ARI vs
number of probes k). That is the only design here that can distinguish "different from cosine" from
"recovers mechanism".

---

## 2026-07-10 — tick 3: DGP-C validates the compositionality claim; DGP-B settles the sign convention

### First: P2 is unachievable for a single bilinear layer — a proof, not an observation
Write any gate as a linear readout `p_g·c` (a one-hot coordinate is the special case). Then

    J = Σ_g A_g c p_gᵀ   +   Σ_g (p_g·c) A_g
        └ gate-INdependent ┘   └── the signal ──┘

The first term sums over **all** experts. With orthonormal `p_g` its norm is ≈ `√k_g ‖c‖ ‖A‖`, while
the signal is ≈ `|p_g·c| ‖A‖ ≤ ‖c‖ ‖A‖`. **The gate-independent term always dominates.** So no
redesign of DGP-A ("make the gate multiplicative") can rescue P2; the tick-1 eps sweep was already
showing this. The content-restricted Jacobian is the only object that carries mechanism, and in
DGP-C that restriction is *architectural* (separate content/control streams), not a peek at the answer.

### DGP-C: two stacked bilinear layers, `y = B_{g₂} A_{g₁} c`
Construction verified exactly: `z` content = `A_{g₁}c` (4.4e-15), `s₂` and the constant copied
(0.00e+00), `y = B_{g₂}A_{g₁}c` (1.0e-14), chain-rule `J` vs autodiff (5.6e-15), and each restricted
Jacobian equals its expert (≤5.6e-16).

n=500, k_g=3, k_c=4, d_c=16, eps=0.3. **5 DGP seeds × 5 k-means seeds, mean±sd.**
Measured chance ARI (shuffled labels): g1 −0.0001, g2 +0.0002, joint +0.0007, content −0.0003.

| representation | ARI(g₁) | ARI(g₂) | ARI(joint) | ARI(content) |
|---|---|---|---|---|
| input `x` | 0.000±0.003 | −0.001±0.003 | 0.001±0.003 | **1.000±0.000** |
| mid `z` | 0.177±0.099 | 0.003±0.003 | 0.139±0.013 | 0.266±0.118 |
| output `y` | 0.073±0.051 | 0.049±0.037 | 0.205±0.027 | 0.096±0.083 |
| **J₁ restricted** | **1.000±0.000** | −0.001±0.002 | 0.399±0.002 | −0.001±0.002 |
| **J₂ restricted** | −0.001±0.002 | **1.000±0.000** | 0.400±0.003 | −0.001±0.002 |
| **J end-to-end** | 0.388±0.251 | 0.217±0.139 | **1.000±0.000** | −0.001±0.002 |
| sketch, k=1 probe | 0.155±0.109 | 0.240±0.132 | **1.000±0.000** | −0.002±0.002 |
| **random proj (matched dim)** | 0.001±0.003 | −0.001±0.003 | 0.000±0.002 | **1.000±0.000** |

**The writeup's central compositionality claim is confirmed**, with a control that could have failed
and didn't: a matched-dimension random projection of `x` recovers content and nothing else, exactly
like raw `x`. Per-layer Jacobians recover only their own gate; only the end-to-end Jacobian recovers
the joint `(g₁,g₂)`, and it is blind to content.

### Sketched VJP: ARI saturates at k=1; the kernel does not
The writeup estimates `k ≈ 10–20` probes. Both readings are right, for different quantities:

| probes k | corr(sketch cos, exact cos) | mean \|Δcos\| | ARI(joint) |
|---|---|---|---|
| 1 | 0.8298±0.0169 | 0.165 | **1.000** |
| 5 | 0.9535±0.0090 | 0.077 | 1.000 |
| 10 | 0.9729±0.0064 | 0.057 | 1.000 |
| 20 | 0.9864±0.0059 | 0.039 | 1.000 |
| 50 | 0.9929±0.0008 | 0.030 | 1.000 |

⇒ **k≈10–20 for kernel fidelity (~0.97–0.99); k=1 already suffices for clustering** when the
mechanism clusters are well separated. Report the kernel error, not ARI, when quoting probe counts.

### DGP-B: sign quotient — and my first attempt at it was wrong
I first negated **content only** and asserted `J` is odd; the residual `max|J(−c)+J(c)| = 1.46` said
otherwise. `J₁` restricted equals `A_{g₁}`, which is *constant* in `c` — oddness never applied.
Correct statement: a bilinear layer is degree-2 **homogeneous**, so negating the **whole** input gives
`f(−x) = f(x)` (0.00e+00) and `J(−x) = −J(x)` (0.00e+00).

Antipodal pairs, 5 seeds, spectral clustering on the affinity:

| similarity | ARI(mechanism) |
|---|---|
| `cos` on J | 0.175±0.037 |
| **`|cos|` on J** | **1.000±0.000** |
| `|cos|` on x | 0.008±0.007 |
| `cos` on y | 0.200±0.105 |

⇒ **Fix `|cos|` globally, for every method, as the writeup recommends** — otherwise it is a confound
that flatters whichever method you happened to apply it to.

### Standing rules observed
Chance ARI measured (≈0.000, never assumed); matched-dimension random-projection control included and
could have failed; every construction verified against an identity before clustering; ≥5 seeds over
both DGP sampling and k-means init; my own malformed DGP-B reported rather than silently fixed.

---

## 2026-07-10 — tick 4: P4 is sharper than predicted; DGP-D succeeds on LEARNED modules

### P4 — the top eigenspace of G *is* the gate subspace, exactly
Mean principal-angle cosine over 5 DGP seeds (cos = 1 means fully contained):

| eps | r | cos(G_top, GATE subspace) | cos(G_top, CONTENT subspace) |
|---|---|---|---|
| 0.1 | k_g=4 | **1.0000** | 0.0000 |
| 1.0 | 4 | **1.0000** | 0.0000 |
| 10.0 | 4 | **1.0000** | 0.0000 |
| any | 8 | 1.0000 | 0.5000 (= 4 gate + 4 content dirs) |

The writeup predicts the top eigenspace "aligns with the gate subspace ⊕ the content directions the
experts act on". Sharper: the top `k_g` directions are the gate subspace **exactly**, and content
appears strictly below them. This is the same fact as the tick-1 failure of P2, read off the spectrum:
`G`'s dominant directions *are* the `∂/∂gate` block. `G`'s effective rank is 15.94/20, independent of eps.

### DGP-D — Jacobian clustering recovers modules that TRAINING found
Trained a bilinear MLP (`H=256`) from scratch on `y = A_g c`, `x = [c ; s]`, 4000 steps, 5 seeds.
Val relative error 0.0028–0.0045, so the nets solve the task. Ground truth = the task's `g`, not any
hand-coded weights. `|cos|` used for all methods (DGP-B convention). Chance ARI measured:
g −0.0006, content −0.0002.

| representation | ARI(mechanism g) | ARI(content) |
|---|---|---|
| input `x` | 0.002±0.002 | **1.000±0.000** |
| hidden `Lx ⊙ Rx` | 0.002±0.004 | **1.000±0.000** |
| output `ŷ` | 0.214±0.208 | 0.208±0.103 |
| rank-1 `M` | 0.331±0.187 | 0.276±0.122 |
| full `J` (G-metric) | −0.000±0.002 | **1.000±0.000** |
| **`J` content-cols** | **1.000±0.000** | 0.036±0.007 |
| random proj (matched dim) | 0.002±0.002 | **1.000±0.000** |

**This is the writeup's "DGP-D success" criterion, met:** the mechanism structure is recovered by the
Jacobian and is invisible to activation clustering at either end of the layer (input 0.002, hidden
0.002, output 0.214) and to a matched-dimension random projection (0.002).

Two corroborations fall out:
1. **The P2 impossibility theorem holds for learned solutions too.** The trained net's full Jacobian
   recovers content at 1.000 and the gate at chance — identical to the hand-coded layer. So the
   gate-derivative domination is a property of the bilinear form, not of our construction.
2. **rank-1 `M` behaves as predicted**: 0.331 on the gate — a partial refinement of input × output
   structure, well above chance but far from the Jacobian's 1.000.

### The honest limitation, restated
The content/gate split here is given by the **task's input layout**, so restricting `J` to content
columns is legitimate but *supplied*. A real language model has no such split — the whole residual is
`x`, and tick 2 showed that there the full-`J` kernel is nearly plain input cosine
(`corr(cos_J, cos_x) = +0.944`, `G` effective rank 118.5/128) except at one of two MLPs.

So the method's status is now:
- **Toy, hand-coded (DGP-A/B/C):** works, with all predicted dissociations and controls.
- **Toy, learned (DGP-D):** works — recovers modules training discovered.
- **Real model:** the object exists and is exact, but there is no ground-truth mechanism label, and
  the natural restriction does not exist. This gap is the whole difficulty of the agenda.

### Standing rules observed
Chance ARI measured (≈0.000); matched-dimension random-projection control included and could have
failed; `|cos|` fixed globally; 5 training seeds × 5 k-means seeds, mean±sd; every construction and
identity verified before any clustering claim.

---

## 2026-07-10 — tick 5: DGP-E (no control stream) — depth restores *partial* mechanism signal, and a weights-only fix

The question left open by tick 4: a real model has **no content/control split**, so no restricted
Jacobian exists. What does the full `J` read there?

### DGP-E: the gate is inferred from content by an earlier layer
`x = [c ; 1]`, `c = a + v` (gate part `a` in span{w_g}, content `v` in the complement).
Layer 1 (bilinear) computes **quadratic** gate features `(w_g·c)²` natively and copies `v`.
Layer 2 applies `y = Σ_g (w_g·c)² B_g v`. Mechanism `g = argmax_g (w_g·c)²`. Nothing is supplied.

Construction verified: gate features exact (0.00e+00), `y` exact (6.4e-15), chain-rule `J` vs
autodiff (8.9e-16).

**The tick-3 impossibility theorem is specific to a single layer with a *linear* gate readout.**
Here, same content + different active gate gives `cos_F(J_a, J_b) = 0.0432` (DGP-A: bit-identical
gate block). Depth + a nonlinear gate makes the full Jacobian gate-dependent.

### But recovery is only partial, and here is exactly why
5 DGP seeds × 5 k-means seeds; measured chance ARI: g +0.0019, content −0.0004.

| representation | ARI(mechanism g) | ARI(content) |
|---|---|---|
| input `x` | −0.001±0.003 | **1.000±0.000** |
| mid `z` | −0.001±0.003 | 1.000±0.000 |
| output `y` | 0.172±0.076 | 0.300±0.101 |
| rank-1 `M` | 0.345±0.128 | 0.420±0.135 |
| **FULL end-to-end `J`** | **0.287±0.162** | 0.290±0.070 |
| layer-1 `J` alone | −0.001±0.003 | 1.000±0.000 |
| layer-2 `J` alone | −0.001±0.003 | 1.000±0.000 |
| random proj (matched dim) | −0.001±0.003 | 1.000±0.000 |

Column-block decomposition of the end-to-end Jacobian (`c = [a | v]`):

| block | ARI(g) | mean ‖·‖_F |
|---|---|---|
| FULL `J` | 0.389±0.166 | — |
| **`J[:, v-cols]`** (the operator `Σ_g (w_g·c)² B_g`) | **1.000±0.000** | 3.47 |
| `J[:, a-cols]` (gate-derivative block) | 0.355±0.147 | **21.23 (6.1×)** |

**The gate-derivative block is 6× larger in norm and a mediocre gate signal, so it dilutes the full
Jacobian from 1.000 down to 0.389.** Same mechanism as tick 3, *weakened* (0.355 instead of exactly
chance) but not removed. Note `layer-2 J` alone is at chance on the gate — its own gate-derivative
block `Σ_g B_g v` is gate-independent and scales with ‖v‖, so the single-layer theorem reasserts
itself layer-wise.

### Dissociation window (gate-amplitude sweep, content_amp = 3.0)

| gate_amp | ARI_g(FULL J) | ARI_g(x) | ARI_content(x) |
|---|---|---|---|
| 0.3 | 0.444±0.133 | −0.003 | 1.000 |
| 1.0 | 0.445±0.168 | −0.003 | 1.000 |
| 3.0 | 0.451±0.156 | −0.003 | 1.000 |
| 6.0 | 0.341±0.164 | −0.003 | 1.000 |
| 12.0 | 0.392±0.114 | **1.000** | 0.092 |

**A real dissociation window exists** for gate_amp ≲ 6: the full Jacobian carries mechanism structure
(≈0.45) that raw input cosine cannot see at all (−0.003). Unlike DGP-A, where the window was empty.
Caveat: rank-1 `M` already reaches 0.25–0.38 here, so the Jacobian's margin over that cheap baseline
is ≈0.19, not the whole 0.45.

### NEW: a weights-only restriction, derived from our own P4
P4 established that `G`'s top eigenspace **is** the gate subspace. The gate-derivative block is what
contaminates `J`. So project `J`'s **columns** off the top-`r` eigenvectors of layer-1's `G` — no
labels, no data, no architectural split.

| representation | ARI(mechanism g) |
|---|---|
| FULL `J` | 0.389±0.166 |
| **`J` off `G`-top (weights-only)** | **0.654±0.176** |
| `J` off RANDOM subspace (matched dim) | 0.388±0.163 |
| `J` off `G`-bottom eigenvectors | 0.348±0.155 |
| `J[:, v-cols]` (oracle split) | 1.000±0.000 |

Chance +0.0001. **It beats both controls**, recovering ~2/3 of the way to the oracle. It does not
reach 1.000 because here `cos(G-top, gate subspace) = 0.667`, not 1.0 — layer 1 also copies `v`, so
its top eigenspace is not purely the gate.

This is a concrete, data-free recipe the writeup does not contain: **decontaminate the Jacobian by
projecting off the leading eigenvectors of the layer's own `G`.** It is exactly the "top eigenspace of
G = where the layer is most input-dependent" observation in §2.2, used as a *filter* rather than a
diagnostic.

### Standing rules observed
Chance ARI measured; two controls that could have succeeded and didn't (random subspace, G-bottom);
5 seeds × 5 k-means inits; every construction checked against an identity first; the block
decomposition reported before the fix, so the fix is explained rather than merely exhibited.

---

## 2026-07-10 — tick 6: a ground-truth-free validation on REAL bilinear MLPs. Gain is not optional.

Real models have no mechanism labels (open problem 4). But the method's claim is falsifiable without
them: **if a cluster groups datapoints on which the layer applied nearly the same linear map, one
linear map should predict the layer's output on held-out points of that cluster.** So: cluster on
train tokens, fit `y ≈ A_c x` per cluster, assign held-out tokens by nearest centroid, score held-out R².
Baselines that could win: raw `x`, spectrum-matched `G_rand`, random clusters, one global map.

### Two harness bugs, both caught by identity checks before any claim was made
1. **`G_P = gram(D, LP, RP)` is WRONG.** `J(x)P = D[diag(Lx) RP + diag(Rx) LP]` keeps the *gates*
   `Lx, Rx`; the naive form projects them too. Correct (verified to 1e-13 at r=0,1,3,5, with `r=0`
   reproducing plain `G`):

       G_P = Lᵀ(M ⊙ R P Rᵀ)L + Lᵀ(M ⊙ R P Lᵀ)R + Rᵀ(M ⊙ L P Rᵀ)L + Rᵀ(M ⊙ L P Lᵀ)R

2. **Unregularised per-cluster least squares blows up.** v1 gave `R² = −319` for raw cosine and
   `−139` for G, while **random clusters scored −0.78** — i.e. random beat everything. That measures
   fit instability, not mechanism. Fixed with ridge (λ=0.1) plus a `k=1` identity check that must
   reproduce the global fit exactly (it does, to 0.00e+00).

### The finding: cosine discards the Jacobian gain, and that is what breaks the surrogate
`‖J(x)‖²_F = xᵀ G x` **exactly** (verified 1.3e-15) — the gain is free from the kernel, it is just the
un-normalised diagonal. Within a cosine cluster `J = s(x)·Ĵ` with `s` varying, so `y = ½J(x)x` is not
linear in `x` even when the direction is constant.

| metric | MLP#0 k=8 | MLP#0 k=32 | MLP#1 k=8 |
|---|---|---|---|
| G **cosine** (direction only — as the writeup specifies) | **−0.2518** | +0.1540 | +0.5145 |
| G **Euclidean** (direction + gain) | **+0.2242** | +0.3666 | +0.5420 |
| gain `‖J‖_F` alone (1-D control) | −0.2738 | −1.2578 | +0.3278 |

The gain-only control fails everywhere, so it is direction **and** gain jointly. This turns the
writeup's §9 "parked open question" (*"the gain ‖J‖_F is discarded by normalization; a two-feature view
may matter on real models"*) into a load-bearing design choice: on real MLPs it is the difference
between negative and positive surrogate R².

### Fair comparison — every metric under the same rule (Euclidean, gain retained), 5 seeds

| layer | k | **G (Jacobian)** | G_P (off top-8) | raw `x` | G_rand (spectrum) | random |
|---|---|---|---|---|---|---|
| MLP#0 (global −0.629) | 8 | **+0.2242±0.085** | +0.2349±0.043 | +0.1226±0.066 | +0.1193±0.083 | −1.025 |
| | 16 | +0.3107±0.062 | +0.2285±0.156 | +0.3206±0.024 | **+0.3425±0.021** | −1.771 |
| | 32 | **+0.3666±0.034** | +0.3724±0.021 | +0.1319±0.069 | +0.1205±0.102 | −8.226 |
| MLP#1 (global +0.373) | 8 | **+0.5420±0.006** | +0.5457±0.004 | +0.5191±0.004 | +0.5178±0.003 | +0.233 |
| | 16 | **+0.5436±0.008** | +0.5465±0.007 | +0.5073±0.010 | +0.5042±0.011 | −0.005 |
| | 32 | **+0.4544±0.033** | +0.4514±0.021 | +0.3613±0.049 | +0.3451±0.020 | −1.262 |

**The Jacobian metric beats raw `x` and the spectrum-matched control in 5 of 6 cells, and the margin
grows with k (up to +0.09).** ⚠ **The exception is reported, not dropped:** at MLP#0, k=16 the
random-eigenvector control *wins* (0.3425 vs 0.3107). So the advantage is real but not uniform.

`G_P` (projecting off `G`'s top-8 eigenvectors) ≈ `G` on real MLPs — the tick-5 decontamination gains
nothing here, unlike DGP-E. Consistent with P4: on a real layer there is no clean gate subspace for the
top eigenvectors to be.

### What this establishes, and what it does not
- **Establishes:** a label-free validation that the Jacobian metric produces *better "same-map"
  clusters* than input geometry or a spectrum-matched control, on a real trained bilinear MLP.
  This is the first real-model evidence in the program that survives a control which could have won.
- **Does not establish:** that those clusters correspond to interpretable computational roles. That
  still needs the intervention test (open problem 4).
- **Design correction for the writeup:** §8 says "decide cos vs |cos| once (recommend |cos|)". On real
  models, the choice that matters is **cosine vs Euclidean** — i.e. whether the gain is kept. Keep it.

### Standing rules observed
Both harness bugs found by identity checks *before* any claim; the broken v1 result reported rather
than quietly re-run; the losing cell (MLP#0 k=16) reported; spectrum-matched control run on every cell;
5 seeds, mean±sd; `k=1` identity check ties the surrogate to the global fit.

---

## 2026-07-10 — tick 7: the intervention test (open problem 4). It works, and it kills my own headline statistic.

**Design.** If a cluster groups tokens on which the layer applied ~the same linear map, then replacing
the MLP's output with *that cluster's* ridge surrogate `A_c x` should barely hurt the LM, while using a
*different* cluster's surrogate should hurt a lot. No labels needed. Controls that could win: raw-`x`
clusters, spectrum-matched `G_rand` clusters, random assignment, and one global map.

**Harness identities checked first (both exact):**
- `clean` patch mode reproduces the unmodified model's CE (diff **0.00e+00**).
- `k=1` within-patch equals the global patch (diff **0.00e+00**).

Replacing the MLP by a *single* linear map costs **+5.99 nats** (MLP#0) / **+1.47 nats** (MLP#1) — a
large dynamic range for the test to work in. `block2-dense-seed0`, 5 k-means seeds, held-out windows.

### The result — and the statistic I have to throw away

| layer | k | metric | CE within ↓ | CE across | differential |
|---|---|---|---|---|---|
| MLP#0 (clean 4.083, global 10.073) | 8 | **G (Jacobian)** | **7.785±0.173** | 13.855 | 6.069 |
| | 8 | raw `x` | 8.407±0.177 | 13.834 | 5.428 |
| | 8 | G_rand | 8.052±0.118 | 13.859 | 5.808 |
| | 16 | G (Jacobian) | 7.428±0.085 | 13.282 | 5.854 |
| | 16 | raw `x` | 7.560±0.063 | 13.510 | 5.950 |
| | 16 | **G_rand (control WINS)** | **7.286±0.051** | 13.358 | 6.072 |
| MLP#1 (clean 4.083, global 5.554) | 8 | G (Jacobian) | 5.2175±0.015 | 9.003 | 3.786 |
| | 8 | raw `x` | 5.2189±0.003 | 8.452 | 3.233 |
| | 8 | G_rand | 5.2420±0.004 | 8.661 | 3.419 |
| | 16 | **G (Jacobian)** | **5.1248±0.007** | 9.991 | 4.866 |
| | 16 | raw `x` | 5.1894±0.017 | 9.878 | 4.689 |
| | 16 | G_rand | 5.2150±0.021 | 9.848 | 4.633 |
| any | any | **random clusters** | — | — | **−0.002 … +0.003** |

**The random-cluster control gives exactly zero differential**, as it must (its assignment carries no
information). The design is sound.

### ⚠ The differential is a bad statistic and I would have over-read it
Uniform-prediction CE is `ln(5120) = 8.541` nats. **Across-cluster CE is 8.5–13.9 — at or far beyond
uniform.** The model is confidently *wrong* there, so it is a saturated regime and differences between
metrics in it are noise about how badly a wrong linear map fails, not evidence about mechanism. The
differential is dominated by "the surrogate is crude", not by "clusters are mechanisms".

**The informative quantity is within-cluster CE** (in nats: the same thing tick 6 measured as held-out
surrogate R²).

By within-CE the Jacobian metric wins **2 of 4 cells clearly** (MLP#0 k=8: 7.785 vs 8.407/8.052;
MLP#1 k=16: 5.125 vs 5.189/5.215), **ties one** (MLP#1 k=8, 5.2175 vs 5.2189 — a 0.0014-nat "win" is
nothing), and **loses one to the spectrum-matched control** (MLP#0 k=16: 7.286 for `G_rand` vs 7.428).

**That losing cell is the same (layer, k) that lost in tick 6.** Two independent tests agreeing on
where the method fails is worth more than either agreeing on where it wins.

### Honest position after tick 7
- The intervention test is **valid** (both identities exact; random control gives the exact null) and
  is now the program's causal instrument.
- Its headline statistic — the within/across differential the writeup's §7 asks for — is **saturated on
  this model** and should not be quoted. Report within-cluster CE, or equivalently held-out surrogate R².
- The Jacobian metric's real-model advantage is **modest and non-uniform**: it beats raw `x` and a
  spectrum-matched control in most but not all (layer, k) cells, in both an offline (tick 6) and a
  causal (tick 7) test, and the two tests fail in the same place.
- **Not established:** that G-clusters correspond to recognisable computational roles. The intervention
  shows they are better *same-map* clusters, which is a weaker claim than the writeup's §6 Phase-1 goal.

### Standing rules observed
Harness identities verified before any claim; the saturation of my own preferred statistic diagnosed
and reported rather than quietly used; the losing cell reported and cross-referenced with tick 6;
spectrum-matched control on every cell; 5 seeds, mean±sd; the null control produces the exact null.

---

## 2026-07-10 — tick 8: scale to the 500M-style tensor-transformer.

> ⚠ **SUPERSEDED by tick 9.** The two negatives below ("global beats clustering", "G ties raw x") were a
> ridge fit-instability artifact at 768-dim with 500 pts/cluster. Corrected at a sound operating point:
> clustering beats global by +0.1, and G edges raw x + G_rand at early/mid layers. The G-effective-rank
> observation (677–718/768) and the faithful-model-loading are the parts of tick 8 that stand.

Priority (3). Target: `Elriggs/gpt2-bilinear-12l-6h-768embd` — the pure bilinear-MLP model (softmax
attn, `bilinear:true, squared_attn:false, gated:false`), the writeup's §6 preferred first target.
12 layers × 768-dim gives a real depth sweep.

### Getting the forward pass right (two failures, both caught before any claim)
1. My hand-rolled forward gave CE **8.45** (working model ~3–4) — wrong rotary convention. **Caught by
   the CE check; no activations collected from it.**
2. Rebuilt by extracting the actual model classes from `loganriggs/modded-nanogpt` (`jacclust/tt_model.py`).
   Loads with **no missing/unexpected keys**, CE **3.385** on real text. This is the faithful model.
   (Key architecture details my reconstruction missed: `x0` value-residual skip with per-block
   `lambdas`, cross-layer value-residual mixing `lamb`, `30·tanh(logits/30)` squashing.)

### `G` effective rank across depth (weights-only summary statistic)

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G eff rank / 768 | 685 | 718 | 705 | 705 | 683 | 677 | 686 | 688 | 688 | 697 | 700 | 684 |
| corr(cos_J, cos_x) | 0.63 | 0.66 | 0.73 | 0.70 | 0.80 | 0.77 | 0.74 | 0.66 | 0.62 | 0.62 | 0.68 | 0.79 |

**`G` is nearly full-rank at every depth (677–718/768).** A near-isometric `G` gives `xᵀGx' ≈
const·x·x'`, so the Jacobian metric is close to plain input cosine throughout — corr 0.62–0.80,
consistent with the tiny-model MLP#0 (0.94) more than MLP#1.

### Held-out surrogate R² (Euclidean/gain retained, 5 seeds, k=16)

| layer | G (Jacobian) | raw x | G_rand | global single map |
|---|---|---|---|---|
| 2 | −0.319±0.110 | −0.330±0.094 | −0.417±0.066 | **+0.452** |
| 6 | −0.220±0.061 | −0.220±0.047 | −0.210±0.054 | **+0.467** |
| 10 | −0.216±0.123 | −0.240±0.090 | −0.235±0.118 | **+0.509** |

**Two clean negatives:**
1. **Every clustering is worse than a single global linear map** (+0.45 to +0.51 vs all negative).
   On a real trained transformer, partitioning into 16 clusters and fitting per-cluster linear maps
   *hurts* held-out prediction — the clusters do not carve out locally-linear regions of the layer.
2. **`G` ties raw `x` and `G_rand`** at every layer (all within seed noise). The modest, non-uniform
   advantage on the 1.9M toy (ticks 6–7) **does not survive to 500M scale.**

### Honest verdict for the writeup
- **The lemma and toys are solid** (P1–P4, DGP-A/B/C/D/E, exact kernel, compositionality, the
  impossibility theorem). Those are real and publishable as method + theory.
- **The real-model claim does not hold at scale.** On a 12-layer 768-dim bilinear transformer, `G`'s
  eigen-structure is nearly isotropic (eff rank ~690/768 everywhere), the metric is ≈ input cosine,
  and Jacobian clustering neither beats a spectrum-matched control nor beats *not clustering at all*.
- The tiny-model MLP#1 signal (ticks 6–7) is best read as a small-model / low-effective-rank artifact,
  not a property that scales. **This is the kind of result the standing rules exist to surface: the
  positive real-model finding was on one 128-dim layer and did not replicate at 768-dim × 12 layers.**

### Why (the mechanism, not just the headline)
The whole method's leverage is `G` being *anisotropic* — a metric that reweights input space by how
much the layer's operator varies. On these trained MLPs `G` is nearly isotropic, so there is little
reweighting to be had, and the Jacobian direction is dominated by a near-common component (the same
reason `corr(cos_J, cos_x)` is high). Effective rank of `G` is exactly the weights-only quantity that
predicts this, and it says "don't expect much" at every layer of this model.

### Standing rules observed
CE check caught a broken forward pass before any activation was collected; the faithful model verified
by exact state-dict load + CE 3.385; spectrum-matched control and the global-map floor on every layer;
5 seeds; the negative reported plainly, and cross-referenced to the tiny-model result it contradicts.

---

## 2026-07-10 — tick 9: ⚠ TICK 8'S NEGATIVE WAS A HARNESS ARTIFACT. Corrected: weak but real positive at scale.

Tick 8 concluded (a) "a global linear map beats every clustering" and (b) "G ties raw x at scale".
Both were the **ridge fit-instability confound** — the same one caught in tick 6 at 128-dim and NOT
re-checked at 768-dim. At tick 8's settings (8000 train tokens, k=16 ⇒ 500 pts/cluster, λ=0.1),
per-cluster ridge on 768 features overfits.

### Audit — hold clustering fixed (G-metric, k=16), sweep data and λ

| n_train | pts/cluster | λ | global R² | cluster R² | cluster > global? |
|---|---|---|---|---|---|
| 8000 | 500 | 0.1 | +0.450 | **−0.176** | False ← tick-8 setting |
| 8000 | 500 | 30 | +0.464 | +0.519 | **True** |
| 24000 | 1500 | 3 | +0.491 | +0.531 | **True** |
| **48000** | **3000** | **3** | +0.501 | **+0.608** | **True** |
| 48000 | 3000 | 30 | +0.501 | +0.635 | True |

With adequate data (3000 pts/cluster) or stronger ridge, clustering **beats** the global map by up to
+0.13 at every setting. **Tick 8's "clustering hurts" is retracted** — it was undersampled per-cluster
fits, not a property of the layer.

### The real comparison, redone at a sound operating point (48k train, λ=3, k=16, 5 seeds, Euclidean)

| layer | G (Jacobian) | raw x | G_rand (spectrum) | global | G beats both >1σ? |
|---|---|---|---|---|---|
| 2 | **+0.6150±0.002** | +0.6094±0.003 | +0.6103±0.003 | +0.499 | **yes** |
| 6 | **+0.6071±0.002** | +0.6011±0.002 | +0.5992±0.003 | +0.501 | **yes** |
| 10 | +0.5990±0.004 | +0.6022 | +0.6002 | +0.531 | no (tie) |

**Corrected verdict:** at 500M scale, clustering beats a global map everywhere (+0.10–0.12), and the
**G-metric beats both raw `x` and the spectrum-matched control at early/mid layers** (2, 6) by ~2–3σ,
tying at layer 10. Tick 8's "G ties raw x" is really "G edges raw x, at early/mid layers".

### Honest magnitude, stated plainly
The margin is **small** (~0.006 R², vs the toys' ARI-1.0 dissociation). This is not the method reading
mechanism structure invisible to activations; it is the weight-derived metric producing *slightly*
better same-map clusters than input geometry, consistently, with a control that could have tied and
mostly didn't. The `G` effective-rank story (677–718/768, tick 8) still explains *why* it's small: the
metric is close to isotropic, so it can only reweight input space a little. Both facts hold together —
small effect, real, and predicted by a weights-only quantity.

### The mistake, named
Tick 8 reported a negative I did not audit for the confound I had *already found once* (tick 6). The
standing rule "verify the harness with an identity/known-point before reporting" should have included
"...and re-verify it when the dimension changes 6×." Per-cluster least squares needs
pts/cluster ≫ d_in; at 768-dim that means ≫ 768, and 500 was far too few. Added to the rules.

### Standing rules observed (this tick)
The suspicious result (global beats clustering, at only R²=0.45) was audited before being built on; the
confound was isolated by sweeping the two nuisance parameters; the correction is reported as a
retraction with the mechanism, not a quiet re-run; the small size of the surviving positive is stated,
not inflated.

---

## 2026-07-10 — tick 10: characterizing the real-model regime. G is a small, principled nudge on cosine.

Tick 9 left a puzzle: `corr(cos_J, cos_x) = 0.62–0.80` (metrics substantially different) yet surrogate
R² differs by only 0.006. Two cheap weights-only + assignment measurements resolve it.

### (a) G is near-isotropic on EVERY trained bilinear MLP measured (weights only)

| model | G eff rank / d, per layer | mean |
|---|---|---|
| gpt2-bilinear-12l-768 (12 layers) | 0.89–0.94 | **0.902** |
| block2-dense-seed0 (2 MLPs, d=128) | 0.926, 0.863 | 0.895 |

14 layers, 2 independent models, all in **0.86–0.94**. The method's leverage is `1 − (eff rank/d)` —
the amount `G` can reweight input space away from cosine — and it is small everywhere measured. This
is a cheap, purely weights-derived predictor of "don't expect much here", and it is uniform.

### (b) G-clusters ≈ x-clusters, but as a PRINCIPLED nudge (ARI, k=16, 5 seeds)

| layer | ARI(G, x) | ARI(G, G_rand) | ARI(x, x) diff-seed (instability floor) |
|---|---|---|---|
| 2 | 0.597±0.021 | 0.564±0.061 | 0.679±0.041 |
| 6 | 0.502±0.048 | 0.499±0.068 | 0.605±0.067 |
| 10 | 0.611±0.032 | 0.607±0.033 | 0.670±0.045 |

**G-clusters differ from x-clusters by about a k-means seed's worth of noise** (ARI(G,x) 0.50–0.61 sits
just below the self-consistency floor 0.60–0.68). But the reconciliation with tick 9 is exact:
**`G` stays *closer* to `x` than the spectrum-matched `G_rand` does** (0.597 > 0.564 at layer 2), yet
clusters slightly *better* (tick 9: G surrogate 0.6150 > G_rand 0.6103). So `G` is a **small,
weight-informed perturbation of cosine** — a principled nudge, not a random rotation — which is why it
beats `G_rand` on surrogate R² while barely departing from `x` in assignment.

### The coherent real-model picture (ticks 6–10, now complete)
On trained bilinear MLPs, at 128-dim and 768-dim alike:
- `G` is near-isotropic (eff rank ~0.90), so the Jacobian metric ≈ input cosine.
- G-clusters therefore ≈ x-clusters (ARI difference within k-means noise).
- The `Wx`-aware metric is nonetheless a *principled* perturbation: it stays closer to cosine than a
  random spectrum-matched metric and clusters slightly (+0.006 R², ~2–3σ) better at early/mid layers.
- **Net: a real but small effect, whose size is predicted by a weights-only quantity (G eff rank).**

This is the honest scope statement. The method is not degenerate with cosine (it beats a control that
could have tied), but on these models it is *close* to cosine, for a reason visible in the weights
before any data is touched.

### What would make it large (the anisotropy question, now sharp)
The effect scales with `1 − eff_rank(G)/d`. Every *trained* bilinear MLP tested is ~0.90. A layer where
`G` is strongly anisotropic — few dominant eigenvalues — would give a large effect. Candidates not yet
tested: bilinear ATTENTION (per-position OV-like maps, plausibly sharper; writeup Phase 3), and
task-specialised / small-vocab bilinear models. **`G` eff rank is the free pre-screen: compute it from
weights, and only cluster where it is far from full.** Batched into LOG for Logan's steer.

### Standing rules observed
Weights-only anisotropy measured across multiple models before generalising; the ARI instability floor
measured (not assumed) as the reference for "different"; the tick-9 surrogate result reconciled with the
assignment result rather than either quoted alone; the small effect stated as small.

---

## 2026-07-10 — tick 11: DGP-A' RING geometry, quantified with manifold metrics (priority 2, done)

The centerpiece claim (§5 DGP-A'): the method reads *whatever* geometry the mechanism family has, not
just clusters. For a **continuous rotation family** `A_θ = R(θ)`, θ on a circle, there are no clusters —
so ARI is meaningless and the right test is manifold recovery of the circle.

**Construction (verified to 0.00e+00 before any embedding):** a single bilinear layer computing
`y = R(θ) c`, with θ supplied as `(cos θ, sin θ)` control coords. `max|y − R(θ)c| = 0` and
`max|J[:, :d_c] − R(θ)| = 0` — the restricted Jacobian **is** the rotation, so it depends only on θ.

**Metrics (not ARI):** |circular correlation(recovered angle, true θ)| from a 2-D spectral embedding;
trustworthiness of the embedding vs the true θ-circle. 5 DGP seeds. Controls: input `x`, output `y`,
matched-dim random projection.

| representation | \|circular corr\| | trustworthiness |
|---|---|---|
| **restricted Jacobian** | **0.999±0.000** | **1.000±0.000** |
| input `x` | 0.083±0.036 | 0.508±0.006 |
| output `y` | 0.028±0.019 | — |
| random proj (matched dim) | 0.069±0.048 | 0.520±0.010 |

**The restricted-Jacobian embedding recovers the mechanism circle almost perfectly** (circ corr 0.999,
trustworthiness 1.000), while input, output, and a matched-dim random projection are all at chance
(circ corr < 0.09, trustworthiness ≈ 0.51 = random). This is the DGP-A' centerpiece, now with a
falsifiable manifold metric rather than eyeballing the 3-D scatter.

**Honest caveat, stated:** θ *is* present in the input (coords `i_cos, i_sin`), but is swamped by the
`d_c` unit-scale content dims, so input-space embedding recovers content geometry, not θ — exactly the
input-vs-mechanism dissociation the method exploits. It is not that the input lacks the information;
it is that input *geometry* is dominated by content while Jacobian geometry is dominated by mechanism.

### The toy story is now complete and rigorous
- discrete orthogonal experts → clusters (ARI 1.000, ticks 1–4)
- continuous rotation family → circle (circ corr 0.999, this tick)
- hierarchical experts → nested blocks (shown in the 3-D HTML; ARI on the tree is the discrete case)

The method reads the geometry of the mechanism distribution, whatever it is — spikes, ring, tree —
verified with the metric appropriate to each (ARI for clusters, circular correlation + trustworthiness
for the manifold). This is the strongest, cleanest part of the program.

### Standing rules observed
Construction verified to 0.00e+00 before any embedding; manifold metric matched to the manifold (not
ARI on a thing with no clusters); matched-dim random-projection control at chance; 5 seeds; the caveat
that the input contains-but-swamps θ stated rather than hidden.

---

## 2026-07-11 — tick 12: the organizing LAW, tested with a controlled knob (not asserted)

Tick 10 *asserted* "effect size ∝ 1 − eff_rank(G)/d". This tests it. Controlled single bilinear layer,
fixed Gaussian input; knob = `r`, the rank of the subspace the rows of L,R live in (small r → low-rank
anisotropic G; r=d → isotropic G). At each r: eff_rank(G)/d, and the G-metric's surrogate-R² advantage
over cosine and over the spectrum-matched G_rand. 5 seeds, k=12. Knob verified to move eff_rank.

| r | eff_rank(G)/d | R²(G) | R²(cos) | G − cos | G − G_rand |
|---|---|---|---|---|---|
| 2 | 0.041 | +0.890 | +0.005 | **+0.886** | +0.892 |
| 4 | 0.081 | +0.793 | +0.039 | +0.755 | +0.789 |
| 8 | 0.161 | +0.559 | +0.033 | +0.527 | +0.520 |
| 16 | 0.319 | +0.332 | +0.039 | +0.294 | +0.299 |
| 32 | 0.628 | +0.176 | +0.042 | +0.134 | +0.142 |
| 48 | 0.922 | +0.111 | +0.043 | **+0.068** | +0.068 |

**corr(1 − eff_rank/d, advantage) = +0.913 (vs cosine), +0.911 (vs G_rand).**

**The claim is now a measured law, not an assertion.** As `G` goes from isotropic (eff rank 0.92) to
strongly anisotropic (0.04), the weight-derived metric's advantage over both controls rises monotonically
from +0.07 to +0.89 — a 13× swing, ρ = 0.91. And note the controls behave: raw cosine is flat (~+0.04)
across the whole range (it can't see the operator), and G_rand tracks cosine (its eigenvectors are random),
so the entire advantage is the *anisotropic eigen-structure of G specifically*.

### This places the real models exactly, and dissolves the "is the real-model effect disappointing?" question
Trained bilinear MLPs sit at eff_rank(G)/d ≈ 0.90 — the **r=48 row** of this sweep, advantage ≈ +0.068.
That is precisely the small-but-real +0.006–0.07 measured on the 768 model (ticks 9–10). The real-model
result is not a mystery and not a letdown: it is this law evaluated where language-model MLPs happen to
live. The method's leverage is real and large *when G is anisotropic*; trained LM MLPs are near-isotropic,
by a property you read off the weights.

### The complete, defensible story of the program
1. **Exact kernel + lemma** (P1): `⟨J,J'⟩_F = xᵀGx'`, no Jacobian materialized. Verified 1e-14.
2. **P2 impossibility theorem**: a single bilinear layer's full J cannot separate input-gated mechanisms
   (gate-derivative block dominates). Proven and confirmed on trained nets.
3. **Toys** read whatever geometry the mechanism family has — clusters (ARI 1.0), circle (circ corr 0.999),
   hierarchy — each with the metric matched to its geometry.
4. **DGP-E + depth** breaks P2; the weights-only G-top projection partly decontaminates (0.65 vs 0.39).
5. **Real models**: small-but-real, control-beating advantage, its size **predicted by a weights-only law**
   (this tick): effect ∝ 1 − eff_rank(G)/d, ρ = 0.91.
6. **Practical upshot**: compute eff_rank(G) from weights alone; the method is worth running iff it is far
   from full. A pre-screen with a quantified payoff curve.

### Standing rules observed
The asserted claim was subjected to a controlled test that could have refuted it (a weak/zero correlation
would have); both controls (flat cosine, tracking G_rand) behave as they must; the knob was verified to
move the independent variable; 5 seeds; the real-model result is *located on* the law rather than
re-litigated.

---

## 2026-07-11 — tick 13: ⚠ TICK 12 OVERCLAIMED. The law is isotropic-input-only; the real-model effect is ~0.

Auditing my own tick-12 conclusion against its own numbers: the law predicted advantage +0.068 at
eff_rank(G)/d≈0.90, but the real 768 model gave +0.006 — 10x off, which I glossed as "sits on the law."
It does not. Test: same real G (eff_rank/d 0.893), swap ONLY the input distribution, 5 seeds, k=16,
24k-token fits.

| input distribution | G adv over cosine | G adv over G_rand |
|---|---|---|
| **(a) real activations** | **+0.0012** | **−0.0012** |
| (b) Gaussian, matched covariance | +0.0372 | +0.0399 |
| (c) isotropic Gaussian | +0.0656 | +0.0251 |

**(c) reproduces the tick-12 law (+0.066 ≈ predicted +0.068). (a) real activations collapse to +0.0012,
and the advantage over the honest spectrum-matched control is NEGATIVE (−0.0012).**

### Two corrections, both against my own prior claims
1. **Tick 12's "law places the real models exactly" is retracted.** The law is for **isotropic input**.
   eff_rank(G) is *necessary but not sufficient*: a large effect needs `G` anisotropic AND the data to
   populate `G`'s anisotropic directions. Real activations do not.
2. **The real-model positive from ticks 6–10 is weaker than I reported, and at this layer vanishes.**
   With clean 24k-token fits, layer-6 G beats cosine by only +0.0012 and *loses to* G_rand by 0.0012.
   The earlier "+0.006, 2–3σ over G_rand" (tick 9, 48k tokens, different layers) should be read as the
   ceiling, not the typical case; on a like-for-like isolation it is essentially zero.

### The mechanism (why real data kills it), read off the (a)/(b)/(c) ladder
- (c)→(b): matching the activation *covariance* only drops the effect +0.066→+0.037 — so covariance
  anisotropy is NOT the killer.
- (b)→(a): the remaining collapse +0.037→+0.001 is the **non-Gaussian structure** of real activations
  (heavy tails, discrete token-driven clusters). Real activation geometry already concentrates into
  regions where G's per-datapoint reweighting adds almost nothing.

### Corrected program bottom line
- Toys, kernel, P2 theorem, geometry recovery: **solid** (unchanged).
- The organizing law is real but **conditional on input isotropy** — a clean statement about the
  operator, not a prediction for real activations.
- **On real bilinear-MLP activations, the Jacobian metric ≈ cosine, and its advantage over a
  spectrum-matched control is within noise of zero.** This is a cleaner and more honest negative than
  ticks 9–10's "small but real": on a like-for-like test (same G, only the input swapped), the real-data
  advantage is +0.0012 / −0.0012.

The method's leverage requires BOTH an anisotropic operator AND input that populates its anisotropy.
Trained LM MLPs fail the second condition even when (weakly) satisfying the first.

### Standing rules observed
Audited a tidy prior conclusion against its own discrepant number rather than leaving it; isolated the
cause with a swap-one-thing ladder (real / matched-cov / isotropic); reported that the real-data effect
is negative vs the control; retracted two of my own earlier over-readings.

---

## 2026-07-11 — tick 14: answering Logan's live questions (decoder-only, outliers, impossibility detail)

### Q: cluster on the decoder / linear-transformation only — better or worse than activations?
`M_i = y_i x_i^T/||x_i||^2` is the best rank-1 linear map the layer applied to datapoint i
(`cos(M_i,M_j)=cos_x·cos_y`, P3). Compared across a ladder:

**Toy (DGP-D, ground-truth gate), ARI:**
| object | ARI(mechanism) |
|---|---|
| input x | 0.002 | hidden Lx⊙Rx | 0.002 | output y | 0.226 |
| **rank-1 M (linear transf)** | **0.355** | full Jacobian | 0.001 | restricted Jacobian | **1.000** |

So M-clustering **beats activations** on the toy (captures the input×output joint that reveals the
gate), beats even the full Jacobian (which is gate-contaminated, P2), but loses to the restricted
bilinear Jacobian.

**Real 124M model, surrogate R² (24k train, λ=3, k=16, 5 seeds):**
| layer | raw x | G-metric | rank-1 M | output y | [x;y] | G_rand | global |
|---|---|---|---|---|---|---|---|
| 2 | 0.5011 | 0.5118 | 0.4920 | 0.4701 | 0.4703 | 0.5049 | 0.4730 |
| 6 | 0.5348 | 0.5354 | 0.5179 | 0.5024 | 0.5027 | 0.5382 | 0.4840 |
| 10 | 0.5321 | 0.5253 | 0.4241 | 0.4562 | 0.4563 | 0.5327 | 0.5217 |

**On the real model M-clustering does WORSE than raw x** — as do output-y and [x;y]. Output-similarity
is the wrong grouping for a surrogate task (you want same-MAP, not same-output points), and without a
gate there's no joint structure for M to exploit. This is despite M getting an unfair peek at y, so it
isn't a false-positive risk. (Caveat: exact M kernel cos_x·cos_y is a product, not Euclidean;
approximated by normalized [x;y]; all three output-informed variants agree.)

### Q: fold out outlier dims/datapoints (LLM.int8 style)?
Layer 6, 124M model. Dropping outliers flips the G-vs-control advantage from **−0.0028 to +0.0066** —
right mechanism — but the effect stays tiny. This model is a weak test: post-RMSNorm inputs have
identical token norms (27.7, no norm-outlier tokens) and dim outliers are only 1.6–1.8× median
variance (vs 100× in unnormalized LLM residual streams). The idea is sound; available bilinear models
are all normed, so there are no dramatic outliers to fold. Worth revisiting on an unnormalized
bilinear model if one exists.

### Impossibility result — precise statement (recorded for the writeup)
Bilinear layer `yₒ = Σᵢⱼ B[o,i,j] xᵢxⱼ`; Jacobian is one contraction `J(x)=2 B_sym(·,x,·)`, linear in x.
For an input-gated layer (`x=[c;s]`, `s=ε·e_g`, `y=A_g c`):
`J = [ ∂y/∂c | ∂y/∂s ] = [ A_g | (1/ε)·[A₁c,…,A_{k_g}c] ]`.
The gate block is (1) O(1/ε), (2) **gate-independent** (all experts, identical across gates for fixed c;
measured 0.00e+00). So `cos(J,J')` is dominated by a gate-blind term → recovers content, not operator.
No ε escapes: small ε → gate block dominates J; large ε → gate visible in raw x. **Dissociation window
empty for a single layer.**
- **Symmetry irrelevant:** J sees only `B_sym` regardless (gauge check `G(D,L,R)=G(D,R,L)`, 1e-13). The
  impossibility is about the gate being an input to the *same* layer, not tensor symmetry.
- **Tensor form vs CP: same object.** CP is a factorization of B; J, G, impossibility identical. CP only
  makes G cheap (Hadamard-Gram) and the params O(hd) not O(d²o).
- **Novelty (honest):** elementary calc; "gate derivatives contaminate Jacobian attribution" is known
  saliency-literature folklore. The clean bilinear specialization + the empty-window consequence
  (falsifying spec P2, motivating depth in DGP-E) is the specific, useful, not-previously-stated bit.

---

## 2026-07-11 — tick 15: the JOINT weight×data anisotropy sweep (Logan's request). Data-aware predictor.

Tick 12 varied weight anisotropy only (isotropic input); tick 13 showed input matters. This is the 2D
sweep. Knob 1: rank r of L,R row subspace → eff_rank(G). Knob 2: input variance concentrated in
{G-top | random | G-bottom | isotropic} subspace. 5 seeds, k=12, matched G_rand control.

| r | data align | eff_rank/d | corr(cos_J,cos_x) | G−cos | G−G_rand |
|---|---|---|---|---|---|
| 4 | G-top | 0.081 | 0.447 | +0.506 | +0.576 |
| 4 | random | 0.081 | 0.407 | +0.562 | +0.644 |
| 4 | G-bottom | 0.081 | 0.354 | +0.321 | +0.393 |
| 4 | isotropic | 0.081 | 0.251 | **+0.760** | +0.813 |
| 12 | G-top | 0.241 | 0.837 | +0.145 | +0.184 |
| 12 | isotropic | 0.241 | 0.475 | +0.380 | +0.382 |
| 48 | any | 0.922 | 0.93–0.99 | +0.03–0.07 | +0.04–0.07 |

**Predictor correlation across all 12 cells:**
- weights-only `1 − eff_rank(G)/d`: **+0.76**
- **data-aware `1 − corr(cos_J,cos_x)`: +0.83** (and +0.83 vs the G_rand advantage)

### Three findings, unifying ticks 12–13
1. **Weight anisotropy necessary, not sufficient:** isotropic G (r=48) → advantage +0.03–0.07 for ANY
   data. No data structure rescues an isotropic operator.
2. **Given anisotropic G, data decides how much shows:** at r=4, alignment swings advantage +0.32→+0.76
   (2.4×). It is the weight×data INTERACTION, as hypothesized.
3. **Naive "aligned → big" is WRONG.** Isotropic data gives the largest advantage (+0.76), not G-top
   (+0.51). The rule is: data must EXPOSE G's eigenvalue spread. Isotropic data spans the whole spectrum;
   data trapped in a locally-flat region of G (top or bottom) sees G≈scalar≈cosine. `corr(cos_J,cos_x)`
   measures this directly.

### This explains the real-model null exactly
Real activations have `corr(cos_J,cos_x) = 0.62–0.94` (high) → G≈cosine ON THAT DATA → small advantage,
regardless of G's global anisotropy. Real activations sit where G is effectively flat.

### The complete when-does-it-work statement (the honest deliverable)
The method beats cosine iff **the operator is anisotropic AND the data populates its eigenvalue spread**.
The single measurable predictor is `1 − corr(cos_J,cos_x)` on the actual data (ρ=0.83). Weights-only
eff_rank(G) is a necessary screen but blind to alignment (ρ=0.76). This is data-aware, cheap, and
predicts every result in the program: toys (isotropic-ish data + gated operator → large), real MLPs
(structured data in flat region → ~0).

### Truth-determination on real LLMs (recorded, Logan asked)
No ground-truth mechanism labels exist on real models. Two label-free PROXIES: (1) surrogate test —
per-cluster linear map predicts held-out layer output, must beat cosine/G_rand/random/global; (2)
intervention test — within- vs across-cluster patch CE. Both verify "locally-linear-consistent clusters",
NOT "interpretable roles". Real ground truth exists ONLY in the toys (constructed labels). This gap is
why the real-model result is a characterized negative, not a discovery.

### Standing rules observed
2D grid with both controls; the data-aware predictor tested AGAINST the weights-only one (it could have
tied/lost); my own naive "alignment" prediction reported as refuted; the real-model null located on the
same predictor rather than treated as separate.

---

## 2026-07-11 — tick 16: squared-attention forward BUG found+fixed (Logan flagged it). Attention testbed now valid.

Loading squared/bilinear-attention models to test whether attention operations are more anisotropic
than MLPs (the one untested large-effect candidate). Initial CE was 7.5 (vs ~3.4 for the good models).
Verified NOT my extraction (official train_gpt2.py GPT class gives identical 7.526) and NOT a key
mismatch (checkpoint has c_q/c_k/c_v/c_proj, no q2/k2). **The repo's `naive_squared_attention` is
UNNORMALIZED** — `pattern=(q·k/D)²` with no row-sum division. Swept variants:

| pattern | CE |
|---|---|
| `(scores/D)²` (repo code) | 7.526 |
| `(scores/D)² / row-sum` (**normalized**) | **3.513** |
| `softmax(scores/√D)` | 5.490 |

The checkpoint was trained with **normalized** squared attention. Fixed `tt_model.py`. Verified after fix:
gpt2-sqrd-attn-12l 3.513, gpt2-bilinear-sqrd-attn-12l 3.432, gpt2-bilinear-18l (573M) 3.358,
**gpt2-bilinear-sqrd-attn-18l (573M) 3.418**. All correct.

**Consequences:** (a) all prior MLP results used the softmax model (squared_attn=False, never calls this
function) — UNAFFECTED. (b) this tick's earlier attention screen (corr(cos_J,cos_x)=0.055) was on the
buggy forward — VOID, redoing. (c) squared-attention models are now valid testbeds at both 124M and 573M.

---

## 2026-07-11 — tick 17: ATTENTION operations are strongly anisotropic (screen clears). Validation next.

Per-query attention-operation Jacobian J_q = ∂z_q/∂x_q (z_q = head output in value space, context
fixed, q's own query/key/value from x_q). Jacobian VERIFIED correct: float64 jacrev == autograd.functional
== finite-diff to 8e-9 (the earlier 4e-1 mismatch was float32 finite-diff being numerically unreliable,
not a Jacobian bug). Leverage = 1 − corr(cos_J, cos_x) over query positions:

| model | layer | attention heads | in-model MLP |
|---|---|---|---|
| sqrd-attn 124M | 6 | 0.786, 0.821, 0.902 | — |
| bil-sqrd-attn 573M | 9 | 1.214, 1.438 | 0.629 |

vs MLP baseline 0.06–0.38. **Attention-operation Jacobians are far more anisotropic than MLP maps** —
leverage >1 (negative corr) on the 573M model means the operation is nearly independent of the query's
own residual direction (which keys you read ≠ what you are).

⚠ **Necessary, not sufficient.** Leverage measures "different from cosine"; a RANDOM per-token matrix
also scores ~1. The screen strongly clears the necessary gate, but meaningfulness needs the surrogate/
pattern-type validation (next). Attention gives a semi-ground-truth MLPs lacked: the attention pattern
type (which relative positions a query reads), against which J_q-clusters can be scored with a control.

**Note:** there is NO closed-form G kernel for attention (context-dependent), so per-token Jacobians are
materialized via autodiff (jacrev). ~450 positions/head is seconds; full clustering is tractable.

## tick 17 cont'd — meaningfulness validation: first attempt INCONCLUSIVE (degenerate label)
Pattern-type = dominant relative offset, bucketed {BOS-sink, self, prev, near, far}. But layer-6 head-4
attends diffusely: label distribution was **1286/1400 = class "far"** (92% one class). ARI to pattern-type:
J_q −0.004, x_q 0.020, random-matrix −0.000, chance −0.006 — all near chance because the LABEL barely
varies, not because the method fails. **Inconclusive; the ground-truth was degenerate.**

Correct next step: (a) a head with varied attention (early-layer prev-token/positional, not diffuse);
(b) a CONTINUOUS pattern-summary label (entropy, mass-on-prev, mass-on-BOS, mean offset) and measure
variance-explained by each clustering, with the random-matrix control (leverage~1) which MUST fail.
Leverage screen (strong, real) stands; meaningfulness is UNPROVEN.

## tick 17 final — attention Jacobian clustering is MEANINGFUL at early layers (validated, controlled)
Continuous pattern-summary target [entropy, mass-prev, mass-BOS, mean-offset, mass-near]; variance-
explained by clustering; 5 seeds. sqrd-attn 124M.

| head | J-clusters | x-clusters | random-matrix | verdict |
|---|---|---|---|---|
| L2H0 | **+0.114** | +0.064 | +0.004 | J > x (1.8×), J ≫ random |
| L2H3 | **+0.227** | +0.014 | +0.004 | **J ≫ x (16×), J ≫ random** |
| L6H0 | +0.111 | +0.130 | +0.004 | J < x (loses) |
| L6H3 | +0.092 | +0.138 | +0.004 | J < x (loses) |

**The random-matrix control (leverage~1) explains ~0 everywhere** → high leverage alone is worthless;
the Jacobian's eigen-structure carries the signal. **At EARLY layers, attention-operation Jacobian
clustering recovers attention-behavior structure that residual clustering misses** (L2H3: 16× over x).
At LATE layers it loses — the residual stream already encodes the attention behavior there.

### This is the program's first real-model POSITIVE that beats both activation clustering AND a control.
Mechanism: early-layer attention is positional/structural (residual doesn't yet encode it → Jacobian
wins); late-layer attention is content-driven (content predicts it → residual wins). Consistent with the
tick-15 law: the method wins where the operation varies in a way the data geometry doesn't capture.

**Honest scope:** validated on one 124M squared-attn model, one head cleanly (L2H3) + one moderately
(L2H0); layer-dependent; the target is a pattern-SUMMARY not verified "interpretable roles". But it is a
controlled, replicated (5-seed) positive with the random-matrix control passing — the strongest
real-model evidence in the whole program. Next: confirm on 573M, more heads, and check whether the
early-layer J-clusters correspond to nameable attention types (prev-token, BOS-sink, positional).

---

## 2026-07-11 — tick 18: ⚠ CORRECTING tick 17's "early layers win" — it was an over-read from 4 heads.

Systematic depth sweep (var-explained of pattern-summary; 3 heads/layer avg; random-matrix control):

| layer | J | x | random | J − x |
|---|---|---|---|---|
| 0 | +0.092 | +0.049 | +0.011 | +0.043 |
| 2 | +0.123 | +0.077 | +0.011 | +0.045 |
| 4 | +0.137 | +0.165 | +0.012 | **−0.029** |
| 6 | +0.143 | +0.095 | +0.012 | +0.049 |
| 8 | +0.159 | +0.055 | +0.012 | **+0.104** |
| 10 | +0.119 | +0.038 | +0.012 | +0.081 |

**Tick 17's "early layers win, late layers lose, clean positional-vs-content mechanism" is RETRACTED.**
It rested on 4 hand-picked heads (L6H0/H3 happened to lose). Averaged over heads 0/2/4:
- J beats x at **5 of 6 layers** (all but L4), and the advantage is **largest at LATE layers** (L8 +0.104,
  L10 +0.081) — the opposite of the early-layer story.
- So the effect is **head-dependent, not cleanly layer-dependent.** No depth law.

**What robustly survives (both controlled, replicated):**
1. J beats the random-matrix control by ~10× at EVERY layer (random 0.011–0.012 flat) → attention
   operations carry real structure; high leverage alone is worthless.
2. J beats residual (x) clustering on average across the model, but **modestly (+0.04–0.10) and
   head-dependently** (L4 loses; tick-17's L6H0/H3 lost).

**Corrected honest headline:** attention-operation Jacobian clustering recovers attention-behavior
structure that a random matrix cannot (robust, ~10×), and modestly beats residual clustering on most
heads (+0.04–0.10 var-explained) — but there is NO clean depth trend, the effect is head-specific, and
the target is a pattern-summary, not verified interpretable roles. This is still the program's best
real-model positive (beats both activation clustering and a control on most heads), just narrower than
tick 17 claimed.

Standing-rule note: tick 17's clean story came from 4 non-random heads; the systematic sweep refuted the
depth mechanism. Over-reading from hand-picked cases is exactly what the "no single-seed/cherry-pick"
rule guards against — caught it one tick later by sweeping.

---

## 2026-07-11 — tick 20: attention positive CONFIRMED at 573M. Advantage is mid/late-layer, not early.

Full pattern-summary validation on the 573M `bilinear-sqrd-attn-18l` (CE 3.42), 5 seeds:

| head | J | x | random-matrix | J beats both? |
|---|---|---|---|---|
| L3H0 | +0.179 | +0.196 | +0.010 | no (early) |
| L3H3 | +0.086 | +0.098 | +0.008 | no |
| L3H6 | +0.044 | +0.063 | +0.006 | no |
| **L9H0** | **+0.246** | +0.105 | +0.009 | **yes (2.3×)** |
| **L9H3** | **+0.143** | +0.028 | +0.006 | **yes (5×)** |
| **L9H6** | **+0.114** | +0.013 | +0.008 | **yes (9×)** |
| L15H0 | +0.094 | +0.020 | +0.008 | yes |
| L15H3 | +0.152 | +0.098 | +0.008 | yes |
| L15H6 | +0.161 | +0.042 | +0.007 | yes |

**Cross-model-validated (124M + 573M):**
1. J beats the random-matrix control ~15–25× at EVERY head (structure real; leverage alone worthless). Fully robust.
2. J beats residual clustering at 6/9 heads (573M) / 5/6 layers (124M); where it wins, 2–9×.
3. **The advantage is at MID/LATE layers, NOT early** — L3 loses on all heads (573M); largest wins at
   L8/L10 (124M) and L9 (573M). Confirms for the 3rd time that tick-17's "early wins" was backwards.

Honest scope unchanged: head-dependent; target is a pattern-summary (J is partly derived from the
pattern, mild circularity, though the random-matrix control rules out triviality). But it is now a
**scale-confirmed, controlled, 5-seed-replicated positive** — the program's strongest real-model result.

---

## 2026-07-11 — tick 21: ⚠ ROTARY SIGN BUG in the attention Jacobian. Magnitudes (ticks 17-20) CORRECTED.

Building the causal cluster-mean-query test, my custom forward failed its CE-reproduction check (4.07 vs
5.89). Root cause: `apply_rot` had flipped sin signs vs the model's `apply_rotary_emb`
(model: `y1=x1c+x2s, y2=-x1s+x2c`; mine: `x1c-x2s, x1s+x2c`). **The wrong rotation direction was used in
ALL attention runs (ticks 17-20)** — so J_q and the pattern-summary were from a rotation-wrong surrogate,
not the model's real attention. The earlier finite-diff check passed because it compared J to the SAME
buggy zfun; only the full-forward CE check (0.00e+00 after fix) caught it.

### Corrected attention screen (124M, CORRECT rotary, 5 seeds)
| head | leverage | J | x | random | vs ticks 17-20 |
|---|---|---|---|---|---|
| L2H3 | 0.926 | +0.083 | +0.040 | +0.013 | was J 0.227/x 0.014 (**"16×" → 2×**) |
| L6H0 | 0.775 | +0.105 | +0.118 | +0.009 | still loses |
| L8H0 | 0.897 | +0.105 | +0.057 | +0.009 | J beats x 1.8× |
| L8H4 | 0.613 | +0.097 | +0.061 | +0.009 | J beats x |

### What survives / what's corrected
- **SURVIVES (qualitative):** attention ops anisotropic (leverage 0.61–0.93); J beats the random-matrix
  control ~8–11× everywhere; J beats residual clustering at most heads (3/4 here).
- **CORRECTED (quantitative):** the magnitudes were INFLATED by the bug. L2H3's "16× over x" is really
  2×. The tick-17→20 attention numbers (incl. the 573M "2–9×") are overstated and need re-running with
  the correct rotary. The random-control margin (~10×) is robust; the J-vs-x margin is smaller (~2×).
- The custom forward now reproduces model CE exactly (0.00e+00), so the causal cluster-mean-query
  intervention is unblocked for next tick.

**Lesson (standing rules):** an internal consistency check (J vs finite-diff of the same function) CANNOT
catch a bug shared by both sides. Only an INDEPENDENT ground truth (the model's own CE) exposed the
rotary convention error. Always verify a custom forward against the real model's output before trusting
any quantity computed from it.

---

## 2026-07-11 — tick 22: attention clusters INSPECTED + causal switch test. Mildly meaningful, NOT better than x.

L8H0 (a J-winner on pattern-summary), K=8, correct rotary, forward verified vs model CE (0.00e+00).

**What the clusters are:** primarily an ATTENTION-DISTANCE decomposition (cluster median offsets 0, 5, 8,
16, 19, 20, 30, 38). Cluster 1 (n=652) = local/self; rest progressively distant. Some content structure
in tail clusters: c2 code/markup (`> \ <? UTF link`), c4 Spanish (`de la ón des`), c7 JSON (`" ", ":`).
Mildly interpretable (range + context-type pockets), NOT crisp roles.

**Causal switch test** (cluster-mean-query intervention; higher CE = more damage; random control):

| clustering | within | swapped | swap−within |
|---|---|---|---|
| J-clusters | 3.9493 | 3.9526 | +0.0033 |
| x-clusters | 3.9460 | 3.9511 | **+0.0051** |
| random | 3.9499 | 3.9501 | +0.0002 (control ✓) |

baseline 3.9439; global-mean-query 3.9499.

**Verdict: J-clusters are real (switch cost +0.0033 >> random 0.0002) but x-clustering is causally MORE
coherent** (larger swap gap, lower within-damage). The pattern-summary advantage of J does NOT translate
to a causal or interpretability advantage over clustering the residuals, on this head.

**Caveats (stated, not hidden):** (1) this head is causally weak — global-mean-query costs only 0.006
nats, so little signal; poor testbed. (2) the intervention swaps the QUERY, and query ≈ rms(W_Q x), so
x-clustering is favored almost by construction; a fair test intervenes on the OPERATION not the query.

**Honest state of the attention thread:** J beats a random-matrix control robustly (attention ops carry
structure), and beats residual clustering ~2× on a derived pattern-summary — but that advantage does NOT
survive as an interpretability/causal win over x-clustering when inspected directly. The strong claim
("J reveals meaningful roles x misses") is UNSUPPORTED on this head. Not cherry-picking a better head.

---

## 2026-07-11 — tick 23: RESOLVED — the QUERY (Wq·x), not the Jacobian, is the causally-meaningful attention object.

Screened heads for causal importance (global-mean-query damage): most <0.01; L6H0 (+0.028) and L2H4
(+0.010) matter. Causal switch test on both, 5 kmeans seeds, swap-within damage (higher = clusters are
causally-distinct operations):

| clustered by | L6H0 | L2H4 |
|---|---|---|
| **query q̃ = rms(Wq·x)** | **+0.0090±0.005** | **+0.0098±0.003** |
| J_output (Jacobian) | +0.0009±0.000 | +0.0022±0.001 |
| residual x | +0.0005±0.001 | +0.0037±0.001 |
| random | +0.0000 | +0.0002 |

**The query readout `Wq·x` is the causally-meaningful clustering** — ~5× above everything, robust across
seeds and both causally-important heads. **The Jacobian is causally near-useless** (≈ random). The raw
residual is inconsistent (the query subspace is swamped in full x).

(Single-seed L6H0 had shown query 0.0194 — a seed-0 outlier; 5-seed mean is 0.0090. Third time seeds
corrected an inflated single-seed magnitude. Direction robust, magnitude halved.)

### This resolves the attention thread, and revises it downward
- **The attention JACOBIAN does NOT deliver causally.** Its ticks 17–22 "wins" (leverage, beating a
  random-matrix on a derived pattern-summary) do NOT translate to a causal or interpretability advantage;
  causally it is ≈ random. The strong claim is dead.
- **The QUERY readout `Wq·x` IS causally meaningful** — clustering by it gives causally-distinct attention
  operations. This is a real, causally-validated, weight-informed finding — just not for the Jacobian.
- **Program theme confirmed:** the useful object is always a WEIGHT-DERIVED READOUT that isolates the
  operation (content-restricted J for MLPs; query `Wq·x` for attention). The FULL Jacobian is contaminated
  in both cases (gate block for MLPs; context/values for attention) — the impossibility theorem's shape,
  recurring.

### Net real-model verdict for the whole program
The Jacobian-clustering method has NO demonstrated real-model causal win: MLPs are a data-driven null,
attention Jacobians are causally ≈ random. But the investigation surfaced a clean positive: clustering
attention QUERIES (a weight readout) recovers causally-distinct operations. The method's VALUE is as a
lens that points to the right weight-derived readout, not as the readout itself.

---

## 2026-07-11 — tick 24: the pattern-summary metric MEASURED THE WRONG THING; query-clusters are content-types.

Characterizing the causally-validated QUERY clusters (L6H0). Two findings, both corrective.

### (a) Pattern-summary var-explained DISAGREES with the causal test (and the causal test is right)
| clustered by | pattern-summary var-expl (5 seeds) | causal switch (tick 23) |
|---|---|---|
| query Wq·x | +0.143 | **+0.009 (best)** |
| J_output | +0.237 | +0.001 |
| residual x | **+0.263 (best)** | +0.0005 |
| random | +0.003 | +0.000 |

The query is WORST on pattern-summary but BEST causally; x is the reverse. Explanation: the
pattern-summary (entropy, mean-offset) is dominated by attention DISTANCE/POSITION (which x and J encode),
while the causally-relevant axis is the query's CONTENT-SELECTIVITY (which Wq·x isolates, position-
orthogonal). **⇒ The entire tick 17–20 pattern-summary methodology measured position, not function.**
That is why "J beats x on pattern-summary" never became a causal win. The ticks 17–20 attention
pattern-summary results are RETRACTED as evidence of mechanism (they measured a position proxy).

### (b) The query-clusters are query-CONTENT types, not nameable mechanistic roles
L6H0, K=8: clusters group by what is querying (c4 prepositions `in/the/and/to/of`; c7 punctuation/code
`( C / </`; c3 sharp `'.'→'.'` self-attend, entropy 0.45). But ATTENTION BEHAVIOR is near-uniform:
almost every cluster attends to a recent DELIMITER (`.` `\n` `,`). So L6H0 ≈ an "attend-to-recent-
punctuation" head, and the causally-distinct query-clusters sub-divide it by the querying token's type,
NOT by distinct roles like "previous-token" vs "induction".

### Honest final state of the attention thread
- Jacobian clustering: causally ≈ random (tick 23). No real-model win.
- Pattern-summary metric: measured position not function; ticks 17–20 "J beats x" retracted as mechanism evidence.
- Query clustering (Wq·x): causally meaningful (robust, tick 23) but the clusters are query-content types,
  not nameable mechanistic roles — at least on this head.
- The clean, surviving positive: clustering by the query readout Wq·x recovers CAUSALLY-DISTINCT
  operations (switching between them costs CE); it just doesn't yield crisp interpretable head-roles here.

---

## 2026-07-11 — tick 25 (Logan): include BOTH q and k weights — the QK score kernel. Verified, but redundant.

The attention score is bilinear across positions: `s(q,k) = (Wq x_q)·(Wk x_k) = x_q^T (Wq^T Wk) x_k`. So
`W_QK = Wq^T Wk` is a weights-only interaction matrix, and there's a G-kernel for the score:
query key-selectivity `c_q = Wk^T Wq x_q`, with `⟨c_q,c_q'⟩ = x_q^T G_QK x_q'`, `G_QK = Wq^T Wk Wk^T Wq`.
Identity VERIFIED (rel err 3e-7).

Causal switch test, L6H0, 5 seeds (swap-within damage; higher = clusters causally distinct):
| clustered by | swap-within |
|---|---|
| query `Wq·x` | +0.0090±0.0050 |
| **QK `G_QK` (`Wk^T Wq x`, both weights)** | **+0.0072±0.0014** (tied) |
| residual x | +0.0005±0.0006 |
| random | +0.0000 |

**Including the K weights (the QK score kernel) does NOT beat query-only causally** — statistically tied
(slightly tighter variance, so marginally more stable, but same mean). Reason: the query already
implicitly contains the key-selectivity — keys are defined relative to queries (score = query·key), so
`Wq x_q`'s direction already determines which keys score high; mapping through `Wk^T` reweights but adds
no causal information. The K weights are "already in" the query.

**Net (attention, complete):** the causally-meaningful per-token object is the QUERY `Wq·x`; the
principled both-weights extension (QK score kernel `G_QK`) is verified but causally redundant with it.

---

## 2026-07-11 — tick 26 (Logan): two-QK / squared TENSOR kernel, the OV pathway, full-head contraction.

Logan: "bilinear attention with two q and k matrices? OV? fold in the full attention matrix (tensor
network contraction)?" Checkpoints on disk are ALL single-QK SQUARED attention (c_q,c_k,c_v,c_proj;
NO c_q2) — the genuine two-different-QK bilinear (`CausalBilinearSelfAttention`, pattern=(q·k)(q2·k2))
has no trained checkpoint, would need training. But SQUARED attention IS the diagonal case (W_Q2=W_Q):
the square makes the query's key-selectivity a rank-1 QUADRATIC form u_q u_q^T (u_q=query covector), so
the correct kernel is cos^2 (== |cos|, per the degree-2 homogeneity result), not cos. L6H0, K=8, 5 seeds,
same swap-within causal metric as ticks 23/25.

(A) QUERY pathway (WHERE) — override head query:
| clustering | swap-within |
|---|---|
| query cos (tick25 baseline) | +0.0096±0.0042 |
| **query cos^2 / rank-1 form (SQUARED-correct)** | **+0.0120±0.0015** (higher mean, ~3× tighter) |
| residual x | +0.0065±0.0026 |
| random | +0.0001 |
→ the tensor/squared kernel modestly-but-consistently beats plain cos and is much more stable. The
"two matrices" structure the square already carries is real and the right object for squared attention.

(B) OV / VALUE pathway (WHAT is copied) — override head value, cluster tokens by head value v_k:
| head | value-cos | resid-x (same intervention) | random |
|---|---|---|---|
| L2H4 | +0.0093±0.0007 | +0.0058±0.0012 | +0.0004 |
| L4H2 | +0.0055±0.0031 | +0.0039±0.0013 | −0.0001 |
| L6H0 | +0.0422±0.0214 | +0.0072±0.0030 | +0.0008 |
| L9H0 | +0.0048±0.0040 | −0.0014±0.0037 | +0.0001 |
| L11H3 | +0.0025±0.0012 | +0.0041±0.0004 | +0.0001 |
→ value/OV clustering is above random at every head, above residual-x at 4/5 (loses at L11H3). Effect is
MODEST (~1.3–1.6× resid-x) at typical heads; L6H0's +0.042 (~6×) is an OUTLIER (L6H0 was pre-selected as
the most causally-important head). NOT "the strongest attention effect" — that was a single-head over-read,
caught by the head sweep. Confound: swapping a value changes output more directly than swapping a query
(values enter linearly, queries via the pattern), so (B) magnitudes aren't apples-to-apples with (A); the
clean statement is value-clustering > residual-x under the IDENTICAL value-swap intervention.

(C) full-head contraction — is OV redundant WITHIN a head? ARI(query-clusters, value-clusters) ≈ 0.07
(3 seeds). WHERE and WHAT are near-INDEPENDENT partitions. So unlike the KEY weights (tick25: redundant
with the query), the OV circuit carries structure the query does NOT — the full head genuinely factors as
WHERE(query) × WHAT(OV), two near-orthogonal causal groupings. Folding OV in is NOT redundant.

**Net (tick 26):** (i) for squared attn the correct kernel is cos^2 — small consistent win + more stable;
(ii) the OV/value pathway is a second, independent causal-clustering axis (above controls at most heads,
modest, head-dependent, L6H0 outlier); (iii) the two circuits are near-orthogonal (ARI 0.07), so the
head = WHERE × WHAT. Two-different-QK bilinear untested (no checkpoint).

---

## 2026-07-11 — tick 27 (Logan: "load the 500M ones and check the keys"): GENUINE two-QK bilinear on a real checkpoint.

Logan was right — the 500M `gpt2-bilinear-sqrd-attn-18l-9h-1152embd` has c_q,c_k,c_q2,c_k2
(config bilinear_attn=True, squared_attn=True): pattern_qk = (q1·k1)(q2·k2)/D^2, a PRODUCT of two DIFFERENT
bilinear forms — the genuine "two q,k matrices". UNNORMALIZED (custom forward matches model CE 3.79 to
2e-4; do NOT add row-norm — that was only the single-QK variant's bug). [The 12l models are single-QK; the
573M attention "positive" of ticks 20-21 was actually this two-QK model, loaded via CausalBilinearSelfAttn.]

Query's key-selectivity is now the pair (q1til,q2til) and their product form; cosine of vec(q1n⊗q2n) =
cos(q1,q1')·cos(q2,q2') (the exact two-matrix kernel). Screen (global-mean-query damage): top heads L8H3
(+0.017), L6H3 (+0.017). Override BOTH q1,q2 with the cluster rep; vary only the clustering feature; 5 seeds:

| head | q1 only | q2 only | both concat | both tensor (product kernel) | resid x | random | mean\|cos(q1,q2)\| |
|---|---|---|---|---|---|---|---|
| L8H3 | +0.0026±.0023 | +0.0004±.0012 | **+0.0042±.0019** | +0.0035±.0010 | +0.0015 | +0.0001 | 0.047 |
| L6H3 | +0.0047±.0011 | +0.0049±.0014 | +0.0065±.0009 | **+0.0072±.0014** | +0.0032 | +0.0002 | 0.073 |
| L3H3 | +0.0009±.0004 | +0.0014±.0010 | +0.0010±.0006 | **+0.0022±.0011** | +0.0008 | +0.0001 | 0.076 |

**Including BOTH query matrices beats either alone at all 3 heads (~1.5–2.4×), above resid-x and random.**
Mechanism: the two query matrices are NEAR-ORTHOGONAL (mean |cos(q1,q2)| ≈ 0.05–0.08 per token) — they
encode complementary selection criteria, so the second matrix is NOT redundant and adds real causal signal.
This is the clean CONTRAST with the KEY weights (tick 25: redundant with the query, adds nothing, because
the query already contains the key-selectivity). q2 is a genuinely independent matrix, so folding it in
helps. The principled product kernel cos(q1)·cos(q2) is competitive with / better than concat (best 2/3).

**Net (tick 27):** on a real two-QK checkpoint, "include both q,k matrices" is a genuine causal WIN
(unlike single-QK's key-redundancy) — driven by the two query matrices being near-orthogonal. The taxonomy:
KEY weights redundant (tick25); squared single-QK gives a modest rank-1 tensor-kernel gain (tick26);
two-different-QK bilinear gives a real gain because the matrices are complementary (tick27).

---

## 2026-07-11 — tick 28 (autonomous): is the two-QK gain a weights-only pre-screen? + matched-dim control. TEMPERS tick 27.

Followed tick 27 with (a) a hoped-for WEIGHTS-ONLY pre-screen and (b) the matched-DIMENSION control the
standing rules demand (both-concat doubles feature dim vs q1-alone — could beat it on dimensions alone).

(a) Weights-only alignment A_w = ||Wq_h Wq2_h^T||_F/(||Wq_h||·||Wq2_h||) for all 162 heads: near-CONSTANT
(mean 0.060, range 0.032–0.138) and corr(A_w, data|cos(q1,q2)|) = +0.054 ≈ 0. **No weights-only pre-screen**
— A_w does not predict the data-level orthogonality or the gain. The S11-style law does NOT materialize for
two-QK. corr(A_w,gain)=−0.507 but n=12 w/ leverage points and data|cos| corr only −0.22 → NOT robust.

(b) Gain (both-concat − best-single) across 12 heads spanning the A_w range: mostly ≈0 or slightly NEGATIVE
(L0H2 +0.0048 the only clear positive; most others −0.001 to −0.004). So tick 27's clean positive is
HEAD-DEPENDENT, not universal — it was measured on the top-3 CAUSAL heads; on causally-unimportant heads
(query barely matters) "both vs single" is noise around 0.

(c) Matched-dim control ([q1;q2] vs [q1;noise], [q2;noise], same 2·hd dim, 5 seeds) on positive heads:
| head | q1 | q2 | [q1;q2] | [q1;noise] | [q2;noise] | q2 real beyond dims? |
|---|---|---|---|---|---|---|
| L0H2 | +.0074 | +.0071 | +.0122 | +.0096 | +.0115 | YES |
| L6H3 | +.0047 | +.0049 | +.0065 | +.0050 | +.0044 | YES |
| L8H3 | +.0026 | +.0004 | +.0042 | +.0002 | +.0028 | YES (noise destroys q1→+.0002; q2 rescues) |
| L3H3 | +.0009 | +.0014 | +.0010 | +.0007 | +.0018 | NO — dimensionality artifact (tick27's tensor "win" here) |

**Net (tick 28, tempering tick 27):** the two-QK "both matrices help" effect is REAL on causally-important
heads (beats BOTH matched-dim controls at 3/4: L0H2, L6H3, L8H3 — the second near-orthogonal query matrix
carries genuine causal signal, not just added dimensions), but is HEAD-DEPENDENT (≈0/negative on most
heads) and has NO weights-only pre-screen (A_w near-constant, uncorrelated). Standing rules did their job:
the broad screen caught that the 3-head positive doesn't generalize, and the matched-dim control salvaged
the real part (3/4) while exposing L3H3 as a dimensionality artifact.

---

## 2026-07-11 — tick 29 (autonomous, closes LOG Q1): OV / WHERE×WHAT on the 500M two-QK model — PARTIAL replication.

Tested whether tick-26's OV factorization (12l single-QK: value pathway beats resid-x at 4/5 heads;
ARI(query,value)≈0.07 ⇒ WHERE and WHAT near-independent) holds on the validated 500M two-QK model. Value-
swap causal test, 5 seeds, controls resid-x (same intervention) + random; ARI(both-query-clusters, value-
clusters).

| head | value-cos | resid-x | random | ARI(qry,val) |
|---|---|---|---|---|
| L0H2 | +0.0014±.0012 | +0.0006±.0014 | +0.0001 | 0.24 |
| L6H3 | +0.0027±.0012 | +0.0031±.0028 | +0.0003 | 0.06 |
| L8H3 | +0.0035±.0006 | +0.0017±.0005 | +0.0002 | 0.22 |
| L9H3 | −0.0013±.0009 | −0.0008±.0006 | +0.0000 | 0.13 |
| L11H5 | +0.0061±.0013 | +0.0036±.0026 | −0.0001 | 0.20 |

**PARTIAL / WEAKER replication.** OV pathway above random at 4/5 heads and beats resid-x at 3/5 (L0H2, L8H3,
L11H5), loses at L6H3, both-negative at L9H3 — weaker + noisier than the 12l (4/5 beat resid-x). And the
WHERE×WHAT independence is only PARTIAL: ARI(query,value) 0.06–0.24, mean ~0.17 — clearly ABOVE the 12l's
~0.07, i.e. the query and value partitions share MODERATE structure on this two-QK model, not near-
independence. (The script's auto-printed "ARI~0, as on 12l" is over-optimistic and is corrected here.)

**Net (tick 29):** the OV/value pathway is a real-but-modest causal axis on the 500M two-QK model too, but
the clean WHERE×WHAT factorization of tick 26 is MODEL-DEPENDENT — sharp on the 12l single-QK model,
only partial (ARI ~0.17, effect at 3/5 heads) on the 500M two-QK model. Closes LOG Q1: OV factorization
does not cleanly generalize across models; report it as 12l-specific until further heads/models checked.

---

## 2026-07-11 — tick 30 (autonomous): WHY two QK matrices? = SIGNED + CONJUNCTIVE attention (mechanistic, all 162 heads).

The two-QK squared pattern p_qk=(q1·k1)(q2·k2) is used RAW (masked, no softmax/abs/row-norm), so unlike the
single-QK squared pattern (q·k)²≥0 it can be NEGATIVE. Measured on real text, all 162 heads of the 500M model
(pattern construction is the one that matches model CE to 2e-4, tick 27; off-diagonal entries):

- **frac_neg_entries = 0.500±0.125** [0.10, 0.95] — half of attention entries are negative on average.
- **frac_neg_mass = 0.490±0.274** [0.008, 0.986] — ~half the attention MASS is subtractive on average; heads
  span near-zero (0.008, almost pure positive mixing) to near-total (0.986, almost pure suppression).
  A squared single-QK head gives 0 on both ⇒ every bit of this is capability the 2nd matrix adds.
- **Conjunctive sharpening UNIVERSAL:** participation ratio PR(product)=0.319 vs PR(s1)=0.594, PR(s2)=0.600;
  PR(product) < min(PR(s1),PR(s2)) at **100% of heads**. The product concentrates on fewer keys than either
  factor — an AND that sharpens selection neither score achieves alone.

**Two capabilities strictly beyond a single squared score: (i) SIGNED (anti-copy) attention, (ii) conjunctive
sharpening.** This is what the two-QK architecture is FOR. Honest caveat: frac_neg_mass does NOT track the
tick-27/28 clustering gain (L6H3 gains at neg_mass 0.16; L8H3 gains at 0.70; high-neg heads L1H1/L9H3 were
low-gain) — signed/conjunctive capability is a GENERAL architectural fact, separate from the head-dependent
clustering gain. Partly answers LOG Q2 (q1/q2 specialization): the two near-orthogonal query matrices exist
so their product can be signed and peaked, not (as far as this shows) a clean content-vs-position split.

---

## 2026-07-12 — tick 31 (autonomous): is signed attention LOAD-BEARING? Causal ablation. TEMPERS tick 30.

tick 30 showed signed attention is pervasive (~49% of mass subtractive). tick 31 asks if it's USED, by
causally ablating the negative pattern. Pattern construction matches model CE 2e-4. clean CE 3.79.

GLOBAL (all heads at once) — uninformative, honestly flagged:
  clamp_neg (relu pattern, remove anti-copy)   +1.77 nats
  flip_neg  (|pattern|, anti-copy -> copy)      +4.78 nats
  drop_ctrl (matched positive-mass removed)     +3.64±0.01 nats
  The matched control does MORE damage than clamp_neg ⇒ zeroing pattern entries across all heads is
  catastrophic regardless of sign; global ablation can't isolate the signed structure. (flip_neg > drop_ctrl
  does show the model is not sign-agnostic, but all three are "model destroyed" magnitudes.)

PER-HEAD clamp_neg damage (the clean measure) — signed attention is CONCENTRATED, not diffuse:
  mean 0.0014, max 0.0565; only 7/162 heads > 0.01 nat, 1 head > 0.05. Most heads' anti-copy is NOT
  individually load-bearing; layers 14 & 16 have NEGATIVE mean damage (removing anti-copy slightly helps).
  Top signed-attn heads: L1H1 +0.0565, L2H5 +0.0280, L8H3 +0.0259, L9H7 +0.0200, L11H5 +0.0138, L5H6 +0.0111.

**Net (tick 31, tempering tick 30):** signed attention is BROADLY PRESENT (tick 30: half the mass subtractive)
but FUNCTIONALLY CONCENTRATED — individually load-bearing at only ~7 heads, led by **L1H1** (+0.057 damage,
and also the highest subtractive-mass head in tick 30 at 0.92 — a coherent genuine anti-copy head). L8H3 &
L11H5 (two-QK causal heads) also appear. Pervasive ≠ important: most of the negative mass is not
individually critical. Standing-rules note: the matched-mass control killed the global claim; the per-head
result is what survives.

---

## 2026-07-12 — tick 32 (autonomous, priority-2): HIERARCHICAL expert geometry — J metric recovers the TREE at both levels.

Completes the geometry item (ring was S10 circ-corr 0.999; hierarchy was shown in the HTML but never
measured). 2-level tree: 3 coarse orthogonal operators × 4 leaves each (leaf = coarse + 0.35·perturbation),
12 leaves, d_c=16, eps=0.05. Content GMM cross-cuts mechanism. Object = content-restricted J (=A_leaf here),
|cos| per S5. 5 seeds over sampling+kmeans.

| metric | coarse ARI | fine ARI | tree-ρ (Spearman, true tree dist {0,1,2} vs |cos|-dist) |
|---|---|---|---|
| **content-restricted J \|cos\|** | **1.000±0.000** | **1.000±0.000** | **0.831±0.003** |
| input x [content] (control) | −0.000±0.001 | +0.001±0.001 | −0.006±0.015 |
| matched-dim random proj of J | 0.957±0.087 | 1.000±0.000 | 0.831±0.003 |
| shuffled-label chance | −0.000 | +0.000 | 0.000 |

3-level mean |cos|-distance ordering (hierarchy respected iff same-leaf < same-coarse < diff-coarse):
  content-restricted J: same-leaf **0.000** < same-coarse **0.108** < diff-coarse **0.949** — textbook tree.
  input x: 0.743 / 0.740 / 0.739 — FLAT, recovers nothing (content cross-cuts, as designed).

**Net (tick 32):** the |cos| metric recovers a planted 2-level hierarchy at BOTH levels (coarse+fine ARI 1.0)
with a graded tree-distance structure (ρ=0.83; same-leaf≪same-coarse≪diff-coarse) — the hierarchical analog
of S10's ring. HONEST CAVEAT: for this hand-built layer content-restricted J EQUALS the planted A_leaf, so
a random linear readout of J recovers the tree too (0.96/1.0) — that control does NOT discriminate here
(the tree is robust in A). The control that genuinely CAN fail is input-x, and it correctly recovers nothing.
So the claim is "the metric reads a planted hierarchy" (like the ring), not "only J among readouts can." A
non-trivial version would train a bilinear MLP with hierarchical modules (à la DGP-D) so restricted-J ≠ the
planted matrix — noted as an extension, not run.

---

## 2026-07-12 — tick 33 (Logan): full attention module as ONE tensor + the VK read. How good, what cost.

Squared attention has NO softmax ⇒ the head is multilinear and IS one tensor. Per query (head space):
  z_q = Σ_k (x_q^T W_QK1 x_k)(x_q^T W_QK2 x_k)(W_V x_k) = ⟨x_q⊗x_q⊗(Σ_k x_k^⊗3), T⟩,  T=W_QK1⊗W_QK2⊗W_V.

COST:
  - DENSE T (legs z[hd], q[d,d], k[d,d,d]) = hd·d^5 = 2.60e17 floats ≈ **1.04 EB** — intractable.
  - FACTORED T = W_QK1⊗W_QK2⊗W_OV (three d×d circuit matrices, each rank≤hd) = 3d² ≈ **16 MB**, contracted
    at forward cost. So "one tensor, contractable" = YES, only in factored/CP form (never materialize dense).
    This is the same point as the impossibility-note: tensor form ≡ CP; the value is that CP is cheap.

HOW GOOD (VK = the value contracted through key-selectivity = realized read z_q=Σ_k s_qk v_k, no softmax).
Cluster queries by progressively more of the contraction; judge all by the SAME non-circular intervention
(joint override of q1,q2,v by the cluster rep; cluster on a readout, intervene on inputs). 5 seeds:
| head | query [q1;q2] | value v (VK values) | FULL read z (VK) | resid x | random |
|---|---|---|---|---|---|
| L6H3 | +0.0014±.0005 | +0.0024±.0025 | −0.0000±.0017 | +0.0010±.0019 | +0.0002 |
| L8H3 | −0.0004±.0003 | +0.0007±.0023 | +0.0005±.0003 | +0.0002±.0008 | −0.0000 |

**The FULL contracted read z_q does NOT beat its parts — it's null.** Grouping queries by z_q=Σ_k s_qk v_k
mixes the query's operation with the specific CONTEXT content present, so it is contaminated exactly like
the full per-token Jacobian (tick 23, ≈ random) — the recurring program theme. The useful objects stay the
isolated weight-derived readouts (query=WHERE, value=WHAT), which analyze the FACTORS of T, not assembled T.
CAVEAT: the joint (q1,q2,v) override has low dynamic range (all features weak here, even query which was
+0.0065 under isolated override in tick 27), so this is "z_q doesn't beat the parts / is null," not a precise
ranking. The conceptual point is robust: contracting the full module reintroduces context contamination.

**Net (tick 33):** the full attention module is exactly one contractable tensor (factored, 16 MB, forward
cost; dense 1 EB intractable) BECAUSE squared attention has no softmax — but clustering by the assembled
per-query object (z_q / VK read) is null vs its parts; the leverage is in the weight FACTORS, not the tensor.

---

## 2026-07-12 — tick 34 (Logan): use the FACTORED structure — include the OVK factor readouts, not just Q.

Logan: two-Q helped (tick 27); can we also fold in the OVK parts as another method? Use the FACTOR readouts
of T=W_QK1⊗W_QK2⊗W_OV as per-token features (q1=Wq1 x, q2, k1=Wk1 x, k2, OVv=W_OV x) — weight-derived, NO
context sum, so not contaminated like the tick-33 contracted read z_q. L2-normalize each, concat, cluster.

First attempt used the joint (q1,q2,v) override (tick-33 style) — UNDERPOWERED: all features collapsed to
~+0.001-0.003, indistinguishable (value override kills dynamic range). Rerun with the SENSITIVE tick-27
intervention (override q1,q2 ONLY), vary only the clustering feature, 5 seeds, 3160 tokens:

| feature | L6H3 | L8H3 | L11H5 | L0H2 |
|---|---|---|---|---|
| q1 only | +0.0060 | −0.0001 | +0.0041 | +0.011 |
| [q1;q2] two-Q | +0.0058 | +0.0001 | +0.0037 | +0.008 |
| **[q1;q2; OVv]** | **+0.0071** | +0.0006 | +0.0036 | +0.012 |
| [q1;q2;k1;k2;OVv] full OVK | +0.0058 | −0.0003 | +0.0032 | +0.003 |
| [k1;k2;OVv] source-side | +0.0049 | −0.0005 | +0.0021 | +0.016 |
| residual x / random | +0.0035 / 0 | ~0 | +0.0013 / 0 | +0.041±.049 / 0 |

**NOT a robust win.** One signal: at the cleanest head L6H3, adding the OV WRITE direction OVv to the query
feature helps modestly (+0.0071 vs [q1;q2] +0.0058, ~1.5σ) — the write direction carries a little
complementary info about the query's function. BUT the KEY readouts k1,k2 DILUTE back to baseline (full OVK
+0.0058) — sensible: k1,k2 describe the token's role as a KEY (source-side), orthogonal to its query
(destination) role, so they add noise to a query clustering. Head-dependent: L11H5 q1-alone best; L8H3 null
(query not causal there); L0H2 too noisy (residual-x ±0.049 — a few high-leverage tokens).

**Net (tick 34):** the factored structure is cheap and well-defined to use, but concatenating OVK readouts
does NOT reliably beat the query factors. At best the OV write adds a little at one clean head; the key
factors hurt. Consistent with the standing finding: WHERE (query) is the causal handle, OV a separate weaker
axis (tick 26/29), not a complementary boost to query clustering. The robust factored win remains the
two-near-orthogonal-query-matrices (tick 27), not the wider OVK concat.

---

## 2026-07-12 — tick 35 (Logan): do the ORIGINAL method (secant y@x^-1) for attention — and the unification.

Logan noticed ticks 27-34 clustered ACTIVATIONS (readouts), not the OPERATOR (Jacobian/secant y x^T) of the
original method. Ran the faithful secant M_q = z_q x_q^T (cos = cos_x·cos_z), x reduced to top-256 PCA,
z_q = head output. Same sensitive intervention (override q1,q2), 5 seeds.

| feature | L6H3 | L11H5 | L0H2 (unreliable) |
|---|---|---|---|
| x_q (raw activation) | +0.0039±.0008 | +0.0009±.0010 | +0.019±.030 |
| [q1;q2] query readout | **+0.0054±.0010** | **+0.0021±.0013** | +0.009±.004 |
| secant M=z x^T (y@x^-1) | +0.0037±.0009 | +0.0013±.0010 | +0.094±.078 |
| z_q output alone | +0.0013 | −0.0005 | +0.001 |
| random | ~0 | ~0 | 0 |

(L0H2 uninterpretable: ±0.078 std, a few extreme-leverage tokens dominate.)

RESULT: the secant (original operator method) WORKS on attention — beats random and z_q-alone — but is
WEAKER than the query readout at both clean heads. z_q output-alone is near-null (context-contaminated).

**THE UNIFICATION (resolves Logan's concern — the readout is NOT a departure from the Jacobian method):**
Squared attention makes z_q an EXACT bilinear layer in x_q (context fixed): z_q = Σ_k v_k (x_q·a1_k)(x_q·a2_k),
a1_k=A1 x_k, a2_k=A2 x_k — i.e. L-rows a1_k, R-rows a2_k, D-cols v_k. Its two GATE values are exactly q1·k1
and q2·k2. So clustering by q1,q2 = clustering by the GATE DIRECTIONS of the per-query bilinear operator =
the content-restricted Jacobian object (the one that won on DGP-D/S8). The full secant M=z x^T re-adds the
context-contaminated output z_q (the cos_z factor drags it below the query readout) — exactly the full-J
contamination of the impossibility theorem. So the program lesson holds and unifies across MLP and attention:
**the useful object is the RESTRICTED operator (its gate/selection factor), not the full operator/secant.**
MLP → content-restricted J; attention → query readout q1,q2 (= the gate factor of the per-query bilinear op).
Query readout > raw activation x_q > full secant ≈ contaminated — all consistent.

---

## 2026-07-12 — tick 36 (autonomous): where is the attention-Jacobian contamination? PARTIAL confirmation + a caveat.

Backing tick 35: is the full per-query J_q contaminated toward the CONTEXT/OUTPUT (values) rather than the
query (selection)? 12l L6H0, J_q=∂z_q/∂x_q via autodiff on 320 sampled tokens, cluster by {J_q, query qtil,
output z_q, residual x}, K=8, 5 seeds.

ARI alignment of J_q-clusters (the informative part):
  ARI(J_q, query qtil) = +0.045   ARI(J_q, output z_q) = +0.082   ARI(J_q, residual x) = +0.069
→ J_q leans toward output/context (0.082) over query (0.045) — the PREDICTED contamination direction (J_q =
Σ_k (∂p_qk/∂x_q) v_k, a query-covector ⊗ context-VALUE sum). But ALL alignments are LOW (0.04–0.08): J_q
doesn't strongly track query, output, OR input — it's mostly its own noise-like structure, reinforcing
tick 23's "J_q ≈ causally random."

CAVEAT (standing rules — report the confound): the causal swap-within here was UNINFORMATIVE (all ~0,
including query which was +0.009 in tick 23/25) because sampling only 320 tokens for tractable Jacobians made
the query-override intervention sparse → no CE dynamic range. So the causal anchor for "J_q ≈ random" remains
tick 23 (full-position override); this tick adds only the ARI evidence for the contamination DIRECTION.

**Net (tick 36):** WEAK/partial confirmation that the attention Jacobian's clustering leans toward the
context/output (values) over the query — consistent with tick 35's account (J_q re-adds the context-value
factor) — but all alignments are low (J_q ≈ own/random structure) and the causal sub-test was underpowered
by Jacobian sampling. Not a headline; tick 35's unification stands on the query-readout vs secant comparison,
not on this.

---

## 2026-07-12 — tick 37 (Logan): SAEs on the Jacobian object vs on activations — MECHANISM recovery.

Question: does an SAE on the Jacobian object recover MECHANISM features an activation-SAE misses? Ground-truth
gated-superposition toy (G=6 experts × C=8 content clusters, content processed by an expert per gate; gate tiny
in x). Matched TopK SAEs (m=64, k=4, 5 seeds) on x vs Jacobian-derived objects. Metric: activity-weighted
per-latent PURITY on gate (mechanism) and content labels. chance gate 0.167, content 0.125.

| SAE trained on | gate-purity | content-purity | recon R² | recovers |
|---|---|---|---|---|
| x (normal SAE) | 0.177±.000 | 0.808±.041 | 0.98 | CONTENT |
| J(x) raw | 0.174±.001 | 0.790±.021 | 0.98 | CONTENT |
| J(x) normalized | 0.174±.001 | 0.784±.031 | 0.97 | CONTENT |
| **restricted-J norm** | **0.965±.033** | **0.125±.000** | 1.00 | **MECHANISM** |
| G^½ x norm | 0.178±.001 | 0.792±.035 | 0.98 | CONTENT |

**YES — but only the RESTRICTED Jacobian.** The restricted-J (content columns) SAE recovers the gate/mechanism
(purity 0.965) and is at EXACTLY chance on content (0.125) — the mirror image of the activation SAE (content
0.808, gate at chance). Neither sees the other's features. The naive full/normalized J, and the G-embedding,
all FAIL (recover content) — the impossibility theorem: the full J's mechanism signal is the O(1/eps)
gate-derivative block which is GATE-INDEPENDENT, and content dominates the variance an MSE-SAE optimizes.
This is the SAE/superposition analog of S8 (restricted-J k-means → modules ARI 1.0).

CAVEATS (honest): (1) restricted-J here = A_g (constant per gate for this pure toy → recon R²=1.0), a
relatively easy recovery; a superposition-of-OPERATORS toy (each token fires a sparse set of experts) would be
a harder test — the contrast (0.965 vs ~chance everywhere else) is decisive but the absolute difficulty is low.
(2) Getting the restricted-J on REAL models needs the content/gate split (architectural) OR the weights-only
G-top-projection recipe (DGP-E); real bilinear MLPs are near-isotropic (S11) so transfer is the open question.
(3) For attention the mechanism object is the query readout (tick 35) — an SAE on q1,q2 is the analog to try.

**Net (tick 37):** SAEs on the mechanism (restricted-Jacobian) object DO recover mechanism features that
activation-SAEs miss (0.965 vs 0.177 gate-purity, mirror-image content) — a clean positive on the toy — but it
must be the RESTRICTED Jacobian; naive J-SAE = x-SAE (both content). Next: harder superposition toy, then real
attention (SAE on q1,q2) and the weights-only-restricted object on a real MLP.

---

## 2026-07-12 — tick 38 (autonomous): SAE on Jacobian — NON-DEGENERATE superposition (fixes tick-37 caveat).

tick 37's restricted-J was degenerate (one gate/token, =A_g). Here: each token fires a SPARSE SET (3 of 10)
of operators; operator activation is a NONLINEAR (quadratic top-k) function of content. Matched TopK SAEs
(m=96, k=6, 5 seeds). Metrics: per-latent operator-purity (chance 0.300) and MMCS of atoms to true operator
dict {vec(A_i)}.

| SAE on | op-purity | MMCS→operators | recon R² |
|---|---|---|---|
| x (content, normal SAE) | 0.743±.039 | n/a (x-space) | 0.96 |
| J_full norm | 0.818±.028 | 0.586 | 0.93 |
| **J_op norm (mechanism)** | **0.898±.056** | **0.926** | 1.00 |

1. **DICTIONARY recovery is the decisive win**: J_op-SAE recovers the actual operator matrices at MMCS 0.926
   (near-perfect) in genuine superposition (3/10 active, NON-degenerate) — which the activation SAE
   fundamentally CANNOT (operators aren't in activation space, n/a). For a dictionary of WHAT each feature
   COMPUTES you must use the Jacobian object; the activation SAE only gives activation directions.
2. **Restriction matters, quantified for SAEs**: J_full MMCS 0.586 << J_op 0.926 — the gate-derivative
   contamination measurably degrades operator recovery (impossibility theorem, SAE version).
3. **Honest caveat on ACTIVE-SET recovery**: x-SAE op-purity 0.743 is well above chance 0.300 (not the
   tick-37 mirror-image null) — because the quadratic gate here CORRELATES with content, so content features
   partially predict which operator fires. So J_op beats x only modestly on active-set (0.898 vs 0.743); the
   decisive gap is the MMCS dictionary recovery, not active-set prediction. A fully content-decoupled gate
   would sharpen the active-set contrast.

**Net (tick 38):** non-degenerate confirmation — the mechanism-object (operator-block Jacobian) SAE recovers
the operator DICTIONARY at MMCS 0.926 in superposition, which activation-SAEs cannot access at all; the
restriction is necessary (J_full 0.586). This is the real, non-trivial version of tick 37's positive.
Remaining: real-model tests (attention SAE on q1,q2; weights-only-restricted object on a real bilinear MLP).

---

## 2026-07-12 — tick 39 (autonomous): real-model attention SAE — mechanism object vs activations. HEAD-DEPENDENT.

Does the toy SAE result (ticks 37-38: SAE on mechanism object recovers mechanism) transfer to a real attention
head? No feature labels on a real LLM, so judged by the CAUSAL swap-within test (label-free, priority-4). Train
matched TopK SAEs (m=32,k=4) on the query readout q=[q1;q2] (mechanism object) and on the residual x; group by
argmax feature; query-override swap-within. 500M two-QK, 4 causal heads, 5 seeds. k-means K=8 as reference.

| head | SAE-q | SAE-x | verdict | k-means q | k-means x |
|---|---|---|---|---|---|
| L6H3 | +0.0070±.0010 | +0.0032±.0011 | query 2.2× | +0.0057 | +0.0012 |
| L0H2 | +0.0031±.0009 | +0.0021±.0021 | query 1.5× | +0.0072 | +0.0069 |
| L11H5 | +0.0031±.0004 | +0.0010±.0009 | query 3× | +0.0009 | +0.0004 |
| L8H3 | +0.0008±.0008 | +0.0041±.0009 | **residual 5× ✗** | +0.0019 | +0.0000 |
| random | ~0 | | | | |

**HEAD-DEPENDENT, not a robust win.** SAE-on-query beats SAE-on-residual at 3/4 heads (modest 1.5–3×) but
REVERSES clearly at L8H3 (residual 5×). k-means query ≥ residual at all 4 (query is the causal object,
consistent with the whole attention arc), but the SAE version is noisier/head-dependent. The single-head L6H3
headline (query 2.2×) did NOT generalize cleanly — caught by running 4 heads (the tick-26 single-head-outlier
lesson). Caveats: SAE-vs-kmeans confounded by group count (32 vs 8); "causally distinct groups" ≠ interpretable
features (no labels — cannot claim monosemanticity); one model.

**Net (tick 39):** the mechanism-object SAE advantage is CLEAN in toys with ground truth (37-38) but only
DIRECTIONAL and HEAD-DEPENDENT on the real attention head (query-SAE wins 3/4, modest, L8H3 reverses) —
consistent with the program's recurring pattern (real-model advantages are fragile/head-dependent; S11 MLP null,
attention head-dependence throughout). Honest: SAEs on the mechanism object are BETTER on average but not
reliably per-head on this real model.

---

## 2026-07-12 — tick 41 (autonomous, closes PRIORITY-1): G-top projection does NOT transfer to real bilinear MLPs.

The one never-run original-priority item. DGP-E recipe (project J's columns off the top-r eigenvectors of G =
the gate subspace) beat controls on the toy (ARI 0.654 vs 0.388). Does it help on the real block2-dense-seed0
MLPs? Surrogate held-out R² (per-cluster ridge h→y; ridge because plain lstsq blew up to R²≈−20 at MLP#0 — the
tick-8 instability), K=8, 5 seeds, G_rand spectrum-matched control on every cell.

| metric | MLP#0 (eff-rank 111/128) | MLP#1 (eff-rank 100/128) |
|---|---|---|
| raw h | +0.169±.091 | +0.737±.006 |
| G-embed | +0.299±.129 | +0.728±.013 |
| **G_rand (control)** | **+0.291±.104** | **+0.742±.008** |
| G-top-proj (best r) | +0.306±.099 | +0.737±.021 |
| global (1 map) | −0.213 | +0.512 |
| best-proj vs G_rand | **+0.015 (within ±0.10 noise)** | **−0.005** |

**NULL — recipe does NOT transfer.** On MLP#0 the projection beats raw h (+0.137) but the spectrum-matched
G_rand control explains it ENTIRELY (0.291 vs 0.306) — the lift is spectrum/whitening, NOT the gate
eigenvectors. On MLP#1 no lift at all. So the G's eigenVECTORS carry no gate-subspace advantage over a
spectrum-matched random-eigenvector metric on either real MLP — confirming S11 (real MLPs near-isotropic;
eff-rank 100–111/128). The DGP-E toy success requires a real gate subspace that real trained MLPs lack.
Standing rule earned its keep: without G_rand, MLP#0's +0.137 over raw-h reads as a win.

**Net (tick 41):** closes priority-1. The weights-only G-top-projection recipe is validated on toys (DGP-E)
but provides NO real-MLP advantage over the spectrum-matched control — the real-MLP null (S11) holds for the
projection recipe too, and is now specifically controlled. Methodological: ridge required (tick-8 lesson;
plain-lstsq surrogate R²≈−20 at MLP#0).

---

## 2026-07-12 — tick 42 (Logan): BILINEAR SAE on the SECANT (Dooms-style, two different inputs x^+ and y). WORKS.

Logan's architecture: reconstruct the per-token secant M=y x^+ as a sparse sum of RANK-1 atoms p_i q_i^T;
bilinear encoder z_i=(q_i·x^+)(p_i·y)=<M,p_i q_i^T>; TopK; optional mixer after TopK; Dooms loss-expansion so
the d×d secant is NEVER instantiated. His uncertainty: does it work with DIFFERENT inputs (x≠y, vs Dooms' x=x)?

Toy (gated superposition, N=10 ops, 3 active/token, as jacsae2). 5 seeds, m=64, k=6.
- **Loss-expansion VERIFIED**: explicit d×d ||M-Mhat||^2 == expanded (inner-product) form, rel err 0 / 1.6e-7.
- op-purity (feature fires when one operator active; chance 0.300):
  - mixer=False: **0.832±0.030**   mixer=True: 0.827±0.029   (tick-38 explicit-J SAE: 0.898)

**RESULT: the architecture WORKS — different inputs are fine.** Op-purity 0.83 >> chance 0.30, close to the
explicit-J SAE (0.90). The asymmetric bilinear form B(x^+,y) recovers operator-selective features. The
loss-expansion is exact ⇒ d×d secant never formed ⇒ FEASIBLE on large d (unblocks the tick-40 real-MLP wall).
MIXER: no effect on op-purity (0.827 vs 0.832) but slightly lower reconstruction loss (0.094 vs 0.107) —
consistent with the rank-1 picture: each atom p_i q_i^T is an input-dir→output-dir SLICE already belonging to
one operator (so selectivity needs no re-bundling); the mixer only helps RECONSTRUCT the full operator.
Caveats: slightly below explicit-J (0.83 vs 0.90; secant is rank-1/weaker per S8); op-purity is feature
selectivity, not full-operator dictionary recovery (rank-1 atoms can't do that individually).

**Net (tick 42):** Logan's bilinear-SAE-on-secant is validated on the toy and, crucially, is the FEASIBLE
(expanded-loss) route to an operator-SAE on real models where the explicit-J SAE was 1.3M-dim/token infeasible.
Next: run it on the real MLP's (x,y) — the previously-blocked cell.

---

## 2026-07-12 — tick 43 (Logan): bilinear-secant SAE on REAL bilinear MLPs — NON-NULL positive + no outliers.

Ran Logan's bilinear-secant SAE (tick 42, expanded loss = feasible) on block2-dense-seed0 MLPs. x=post-norm
input h, y=MLP output, secant M=y h^+; sparse rank-1 dictionary (m=256, k=16), 3 seeds.

| | secant recon FVU (trained / random-atoms) | functional FVU (M̂h vs y) |
|---|---|---|
| MLP#0 (layers.1) | 0.222±.000 / 0.991 | 0.147 |
| MLP#1 (layers.3) | 0.352±.001 / 0.992 | 0.240 |

**WORKS — the operator-SAE is non-null on the real MLP.** Sparse rank-1 operator atoms reconstruct the
per-token secant far better than random atoms (0.22–0.35 vs 0.99), and the reconstructed operator applied to h
reproduces 76–85% of the MLP output. The MLP's local maps have shared sparse rank-1 structure a dictionary
recovers; expanded loss makes it feasible (scales to d=1152).

**KEY NUANCE: reconstruction succeeds where CLUSTERING was null.** S11/tick-41: the Jacobian object gives no
clustering advantage on real MLPs (near-isotropic, no gate subspace). But a sparse operator DICTIONARY still
reconstructs the secants — near-isotropy kills clustering separation, not sparse operator reconstruction.
FIRST non-null real-MLP result for the operator object; enabled by Logan's bilinear-SAE architecture.

**OUTLIERS (Logan's ask): none to fold out.** block2 is post-RMSNorm — 0 dims with kurtosis>20, max/median
dim-ratio ~8, top-1% tokens hold only ~2% of secant mass (anti-concentrated), per-token FVU uncorrelated with
token norm (corr ≈ 0.00 / −0.06). Error is spread uniformly, not stuck on outlier tokens/dims. Matches the S11
note (these models lack LLM.int8-style outliers).

Caveats: FVU 0.22 decent not excellent; functional FVU is a consistency check (encoder sees y), not prediction;
a dense low-rank baseline would sharpen how much sparsity buys; block2 is tiny (1.9M) — the real outlier/scale
test is the 500M MLP (d=1152), now runnable via the expanded loss.

---

## 2026-07-12 — tick 44 (autonomous): bilinear-secant SAE on the 500M MLPs (d=1152) — scales, layer-dependent, no outlier pathology.

Continues tick 43 (block2 d=128) to the 500M gpt2-bilinear-sqrd-attn-18l. Expanded loss ⇒ the 1152×1152 secant
is NEVER formed. m=512, k=32, 12288 tokens, 3 seeds. Layers 6 & 12.

| layer | secant recon FVU (trained / random) | functional FVU (M̂h vs y) | y dim-ratio | kurtosis>20 | corr(FVU,‖M‖²) |
|---|---|---|---|---|---|
| 6 | 0.117±.001 / 1.000 | 0.074 | 26 | 0/1152 | −0.07 |
| 12 | 0.430±.001 / 1.000 | 0.343 | 20 | 0/1152 | −0.43 |

1. **Large-d feasibility CONFIRMED** — operator-SAE ran at d=1152, secant never materialized. The expanded-loss
   trick delivers the whole point (this was the blocked cell in tick 40).
2. **Non-null positive, LAYER-DEPENDENT.** Layer 6 strong: FVU 0.117 (88% of secant reconstructed with 32 sparse
   rank-1 operators), functional 0.074 ⇒ recovered operators reproduce 93% of MLP output. Layer 12 weaker
   (0.430) — early MLPs far more sparse-operator-structured than late (late = higher-rank/distributed).
3. **OUTLIERS: mild and well-captured, not a problem.** Larger y dim-range than block2 (max/median 20–26 vs ~8)
   so mild outlier dims exist, but 0 dims with extreme kurtosis (post-RMSNorm, no LLM.int8 spikes). corr(per-tok
   FVU, ‖M‖²) NEGATIVE (−0.07/−0.43): high-norm/outlier tokens reconstructed BETTER, residual error is in the
   low-norm bulk. The SAE spends capacity correctly on outliers — no degradation.

**Net (tick 44):** the bilinear-secant operator-SAE is a feasible, non-null, scale-confirmed real-MLP result —
strong on early layers (93% of output from 32 sparse operators), weaker late, with mild but well-handled
outliers. Caveats carried from tick 43: functional FVU is a consistency check (encoder sees y); a dense
low-rank baseline (PCA at matched rank) is the remaining rigor item to show sparsity specifically helps.

---

## 2026-07-12 — tick 45 (autonomous): the control that could kill tick 43/44 — sparse operator-dict vs dense low-rank. PASSES.

Is the bilinear-secant SAE genuinely SPARSE-adaptive or just globally LOW-RANK (in which case a small fixed
basis would tie it)? Control: dense rank-R (m=R, all active) sweep vs sparse SAE, secant FVU, 3 seeds (2 @500M).

| | dense rank-R (all active) | SPARSE |
|---|---|---|
| block2 MLP#0 | R4:.886 R8:.798 R16:.714 R32:.593 R64:.440 | m256 k16: **0.222** |
| block2 MLP#1 | R4:.794 R8:.725 R16:.654 R32:.577 R64:.478 | m256 k16: **0.353** |
| 500M layer-6 | R8:.171 R16:.160 R32:.149 R64:.134 | m512 k32: **0.121** |

**PASSES — sparsity is real.** Sparse beats dense at matched-or-HIGHER active count everywhere:
- block2 MLP#0 DECISIVE: sparse-16 (0.222) beats dense-16 (0.714) 3×, beats dense-64 (4× the active count,
  0.440) by 2×. The secant collection is NOT globally low-rank — needs a large operator dictionary used
  sparsely. block2 MLP#1 same (sparse-16 0.353 < dense-64 0.478).
- 500M layer-6 HONEST CAVEAT: sparse-32 (0.121) still beats dense-64 (0.134) but only modestly, and dense
  saturates fast (rank-8 already 0.171) ⇒ this layer's secant is MOSTLY low-rank (~87% at rank-8) with a small
  genuine sparse gain on top. The operator dictionary is real here but thinner than on block2.

**Net (tick 45):** the tick-43/44 operator-SAE result survives its strongest control — the sparse operator
dictionary genuinely beats dense low-rank (strongly on the small model, modestly on the 500M layer where maps
are more intrinsically low-rank). Sparsity/overcompleteness is not an artifact of low-rank compression.

---

## 2026-07-12 — tick 46 (autonomous): bilinear-secant SAE across all 18 layers of the 500M — depth profile.

Extends tick 44 (L6,L12) to the whole model. Per layer: sparse-SAE (m=512,k=32) secant FVU + functional FVU +
sparsity margin (dense rank-32 minus sparse). 2 seeds (tick-44 seed-variance ±0.001; noted). 12288 tokens.

| L | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sparse FVU | .24 | .34 | .53 | .59 | .40 | .43 | **.12** | .41 | .45 | .42 | .53 | .37 | .44 | .41 | .48 | .23 | .16 | .16 |
| margin (dense−sparse) | **+.26** | .18 | .20 | .19 | .13 | .17 | .03 | .16 | **.21** | .16 | **.23** | .13 | .18 | .16 | .18 | .08 | .06 | .12 |

mean sparse FVU 0.37; mean margin +0.16 (sparse beats dense-32 at EVERY layer).

FINDINGS:
1. **Non-monotone (U-ish) reconstructibility**, not early-good/late-bad. Best at L6 (0.12) and late L16–17 (0.16,
   functional FVU 0.05 ⇒ ~95% of those MLPs' output from 32 sparse operators); WORST in the middle (L2–3, L10 =
   0.53–0.59). Early and late layers are operator-reconstructible; the middle is hardest.
2. **"Low FVU" ≠ "genuinely sparse" — the margin separates them.** The best-FVU layers (L6, L16) have the
   SMALLEST margins (+0.03, +0.06) — easy to reconstruct because LOW-RANK, not sparse. The operator dictionary
   genuinely earns its keep at the LARGE-margin layers (L0 +0.26, L8 +0.21, L10 +0.23), where sparse beats dense
   low-rank substantially despite higher absolute FVU. So "operator dictionary is real" is strongest at L0/L8/L10.
3. Sparse beats dense-32 at every layer ⇒ tick-45 sparsity conclusion holds model-wide.

**Net (tick 46):** the bilinear-secant operator-SAE applies across the whole 500M (sparse everywhere), with a
non-monotone depth profile — reconstructibility peaks at L6 and late layers, but genuine SPARSE (vs low-rank)
operator structure is strongest at L0/L8/L10. Two distinct phenomena: low-rank-easy layers vs sparse-operator
layers. Caveat: 2 seeds (low FVU variance); functional FVU is a consistency check (encoder sees y).

---

## 2026-07-12 — tick 47 (autonomous): does the secant reconstruction need y? — resolves the analysis-vs-transcoder fork.

The bilinear-secant SAE encoder sees the output y ⇒ it's an ANALYSIS tool (decomposes a known map), not a
predictor. Test: h-ONLY bilinear encoder z_i=(a_i·h)(b_i·h) (sees only h) vs full (sees y), same secant target
& rank-1 atoms. 500M, 2 seeds.

| layer | full (sees y) FVU | h-only FVU | tick-46 margin | verdict |
|---|---|---|---|---|
| L6 (low-rank) | 0.122 | 0.140 | +0.03 | h-predictable |
| L16 (low-rank) | 0.164 | 0.183 | +0.06 | h-predictable |
| L8 (sparse) | 0.446 | **0.884 (collapses)** | +0.21 | NEEDS y |

**Near-perfect anti-correlation: genuine sparse-operator layers are exactly the ones where the operator is NOT
h-predictable.** Low-rank layers (L6, L16, small margin) are h-predictable — a predictive transcoder is viable
there, BUT the "operators" are just low-rank compression. The high-margin sparse layer (L8) collapses without y
(0.88 ≈ random) — you need the OUTPUT to disambiguate which operator fired.

**Resolves the fork:** the predictive transcoder (encoder from h) works ONLY on the uninteresting low-rank
layers; on the genuinely-sparse layers, the operator-SAE is fundamentally an ANALYSIS tool that requires (h,y).
Connects to the impossibility theorem — where operators are genuinely input-gated, the gate isn't cleanly
input-readable; the output reveals it. Honest confound: h-only is a cubic model of quadratic y (degree
mismatch), but the layer CONTRAST (works L6/L16, fails L8, same architecture) rules out a pure-architecture
artifact — L8's operators are genuinely less h-predictable.

**Net (tick 47):** the bilinear-secant operator-SAE is an analysis tool (needs h,y) precisely where it's most
useful (sparse layers); the predictive-transcoder alternative only works where the structure is low-rank.
This is the natural endpoint of the operator-SAE arc (ticks 42–47): a feasible, controlled, depth-characterized,
architecturally-understood real-model analysis method.

---

## 2026-07-12 — tick 48 (autonomous): are the recovered operators a STABLE dictionary? YES — precursor to interpretation.

Before any interpretation: do independently-trained SAEs find the SAME operators (canonical) or an arbitrary
basis? Rank-1 atom MMCS across 3 seeds; control = MMCS of two random rank-1 dicts (chance). 500M, m=512, k=32.

| layer | #used atoms | dict MMCS (3-seed) | tag |
|---|---|---|---|
| L0 | 508 | **0.790±0.006** | sparse, highest margin |
| L8 | 379 | 0.579±0.009 | sparse, needs-y |
| L6 | 137 | 0.652±0.027 | low-rank |
| chance (random dicts) | — | 0.000 | — |

**STABLE/CANONICAL** — all layers' dictionaries MMCS 0.58–0.79 >> chance 0.000. Independent seeds recover
substantially the same operators ⇒ real recurring structure, not an arbitrary compression basis. Green light
for interpretation. TRIANGULATION: L0 is the cleanest operator layer — highest sparsity margin (tick 46, +0.26)
AND most stable dictionary (0.79). The layer where sparse operators most earn their keep is also the most
canonical. L8 (needs-y sparse) is least stable (0.579) but well above chance — its 379 atoms more seed-dependent.
Caveats: MMCS 0.79 high not 1.0 (partial dictionary, some seed-dependent atoms); 3 seeds (std tiny ±0.006–0.027).

**Net (tick 48):** the bilinear-secant operator dictionaries are stable/canonical — the operators are real,
worth interpreting, most cleanly at L0. Completes the operator-SAE arc's rigor: feasible, non-null, sparse
(not low-rank), depth-mapped, analysis-tool (needs h,y), and now STABLE. Autointerp of L0's 508 operators is
the natural next (needs a labeler / Logan steer).

---

## 2026-07-13 — tick 49 (autonomous): autointerp PILOT of L0 operators — interpretable, control PASSES.

Are the stable L0 operators (tick 48) interpretable? Rigorous pilot: 8 real operator features + 4 random
rank-1 CONTROL features, top-activating tokens presented BLIND (shuffled) to a labeler subagent told to be
skeptical (some are random). Score coherence 0–10. Control that could fail: if real ≈ control, coherence is
illusory.

| | blind scores | mean |
|---|---|---|
| **real operators** (8) | 5,9,9,9,8,9,9,9 | **8.4** |
| random controls (4) | 2,5,3,3 | 3.25 |

**PASSES — L0 operators are interpretable.** Blind labeler separates real (8.4) from control (3.25) decisively
(7/8 real ≥8; 3/4 controls ≤3). The top operators are SINGLE-TOKEN-IDENTITY features: "for", "and", ",",
"the", "is", "I" — exactly what layer 0 should have (token/detokenization features). Honest caveat the control
surfaced: one control scored 5 (accidentally correlated with "give/given/better") — random rank-1 directions
CAN hit a frequent token by chance, so individual autointerp scores are noisy; only the aggregate separation is
reliable. NUANCE: "interpretable" at L0 = token-IDENTITY, not rich computation. The mechanistically-richer
operators would be deeper (higher FVU, need-y per tick 47), so this pilot establishes interpretability at the
easy end.

**Net (tick 49):** the recovered operators are not just stable (tick 48) but interpretable at L0 (blind labeler
beats control 8.4 vs 3.25) — token-identity features. Completes the operator-SAE arc's validation:
feasible → non-null → sparse-not-low-rank → depth-mapped → analysis-tool → stable → interpretable(L0). Deeper-
layer autointerp (richer operators) is the remaining scientific step (needs many labeler calls / Logan steer).

---

## 2026-07-13 — tick 50 (autonomous): deeper-layer autointerp (L8/L10) — stable but largely opaque to surface autointerp.

Extends tick 49 (L0) to the richer deep layers. Same rigorous blind pilot (8 real + 4 random-control features,
skeptical labeler told some are random & to score repeated-doc/memorized low).

| layer | real mean | control mean | separation |
|---|---|---|---|
| L0 (tick 49) | 8.4 | 3.25 | +5.1 (decisive) |
| L8 | 3.75 | 2.75 | +1.0 (weak) |
| L10 | 4.1 | 2.25 | +1.9 (weak-moderate) |

**Deep operators are STABLE (tick 48) but largely NOT token-interpretable.** Blind labeler barely separates real
from control at L8/L10 (+1.0/+1.9) vs L0's +5.1. Most real deep features score 2–5 (varied context, no nameable
token pattern); labeler correctly flagged several as "repeated-doc/memorized" (SAE partly captures recurring
strings like texmf-dist/ paths, not general operators). NOT fully null: a crisp CODE feature at BOTH L8 (SQL/
Django/JS, score 8) and L10 (if/class-defs/.css, score 8) — surface autointerp finds DOMAIN-level deep operators
but not fine computational ones.

Reading: operators are token-interpretable where the layer does token-level work (L0), computational-but-opaque
where it doesn't (L8/L10) — consistent with tick 47 (deep operators are genuinely input-gated computation, not
surface features). CAVEAT: surface autointerp (top-activating tokens) is a WEAK probe for computational features
⇒ "not token-interpretable" ≠ "not meaningful"; a weight-based interpretation (which input→output directions each
operator connects) would be more faithful but is a bigger effort. Small n (8+4, one labeler) — trend clear, not
precise.

**Net (tick 50):** interpretability of the stable operator dictionaries is DEPTH-DEPENDENT — crisp token features
at L0, thin/domain-only at L8/L10. Bounds the interpretability claim honestly: the operator-SAE recovers a stable,
sparse, well-reconstructing dictionary everywhere, but surface-interpretability holds only at the token-level
(shallow) layers. Deep computational operators need a weight-based interpretation method (open).

---

## 2026-07-13 — tick 51 (Logan): loss curve + data-size check — TWO honest corrections to ticks 43–50.

Logan flagged "20k tokens isn't a lot." Ran train-vs-HELD-OUT FVU + token sweep on 500M L8 (m=512, k=32),
192k-token pool, 20k held out. ALSO corrected the MLP-input hook (was block-input normed; now the MLP module's
true input = post-attention normed h, via forward_pre_hook).

| N_train | train FVU | held-out FVU | gap | used atoms |
|---|---|---|---|---|
| 12,000 | 0.464 | 0.532 | +0.068 (mild overfit) | 398 |
| 48,000 | 0.489 | 0.510 | +0.021 | 384 |
| 150,000 | 0.493 | 0.503 | +0.010 (converged) | 392 |

Loss curve: converges by ~step 1500 (6000 steps ample); held-out tracks train at 150k.

**CORRECTION 1 (data):** ticks 43–50 used 12k tokens and reported TRAIN FVU → ~0.05–0.07 OPTIMISTIC. The
faithful held-out converged L8 FVU is ≈0.50, not the reported 0.45. Gap closes to +0.01 at 150k ⇒ the operator
dictionary GENERALIZES (real, not memorized), but absolute FVUs were rosy; correct protocol = ≥50k tokens,
held-out FVU. Should be re-applied to the depth sweep (tick 46) and sparsity-margin control (tick 45).

**CORRECTION 2 (hook):** ticks 43–50 hooked the block INPUT and normed it; the true MLP input is the
post-ATTENTION normed residual. Fixed by hooking blk.mlp's input directly. Also nudges L8 0.45→~0.50. The
(h, y=mlp(h)) pair was always self-consistent, so operators were valid — but sampled from a slightly-off input
distribution. Qualitative conclusions (sparse>low-rank, stable, depth profile, interpretable-shallow) hold;
absolute FVUs shift ~+0.05 and should be re-quoted held-out with the correct hook.

**Net (tick 51):** the operator-SAE dictionary is real and generalizes (held-out gap →0.01 at 150k), but the
reported FVUs were train-on-12k and ~0.05 optimistic, and the MLP-input hook was the block input not the
post-attn input. Both fixed going forward; qualitative story intact.

---

## 2026-07-13 — tick 52 (Logan): BATCH-TopK + lottery-ticket (feature-count sweep) + k-sweep + complexity histogram.

Logan: switch to BatchTopK (was per-token TopK); drop 10x-data; headline = LOTTERY-TICKET (does FVU improve
with total #features?). 500M L8, correct MLP-input hook, held-out FVU on 20k, 59k train. BatchTopK = top (B·k)
over the batch ⇒ variable features/token. Figure: jacclust/bsae_scaled.png.

(A) LOTTERY-TICKET — FVU vs total #features m (avg-k=32):
| m | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| held-out FVU | 0.503 | 0.481 | 0.467 | 0.461 | 0.458 | 0.455 |
→ CONFIRMED: more features → lower FVU, but SATURATING (log-diminishing): 32× features (256→8192) buys only
0.048 FVU, most of it in the first 2–3 doublings, flat past m≈4096. Lottery-ticket effect is real but bounded.

(B) k-sweep (m=4096): k=8→0.562, 16→0.511, 32→0.458, 64→0.398, 128→0.329. Steady ~linear-in-log-k decrease
(sparsity–fidelity tradeoff). L8 needs high k (dense) to get low FVU — it's a hard layer.

(C) BatchTopK complexity split (feats/datapoint at avg-k=32): mean 31.4, **MEDIAN 20**, range **0–459**. Strongly
RIGHT-SKEWED — most datapoints are "easy" (≈20 rank-1 atoms), a long tail are "hard" (up to 459), some use 0.
This is the "split by complexity" BatchTopK enables (per-token adaptive rank), which per-token-TopK (fixed k)
cannot show. Features-per-datapoint = effective rank of that token's reconstructed operator.

NOTE (methodology, carried from tick 51): switched from per-token TopK → BatchTopK; held-out FVU on correct
MLP-input hook. Earlier per-token-TopK m=512 held-out was 0.503; BatchTopK m=512 gives 0.481 (BatchTopK's
adaptivity helps ~0.02).

**Net (tick 52):** lottery-ticket confirmed but saturating (~0.045 FVU over 32× features); clean FVU-vs-k curve;
BatchTopK reveals a heavy right-skewed per-datapoint complexity distribution (median 20, tail to 459, some 0) —
the operators split datapoints by how many atoms they need.

---

## 2026-07-13 — tick 53 (autonomous): CORRECTED depth sweep (held-out, true MLP hook, BatchTopK) — supersedes tick 46.

Re-ran the 18-layer depth profile with the tick-51/52 fixes (correct blk.mlp-input hook, held-out FVU on 20k,
BatchTopK m=1024 k=32). Figure: jacclust/depth_corrected.png.

Per-layer held-out sparse FVU (functional in parens): L0 .451(.31) L1 .331(.27) L2 .621(.48) L3 .634(.49)
L4 .435 L5 .206(.17) L6 .231(.20) L7 .434 L8 .468(.42) L9 .432 L10 .551 L11 .400 L12 .472 L13 .454 L14 .537
L15 .266 L16 .191(.07) L17 .168(.05). mean 0.405.

vs tick-46 (train, wrong hook, mean 0.37): absolute FVUs are ~0.05–0.10 HIGHER (correct hook is harder,
especially early: L0 0.243→0.451; L6 0.12→0.231 so L6 is no longer the singular best). But the QUALITATIVE
findings all HOLD:
- Non-monotone (U-ish): best at LATE layers (L16 .19, L17 .17, functional .05 ⇒ ~95% of output) and mid
  (L5 .21, L6 .23); worst early-mid (L2 .62, L3 .63) and L10/L14. 
- Sparse beats dense-32 at EVERY layer (margin mean +0.148, all positive) — tick 45/46 sparsity holds faithfully.
- TWO REGIMES persist: low-FVU-SMALL-margin = low-rank-easy (L5 +.058, L6 +.046, L15 +.074); high-margin =
  genuinely-sparse operators (L0 +.264, L1 +.217, L8 +.200, L10 +.217). L0/L8/L10 remain the sparse-operator
  layers (matches tick 46 despite the number shift).

**Net (tick 53):** faithful depth profile — mean held-out FVU 0.405 (not the rosy train 0.37); L6 no longer
singular-best (late layers are); but non-monotone shape, sparse>dense everywhere, and the low-rank-vs-sparse
two-regime split (sparse at L0/L8/L10) are all robust to the corrections. Figure sent. This is the number to
quote for the depth profile going forward.

---

## 2026-07-13 — tick 54 (autonomous): what drives the BatchTopK complexity split? — it's MAGNITUDE, not semantics.

tick 52 showed feats/datapoint ranges widely (0–459). Characterized on 500M L8 (BatchTopK m=2048 k=32, 25k
held-out). Figure: jacclust/complexity_split.png.

Spearman corr of #features-per-datapoint with:
| covariate | ρ |
|---|---|
| **secant norm ‖M‖² = ‖y‖²/‖h‖²** | **+0.945** |
| output norm ‖y‖ | +0.945 |
| input norm ‖h‖ | +0.019 (h rms-normed ≈ const) |
| token corpus frequency | +0.181 |

**The complexity split is ALMOST ENTIRELY a MAGNITUDE effect.** #features ≈ f(‖M‖) (ρ 0.945). Mechanical:
BatchTopK keeps the largest ⟨M,atom⟩ ∝ ‖M‖ over the batch, so a high-output token clears the threshold on many
atoms, a low-output token on few. Token identity/frequency barely matter (0.18, 0.02).
- LOW (1–5 feats): single letters / word fragments ('Q','-','i','ar',' M',' the') — mid-word, MLP does little.
- HIGH (top-1%, ≥161): '\n',' to','.',' is',' by', content words — MLP writes a large output.
(Also: with m=2048 there are NO zero-feature tokens; the zeros in tick 52 were a larger-dict threshold artifact.)

**Net (tick 54):** features-per-datapoint tracks "how much the MLP is doing" (output magnitude), NOT a rich
computational-complexity hierarchy — an honest deflation of the tick-52 complexity-split framing. To get a
magnitude-independent complexity measure you'd normalize the secant per token (or threshold on cos not
magnitude) — noted, not run. The operator dictionary is still real (ticks 42–53); this just clarifies that the
per-datapoint feature COUNT is a magnitude proxy, not semantic difficulty.

---

## 2026-07-13 — tick 55 (autonomous): magnitude-free BatchTopK — resolves the complexity split (mostly magnitude, small real residual).

Follow-up to tick 54. Select atoms by NORMALIZED alignment cos(M,atom)=z/||M|| (per-token) instead of raw
magnitude; score per-token FVU (equal weight). 500M L8, m=2048, avg-k=32, 25k held-out. Figure: magfree.png.

| BatchTopK | feats/token | corr(#feat, ‖M‖²) | per-tok FVU |
|---|---|---|---|
| RAW (magnitude) | mean 32, median 22, range 0–367 | +0.943 | 0.516 |
| NORMALIZED (cos) | mean 32, median 32, range 6–67 | **+0.149** | 0.494 |

**The complexity split was ~90% MAGNITUDE.** Normalizing (select by cos) drops corr(#feat,‖M‖) 0.943→0.149 and
collapses the range 0–367 → 6–67 (heavy right-skew → roughly symmetric, mean=median=32). BUT a SMALL genuine
residual remains: magnitude-free feature counts still span 6–67 (~10× spread, residual corr +0.15) — some
operators really do need more atoms to reconstruct their DIRECTION. Per-token FVU marginally better normalized
(0.494 vs 0.516, equal weighting).

**Net (tick 55):** closes the complexity-split thread (52→54→55). The wide per-datapoint feature-count spread
is ~90% a BatchTopK-thresholds-on-magnitude artifact; the real magnitude-free directional-complexity residual
is modest (6–67, ~10×). Honest resolution: there IS operator-complexity structure beyond magnitude, but it's
small — most of the dramatic 0–459 spread was ‖M‖.

---

## 2026-07-13 — tick 56 (Logan): transcoder vs secant-SAE (tied/untied) on the down-projection D.

L8 down-proj D: z=Left(h)*Right(h) (hidden 4608) -> y=Dz (1152), a LINEAR map. Compare transcoder (z->y,
output loss) vs secant-SAE (M=yz+, operator loss), tied vs untied. m=1024 BatchTopK k=32, held-out.

| method | output-FVU | operator-FVU | note |
|---|---|---|---|
| transcoder (z->y) | **0.450** | — | fair z-only predictor |
| secant TIED | 0.802 | 0.877 | sees y |
| secant UNTIED | 0.813 | 0.883 | sees y |

dict MMCS transcoder{d⊗w} vs secant-tied{p⊗q} = **0.133** (different dictionaries).

**1. Transcoder WINS on a linear readout.** At the shared reconstruct-y task, transcoder 0.45 vs secant 0.80
(secant even sees y). Reason: the secant optimizes ‖M−M̂‖² where M=yz⁺ lives in 1152×4608 — reconstructing the
whole operator's action on all 4608 input dirs with 32 rank-1 atoms is far harder than fitting the 1152-dim
output. So the outer-product/secant is INEFFICIENT for a high-input-dim linear readout — reconstructs the whole
operator when you only need the output. Use a transcoder there. The secant/operator framing earns its keep only
where the operator VARIES per token and is the object (bilinear gating), not a constant linear map.

**2. Untying → GOODHART (confirmed).** cross-seed dict stability: TIED 0.708 vs UNTIED 0.428 (crashes), with NO
reconstruction gain (held-out op-FVU 0.877→0.883, slightly WORSE = extra encoder params overfit). So untying
bought a less-canonical, seed-dependent dictionary for nothing. The tied code s_i=⟨M,atom_i⟩ (alignment) is a
genuine regularizer against the Goodhart. Note: TopK fixes k, so it's not L1-shrinkage Goodhart; it's the
canonical-ness/overfit flavor (recon flat/worse, stability down).

**Net (tick 56):** transcoder >> secant for reconstructing a LINEAR readout (secant reconstructs the full
operator wastefully); the two learn different dictionaries (MMCS 0.13); untying the secant SAE Goodharts
(stability 0.71→0.43, no recon gain) — tying regularizes. Secant/operator approach is for per-token-VARYING
operators, not constant linear maps.

---

## 2026-07-13 — tick 57 (autonomous): best sparse dictionary for the FULL bilinear MLP — linear vs bilinear TC vs secant.

Extends tick 56 (linear readout) to the full layer y=D(Lh⊙Rh). 500M L8, h,y=1152-d, m=1024 BatchTopK k=32,
held-out output-FVU.

| dictionary | output-FVU | note |
|---|---|---|
| A linear transcoder (h→y, fair) | 0.683 | linear encoder — can't capture degree-2 |
| **B bilinear transcoder (h→y, fair)** | **0.642** | (a·h)(b·h) features — matches layer; Dooms |
| C secant-SAE tied (sees y) | 0.422 (op-FVU 0.476) | analysis object, not a fair predictor |

1. Among FAIR predictors, the BILINEAR transcoder wins (0.642 vs linear 0.683) — degree-2 features capture the
   gating. But modest, and neither is great: squeezing a 4608-hidden-unit MLP into 1024 feats / 32 active. A
   bilinear TC with m=4608, A=L,B=R,Wd=D IS the MLP exactly; at m=1024 it's a sparse approx.
2. **The tick-56 "secant wasteful" result is INPUT-DIM-dependent.** Down-proj (input 4608 ≫ output 1152): secant
   WORST (0.80, operator M=yz⁺ is huge 1152×4608). Full layer (input 1152 = output): secant COMPETITIVE (0.42,
   though sees y). Rule: the outer-product/secant cost scales with INPUT dim — wasteful for wide readouts
   (input≫output), fine for square maps.

**Net (tick 57):** to PREDICT a bilinear MLP's output sparsely → bilinear transcoder (matches degree-2
structure, size toward hidden dim). The secant/operator-SAE is an analysis tool (needs y); competitive on
square maps, wasteful on wide readouts; its niche is the per-token-VARYING operator, not output reconstruction.
Refines the architecture guidance: transcoder for readouts, tied bilinear transcoder for the MLP, secant for
operator analysis.

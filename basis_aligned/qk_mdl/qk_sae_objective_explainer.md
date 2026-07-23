# The SAE on QK: inputs, outputs, objectives — and where bilinearity enters

*Companion to `qk_ov_lora_explainer.md` (the OV metric derivation). This note specifies
exactly what the dictionaries train on, stage by stage, with shapes.*

Model constants: vocabulary $V = 50304$, heads $N_H = 9$, head dimension $d_h = 128$,
two bilinear branches per head, **no softmax anywhere**.

---

## 1. What the SAE sees (inputs)

**Not activations — a weight-derived table.** For head $h$ and branch $\beta \in \{1,2\}$,
every vocabulary token $t$ has a row

$$X_t \;=\; \big[\, \hat q^{(\beta)}_t \;\big\|\; \hat k^{(\beta)}_t \,\big] \;\in\; \mathbb{R}^{256},$$

its unit-RMS **query factor** and **key factor** for that head-branch, folded exactly from
embedding × $W_q / W_k$ (fold verified to $\sim 10^{-15}$). So each of the
$9 \times 2 = 18$ dictionaries trains on

$$X \in \mathbb{R}^{50304 \times 256},$$

and its **output** is a reconstruction $\hat X$ of the same shape — replacement factor
tables. Data never enters the training inputs; it enters only through frequency weights
in the stage-2 loss and through the held-out audit.

---

## 2. Stage 1 — MSE fit (bilinear-blind initialization)

Standard sparse autoencoder on rows:

| piece | form | shape |
|---|---|---|
| encoder | $z = (X - b)\,W_e^\top$, keep $k$ largest $\lvert z\rvert$ per row | $z \in \mathbb{R}^{V \times n}$ |
| decoder | $\hat X = b + \sum_{\text{kept}} z_c \, D_c$ | $D \in \mathbb{R}^{n \times 256}$, unit-norm rows |
| loss | relative MSE $\;\lVert \hat X - X\rVert^2 / \lVert X - \bar X \rVert^2$, uniform over tokens | scalar |

Each head-branch fits **independently**; nothing here knows attention exists. It's matrix
sketching, used as initialization only.

---

## 3. Stage 2 — the context finetune (bilinear-aware; the objective that matters)

The loss is not row error but **distortion of what the rows compute**, priced by what
downstream reads. Per training step, on a sampled token set ($M = 1024$):

**(a) Build the pattern — this is where bilinear attention enters.** Per branch,
$S^{(\beta)}(a,b) = \hat q_a \cdot \hat k_b / d_h$ (an $M \times M$ block), from both the
reconstruction and the original. The attention pattern is the **product of the two branch
scores** — exactly this model's attention, since there is no softmax:

$$P(a,b) \;=\; S^{(1)}(a,b)\; \cdot\; S^{(2)}(a,b).$$

**(b) Push the pattern error through the OV reader.** With
$u_t = W_o^h W_v^h\, \hat e_t \in \mathbb{R}^{1152}$ (what attending to token $t$ writes
into the residual stream) and $\Delta P = \hat P - P$, charge the context-expected delivered
error over i.i.d. length-$T$ unigram contexts ($T = 512$):

$$\mathbb{E}\lVert e_i \rVert^2 \;=\; T\big(s_i - \lVert \mu_i \rVert^2\big) \;+\; T^2 \lVert \mu_i \rVert^2,
\qquad
\mu_i = \sum_t q_t\, \Delta P(i,t)\, u_t,\quad
s_i = \sum_t q_t\, \Delta P(i,t)^2 \lVert u_t \rVert^2,$$

averaged over query tokens with unigram weights, normalized by the same energy of the
original signal — and (since tick 165) **averaged incoherently over rotary offsets**:
sample 8 offsets $\Delta$, rotate the query factors by the model's own RoPE tables, and use
$T^2\, \mathbb{E}_\Delta \lVert \mu_\Delta \rVert^2$ for the systematic term (the coherent
version $\lVert \mathbb{E}_\Delta \mu_\Delta \rVert^2$ washes out 98.8% of the signal —
the tick-163 diagnosis).

**Consequences of the product structure:**

1. **The two branch dictionaries of a head train jointly** — the loss couples them.
   Branch 1's error at pair $(a,b)$ matters in proportion to branch 2's score there;
   the optimizer exploits this (be sloppy where the other branch gates the pair to ~0).
2. **Errors decompose to squared order**:
   $\Delta P = S^{(1)}\Delta S^{(2)} + S^{(2)}\Delta S^{(1)} + \Delta S^{(1)}\Delta S^{(2)}$ —
   each branch's error is weighted by the *other* branch's signal. No per-branch MSE can
   see this.

---

## 4. The audit (fully bilinear, no approximations)

Reconstructed tables are re-unit-RMS'd, run through the exact score machinery — rotary
via the model's cos/sin difference tables, causal mask, branch product, **unnormalized**
pattern (no softmax to renormalize) — patched into the real forward pass at layer 0, and
measured as held-out ΔCE on 307k FineWeb predictions. The training objective approximates;
the audit never does.

---

## 5. Scorecard: where bilinearity is and isn't accounted for

| component | bilinear-aware? | how |
|---|---|---|
| the folded object itself | ✔ | factor tables *are* the exact bilinear parameters (rank ≤ 128/branch by construction) |
| stage-1 MSE fit | ✘ | rowwise, per branch, uniform |
| stage-2 context objective | ✔ | branch-product pattern, joint per-head training, OV-weighted |
| OMP / linear encoders | ✘ | rowwise on $X$ |
| anchor selection (tick 165) | ✔ | scored by the product-aware † attribution |
| anchor rows themselves | — | exact rows, so trivially correct |
| audit | ✔ | exact machinery end to end |

**Open structural extension** (from the original spec, not yet run): a *joint product
decomposition* — atoms living in the product space of the two branches instead of one
dictionary per branch, so the code can exploit correlations between a token's branch-1 and
branch-2 factors rather than paying for them twice. Composes naturally with the anchor
hybrid.

# Working notes: jointly compressing QK and its OV reader (tick 161)

*A chain-of-thought walkthrough of what `qk_ov_lora.py` actually computes, with the shape of
every object and why each term looks the way it does. Companion to `ov_metric_explainer.md`
(which derived the fixed-reader metric); this note re-derives everything from scratch so it
stands alone.*

Model constants used throughout: vocabulary $V = 50304$, residual width $D = 1152$, heads
$N_H = 9$, head dimension $d_h = 128$, two bilinear branches, no softmax. Sample size in the
training loop $M = 1024$, context length for the expectation $T = 512$.

---

## 1. Start from what the model actually computes at layer 0

The thing we have been compressing all along is the layer-0 QK circuit in its exactly folded
form: four factor tables

$$\hat q^{(1)}, \hat k^{(1)}, \hat q^{(2)}, \hat k^{(2)} \;\in\; \mathbb{R}^{V \times N_H \times d_h} = \mathbb{R}^{50304 \times 9 \times 128},$$

one row per vocabulary token, unit-RMS rows. For head $h$, the pre-rotary score between query
token $a$ and key token $b$ on branch $\beta$ is a plain inner product:

$$S^{(\beta)}_h(a,b) \;=\; \frac{1}{d_h}\,\hat q^{(\beta)}[a,h,:] \cdot \hat k^{(\beta)}[b,h,:] \qquad \text{(scalar)},$$

and because this model has **no softmax**, the attention pattern is just the product of the
two branch scores (times the causal mask):

$$P_h(a,b) \;=\; S^{(1)}_h(a,b)\; S^{(2)}_h(a,b).$$

Shape check: the full score map per head-branch is $V \times V \approx 2.5\text{B}$ entries —
never materialized. Everything below works on the *factor tables* (each $V \times 128$ per
head-branch) and on $M \times M$ sampled sub-blocks of the score map.

The dictionary compresses one head-branch at a time. Concatenate the q-half and k-half rows:

$$X \;=\; [\,\hat q^{(\beta)}[:,h,:] \;\|\; \hat k^{(\beta)}[:,h,:]\,] \;\in\; \mathbb{R}^{V \times 256}.$$

A dictionary is: decoder atoms $D_n \in \mathbb{R}^{n \times 256}$ (unit-norm rows), bias
$b \in \mathbb{R}^{256}$, encoder $W_e \in \mathbb{R}^{n \times 256}$. Encoding is linear
top-$k$:

$$z = (X - b)\,W_e^\top \in \mathbb{R}^{V \times n}, \qquad \text{keep the } k \text{ largest } |z| \text{ per row}, \qquad \hat X = b + \textstyle\sum_{\text{kept}} z_c\, D_n[c] \in \mathbb{R}^{V \times 256}.$$

Description length per head-branch (the frozen convention):
$32\,(n \cdot 256 + 256 + Vk) + Vk\log_2 n$ bits — decoder floats, bias floats, one coefficient
float plus one $\log_2 n$ atom index per stored nonzero. Times 18 head-branches.

So far this is the tick-160 setup. The question of this tick: *the reconstruction error
$\Delta P = \hat P - P$ is judged by what downstream does with it — what if downstream is
allowed to meet us halfway?*

---

## 2. What exactly does OV read? (deriving the reader $u_j$)

Follow one head's output. At query position $i$ in a real sequence with tokens
$t_1, \dots, t_T$:

- value of position $j$: $v_j = W_v^h\, \hat e_{t_j}$, where $\hat e_t \in \mathbb{R}^{1152}$ is
  the unit-RMS embedding row and $W_v^h \in \mathbb{R}^{128 \times 1152}$ is head $h$'s slice
  (rows $128h$ to $128h+127$) of `c_v.weight` $\in \mathbb{R}^{1152 \times 1152}$;
- head output: $o_i = \sum_{j \le i} P_h(i,j)\, v_j \in \mathbb{R}^{128}$;
- written to the residual stream through $W_o^h \in \mathbb{R}^{1152 \times 128}$ (head $h$'s
  column slice of `c_proj.weight`).

Push $W_o^h$ inside the sum and the whole head collapses to one clean statement:

$$\boxed{\;\text{head } h \text{ writes } \sum_{j \le i} P_h(t_i, t_j)\, u_{t_j}, \qquad u_t \;=\; W_o^h W_v^h\, \hat e_t \;\in\; \mathbb{R}^{1152}.\;}$$

$u_t$ is "what attending to token $t$ with weight 1 deposits in the stream." In code:
`U_h = Vv[:, h] @ Wo[:, h].T`, shape $(V, 1152)$; each of the 9 such matrices costs 232 MB in
float32, so we build one at a time. Note $u$ lives in the image of $W_o^h W_v^h$, a rank-$\le
128$ map — the reader sees at most a 128-dimensional slice of the stream, and (checked
empirically below) uses ~80–96 directions of it at the 90%-energy level.

This is why pattern error is the wrong currency: an error $\Delta P(i,j)$ only matters through
the vector $\Delta P(i,j)\, u_{t_j}$ it appends to the stream. Two consequences we've already
measured: (a) errors on rare tokens are over-weighted if you average uniformly over the
vocabulary; (b) errors at different $j$ can *cancel* after multiplying by $u$, and SVD
residuals cancel ~3× more than dictionary residuals — so metrics that allow full cancellation
flatter SVD.

---

## 3. The delivery error, and why nothing $V \times V$ ever exists

Compressing QK only (tick 159/160), the delivered error at query position $i$ is

$$e_i \;=\; \sum_{j=1}^{T} \Delta P(t_i, t_j)\; u_{t_j} \;\in\; \mathbb{R}^{1152}, \qquad \Delta P = \hat P - P.$$

This depends on the whole context $t_1 \dots t_T$, which we don't want (weight-only program:
data enters evaluation, not the objective). So take the expectation over a *model* of
contexts: $T$ positions drawn i.i.d. from the unigram distribution $q \in \mathbb{R}^{V}$
(FineWeb token frequencies, +0.5 smoothing, normalized). Each summand $c_j = \Delta P(i, t_j)
u_{t_j}$ is then i.i.d. with

$$\mu_i \;=\; \mathbb{E}[c] \;=\; \sum_{t} q_t\, \Delta P(i,t)\, u_t \in \mathbb{R}^{1152}, \qquad s_i \;=\; \mathbb{E}\|c\|^2 \;=\; \sum_{t} q_t\, \Delta P(i,t)^2 \|u_t\|^2 \in \mathbb{R}.$$

Sum of $T$ i.i.d. vectors: $\mathbb{E}\,e_i = T\mu_i$ and
$\mathbb{E}\|e_i\|^2 = \sum_j \mathbb{E}\|c_j\|^2 + \sum_{j \ne l} \mathbb{E}[c_j] \cdot \mathbb{E}[c_l]$, which is exactly

$$\boxed{\;\mathbb{E}\|e_i\|^2 \;=\; T\,\big(s_i - \|\mu_i\|^2\big) \;+\; T^2\,\|\mu_i\|^2\;}\qquad (\dagger)$$

(variance term + squared-mean term; $s_i - \|\mu_i\|^2 \ge 0$ by Jensen, clamped in code only
for float noise). Finally average over the query token, $\sum_i q_i\, \mathbb{E}\|e_i\|^2$.

Why this form is the interesting one — read off the two extremes:

- Keep only the $T$-term (forbid cancellation entirely): that's the naive "norm-weighted
  pattern MSE" rung of the ladder. It over-charges errors that would have cancelled.
- Keep only the $T^2$-term (allow *full* cancellation): that's the Gram rung. It under-charges
  scatter — an error can hide by having mean zero while being huge pointwise.
- Eq. $(\dagger)$ says the statistically correct mix is: *incoherent* error grows like $T$,
  *systematic* error grows like $T^2$. At $T = 512$ the systematic term is heavily weighted,
  which is precisely the part where cancellation genuinely helps — cancellation is now
  *priced*, not forbidden and not free.

Computation on a step: sample $M = 1024$ tokens (indices into $V$), renormalize $q$ on the
sample. $\Delta P$ restricted to the sample is $M \times M$. Then $\mu$ for all $M$ query rows
is one matmul $(\Delta P \odot q)\,U_s$ with $U_s \in \mathbb{R}^{M \times 1152}$, and $s$ is an
elementwise $M \times M$ contraction against the $(M,)$ vector $q_t \|u_t\|^2$. Nothing bigger
than $M \times D$ is created.

---

## 4. Now let the reader move: the joint error

Logan's proposal: co-train a low-rank edit of OV. The delivered quantity for the compressed
head becomes $\hat P(i,j)\, \hat u_j$ with an edited reader $\hat u$, and the **faithful**
requirement is that it match the *original* delivery $P(i,j)\, u_j$. So the per-pair error is
no longer a scalar times a fixed vector — it's a difference of two different vectors:

$$d_{ij} \;=\; \hat P(i,j)\, \hat u_j \;-\; P(i,j)\, u_j \;\in\; \mathbb{R}^{1152}.$$

This is the crux of the whole design, so it deserves a beat: we do **not** ask the compressed
head to be good for next-token prediction (that objective overfit instantly in the tick-158
CE-polish experiment, and worse, it lets the optimizer *re-purpose* the head — see §6). We ask
it to impersonate the original head's writes to the stream. Pattern and reader may re-divide
labor between themselves, but their *product* is pinned to the original circuit.

Eq. $(\dagger)$ generalizes verbatim with $c_j = d_{i,t_j}$:

$$\mu_i = \sum_t q_t\, d_{it}, \qquad s_i = \sum_t q_t\, \|d_{it}\|^2, \qquad \mathbb{E}\|e_i\|^2 = T(s_i - \|\mu_i\|^2) + T^2 \|\mu_i\|^2.$$

First worry: $s_i$ now needs $\|d_{it}\|^2$ for every pair — naively an $M \times M \times 1152$
tensor (4.8 GB, and per training step). Expand the square instead:

$$\|d_{it}\|^2 \;=\; \hat P_{it}^2\, \underbrace{\|\hat u_t\|^2}_{\nu_t} \;-\; 2\, \hat P_{it} P_{it}\, \underbrace{\langle \hat u_t, u_t\rangle}_{\gamma_t} \;+\; P_{it}^2\, \underbrace{\|u_t\|^2}_{\omega_t}.$$

Only three *per-token scalars* $\nu, \gamma, \omega \in \mathbb{R}^{M}$ appear — precompute
them once per step, and $s$ is again three elementwise $M \times M$ contractions. The mean term
is two matmuls:

$$\mu \;=\; (\hat P \odot q)\,\hat U_s \;-\; (P \odot q)\,U_s \qquad \in \mathbb{R}^{M \times 1152}.$$

Total per step per head ≈ a few GFLOP; the 1500-step, 9-head training is minutes, and the
FineWeb audit dominates wall-clock. The loss is normalized by the same $(\dagger)$ energy of
the original signal ($\hat P \to 0$ formally, i.e. plug $P, u$ alone into the formula) so
"loss = 1" means "as wrong as deleting the head."

One subtlety that prevents a cheat: reconstructed table rows are re-unit-RMS'd before scoring
(both in the objective and in the audit), and the target is the original $P$ — so the
dictionary can't shrink the pattern and let the LoRA amplify the reader to compensate in norm
while degrading direction. The scale gauge is pinned on both sides.

---

## 5. The LoRA itself: shapes and bits

Per head, two low-rank edits, standard parameterization (zero-initialized $B$ so training
starts at *exactly* the original reader):

$$W_v^h \leftarrow W_v^h + A_v B_v, \quad A_v \in \mathbb{R}^{128 \times r},\; B_v \in \mathbb{R}^{r \times 1152}; \qquad W_o^h \leftarrow W_o^h + A_o B_o, \quad A_o \in \mathbb{R}^{1152 \times r},\; B_o \in \mathbb{R}^{r \times 128}.$$

Edited reader on a token sample, three small matmuls:

$$\hat U_s \;=\; \Big(E_s\,(W_v^h + A_v B_v)^\top\Big)\,(W_o^h + A_o B_o)^\top: \quad (M,1152) \to (M,128) \to (M,1152).$$

Bits, charged on top of the dictionary: $2r(1152 + 128)$ floats per head $\times\, 32$ bits
$\times\, 9$ heads. At $r=16$: $40{,}960$ floats/head $\to$ **11.8 Mbit total**; at $r=64$:
**47.2 Mbit**. Against dictionary budgets of 224–1289 Mbit this is 1–5% — if co-adaptation
works, it's nearly free.

Trainables per head in the `joint` arm: both branch dictionaries ($D_n, W_e, b$ each) plus the
four LoRA factors — Adam, learning rate $3 \times 10^{-4}$, gradient clip 1.0, 1500 steps. The
`lora_only` arm freezes the dictionaries at their MSE fit and trains just the four LoRA
factors — that isolates "how much does pure re-reading buy with the pattern held dumb."

For the *audit*, the model itself must run with the edited reader: the per-head deltas
$A_vB_v$ are added into rows $128h{:}128(h{+}1)$ of `c_v.weight` and $A_oB_o$ into columns
$128h{:}128(h{+}1)$ of `c_proj.weight` (upcast, add, cast back), the layer-0 scores are patched
with the dictionary tables (full machinery: rotary + causal mask, unlike the pre-rotary
training objective), held-out CE is measured on the standard 307k FineWeb predictions, and the
original weights are restored. Weight edit and score patch travel together — that pair *is*
the compressed circuit.

---

## 6. Why this could go wrong, and the three meters watching for it

**The gauge argument for why "moving OV" is not automatically cheating.** In the tensor-network
picture the score is a *scalar bond*: QK hands one number per (query, key) pair to OV. A bond
of dimension 1 has only scalar gauge freedom — rescale scores by $c$, reader by $1/c$, per
head. Anything beyond scale (any rotation of the reader) changes the function of *both* sides:
it is a lossy re-parameterization, i.e. genuinely moving computation between QK and OV. So the
honest description of what the joint arm compresses is not "the QK factor tables" but **the
composed head**: the map (query token, key token) $\mapsto$ residual-stream write. That's
arguably the natural functional unit anyway — the QK/OV cut inside it is a choice of internal
coordinates, and the composed object's complexity can be far below its parts' (layer-5 head 7
is causally rank *one* despite full-rank weights). MDL of the composed head is exactly what
"optimal downstream reading" means: spend bits only on structure that survives the contraction
with the reader.

**But the failure mode is real.** If the objective were downstream CE, the optimizer would
happily (a) turn nearly-dead heads (2 and 5 cost +0.001–0.002 when collapsed to position-only)
into nothing, (b) convert the reader into a static unigram prior that mimics average attention
output — re-implementing the lost selectivity as a bias that has nothing to do with attention,
which could equally live in an MLP. That is computation migration: the "QK MDL" would then
measure the model's ability to route around a lesion. The faithful delivery objective blocks
the incentive, but incentives aren't guarantees, so three meters are computed for every arm:

1. **Control audit — exact scores + edited reader.** Run the model with the *original* QK
   scores but the LoRA'd OV. If the reader were still doing its original job, CE shouldn't
   move: the edit should only matter in combination with the dictionary's errors. A large
   shift here = the reader became a free-standing model edit.
2. **Static-share meter.** Apply $(\dagger)$ to the *signal* (not the error): with
   $\mu^{\text{sig}}_i = \sum_t q_t P(i,t)u_t$, the share
   $\sum_i q_i T^2\|\mu^{\text{sig}}_i\|^2 \,/\, \sum_i q_i\,(\text{full }(\dagger))$ is the
   fraction of the head's delivered energy that is *context-independent* (the part a static
   bias could produce). Computed for the original head and for the compressed one. If
   compression + LoRA inflates this share, attention is being converted into a constant — the
   signature of migration, quantified. (Baseline fact worth knowing: the original layer-0
   heads are already ≈0.99 static under i.i.d.-unigram contexts at $T = 512$ — the $T^2$ term
   dominates — so the meter watches for *changes* around that value.)
3. **Reader-drift meters.** $\|\hat U - U\|_F / \|U\|_F$ over the full vocabulary (chunked,
   8192 rows at a time), and the frequency-weighted content rank: eigendecompose
   $C = \sum_t q_t u_t u_t^\top \in \mathbb{R}^{1152 \times 1152}$ and count eigenvalues to 90%
   of the trace, before and after. A reader that rotated onto a different subspace, or
   collapsed rank, changed its job.

There is also a bookkeeping cost worth stating once: if every circuit's compression is allowed
to edit its neighbors, per-circuit description lengths stop being additive (the OV would be
"described" twice). So any later work (the layer-1 object) must consume the *edited* OV, with
its delta charged exactly once.

---

## 7. The arms, and what each comparison isolates

All seed 0, all audited on the standard FineWeb 307k held-out predictions; comparators are the
tick-160 sweep numbers at identical dictionary budgets.

| arm | trains | budgets | question it answers |
|---|---|---|---|
| `joint`, $r{=}16$ | dicts + LoRA | (512,4) (1024,8) (4096,8) (4096,16) | does co-adaptation beat the fixed-reader context objective at matched bits? |
| `lora_only`, $r{=}16$ | LoRA only (dicts = MSE fit) | (1024,8) (4096,16) | how much does pure re-reading buy with a pattern trained blind to OV? |
| `joint`, $r{=}64$ | dicts + LoRA | (1024,8) (4096,16) | is reader capacity the binding constraint? |

The sharpest single question: the fixed-reader context objective plateaus at ≈ +0.005 for rich
budgets while MSE+OMP keeps improving. If a co-adapted reader breaks that plateau at
(4096,16), the plateau was the reader's fault; if not, it's the objective's approximation
floor (i.i.d. unigram contexts, pre-rotary scores) — and the fix is a better context model,
not a bendable reader.

**Result (all 8 arms done — clean negative, with informative meters).** Held-out ΔCE on
FineWeb, seed 0; sweep comparators at the same dictionary budgets in parentheses
(MSE-linear / MSE+OMP / fixed-reader context):

| arm | total Mbit | ΔCE | comparators |
|---|---|---|---|
| joint $r{=}16$, (512,4) | 235.9 | **+0.0069** | (.0144 / .0124 / **.0069**) |
| joint $r{=}16$, (1024,8) | 467.2 | **+0.0049** | (.0076 / .0059 / .0054) |
| joint $r{=}16$, (4096,8) | 934.6 | +0.0057 | (.0043 / **.0034** / .0042) |
| joint $r{=}16$, (4096,16) | 1253.4 | +0.0053 | (.0031 / **.0018** / .0052) |
| joint $r{=}64$, (1024,8) | 502.6 | +0.0054 | same as above |
| joint $r{=}64$, (4096,16) | 1288.8 | +0.0061 | same as above |
| LoRA-only, (1024,8) | 467.2 | +0.0087 | dict frozen at MSE fit |
| LoRA-only, (4096,16) | 1253.4 | +0.0036 | dict frozen at MSE fit |

Reading: co-adapting the reader buys essentially nothing anywhere — the joint arms land on
top of the fixed-reader context numbers (+0.0005 at the flagship budget, within seed spread),
rank 64 is no better than rank 16, and the plateau at (4096,16) does **not** break. LoRA-only
on a blind MSE pattern is *worse* than doing nothing at (1024,8) despite the reader moving 6.5%
— re-reading cannot rescue a pattern fitted blind to OV. The migration meters came back
uniformly quiet: every control audit (exact scores + edited reader) is ±0.0000, static share
stays at 0.99, reader drift under the joint objective is ~1% relative Frobenius, content rank
unchanged. So the faithful objective held the reader in place *and* the reader was not the
bottleneck: the original OV is already essentially the optimal reader of its own head's
compressed pattern, and the ≈+0.005 plateau is the objective's context-model floor (i.i.d.
unigram contexts, pre-rotary scores) — the fix is a better context model (co-occurrence
$q$, rotary inside the objective, blended loss), not a bendable reader.

---

## 8. Shape cheat-sheet

| object | shape | meaning |
|---|---|---|
| $\hat q^{(\beta)}, \hat k^{(\beta)}$ | $(50304, 9, 128)$ | folded factor tables, unit-RMS rows |
| $X$ (per head-branch) | $(50304, 256)$ | q-half ‖ k-half, the thing the dictionary fits |
| $D_n$ / $W_e$ / $b$ | $(n,256)$ / $(n,256)$ / $(256,)$ | decoder atoms, encoder, bias |
| $S^{(\beta)}_h$ on a sample | $(1024, 1024)$ | branch score sub-block, $XX^\top/128$ |
| $P = S^{(1)} \odot S^{(2)}$ | $(1024, 1024)$ | no-softmax pattern (pre-rotary in objective) |
| $\hat e$ rows / $E$ | $(50304, 1152)$ | unit-RMS embeddings |
| $W_v^h$ / $W_o^h$ | $(128,1152)$ / $(1152,128)$ | per-head slices of `c_v` / `c_proj` |
| $U_h = (E W_v^{h\top}) W_o^{h\top}$ | $(50304, 1152)$ | reader: stream-write per attended token |
| $A_v, B_v, A_o, B_o$ | $(128,r), (r,1152), (1152,r), (r,128)$ | LoRA factors, $B$'s zero-init |
| $\nu, \gamma, \omega$ | $(1024,)$ each | $\|\hat u\|^2,\ \langle \hat u, u\rangle,\ \|u\|^2$ on the sample |
| $\mu$ | $(1024, 1152)$ | mean delivered error per query token |
| $s$ | $(1024,)$ | mean squared delivered error per query token |
| $C = \sum_t q_t u_t u_t^\top$ | $(1152, 1152)$ | reader content covariance (rank-90 meter) |

Bits at a glance: dictionary $= 18 \times \big[32(256n + 256 + Vk) + Vk\log_2 n\big]$; LoRA
$= 9 \times 2r(1152{+}128) \times 32$ (11.8 Mbit at $r{=}16$, 47.2 at $r{=}64$); raw object
$= 32 \times 18 \times V \times 256 = 7417.6$ Mbit.

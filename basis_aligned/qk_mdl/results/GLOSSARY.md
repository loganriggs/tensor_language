# Glossary

Working definitions as used throughout results/ and the LOG. Shapes refer to bilin18
($V{=}50257$, $d{=}1152$, $H{=}9$ heads, $h{=}128$, $L{=}18$) unless noted.

**assignment** — the discrete half of a vq reduction: a map $a: V \to [k]$ from tokens to
atoms/classes; costs $V\log_2 k$ bits. Distinct from the *values* (the atoms), which are
continuous and trainable.

**atom / codebook / centroid** — one shared vector in a vq or dictionary reduction; the
codebook is the set of $k$ atoms.

**audit** — a ΔCE measurement of a patched model against the live baseline, on held-out
pile-10k chunks at $T=512$, batch 4. "The audit" always means this unless a different
distribution is named.

**baseline CE** — live-model cross-entropy on the audit set: 3.2341 (bilin18), 3.3717
(sqrd12); late-region chunks: 2.8633 (bilin18).

**behavioral Lloyd** — assignment refinement scored by the binding metric itself
(first-order: $g_t\cdot(C_{c'}-C_{a_t})$ from backprop through the *patched* model) instead
of a geometric proxy. Verdict: repairs unlucky partitions to best-tier, never beats it.

**binding metric** — the metric a claim must survive: ΔCE (per Logan). Pattern-MSE and FVU
are search-loop tools only; they dissociate from ΔCE by 10×+ routinely.

**branch / head-branch** — bilin18's attention multiplies two independent score maps
("branches"); a head-branch is one (head, branch) pair with its own $\hat q, \hat k$ factor
tables. 9 heads × 2 branches = 18 head-branches per layer.

**carriage** — what a circuit *transports* (OV values, residual content), as opposed to
*selection* (where attention looks). Carriage needs token identity; selection needs classes.

**C/S expansion** — computing RoPE'd scores from position-free factor tables via the
difference identities $\cos\Delta = c_i c_j + s_i s_j$, $\sin\Delta = s_i c_j - c_i s_j$,
using the model's own trig tables (keeps folding exact).

**composed vs marginal** — a *marginal* audit patches one component with everything else
live; a *composed* audit patches many simultaneously. Marginals systematically understate
composed cost (superadditivity) — every headline in this program is a composed number.

**cond-mean table** — $\bar z(t) = \mathbb{E}[z_i \mid t_i{=}t]$ for a contextual quantity
$z$; the 0th-order-in-context reduction. Data-estimated (report estimation tokens); unseen
tokens fall back to the global mean; renormalize to the object's natural shell (see *gauge*).

**ΔCE** — audited cross-entropy of the patched model minus baseline. Negative = better than
the live model (usually means data adaptation or denoising).

**description length (DL) / MDL** — bits to describe the computation: structural bits
(floats @32 b frozen convention + assignment bits) reported *beside* estimation-token counts,
never mixed (convention set tick 56).

**factor / factor table** — the per-token query or key vector of a head-branch after
QK-norm, before RoPE: $\hat q^{ab} \in \mathbb{R}^{V\times h}$. Exact at layer 0 (weights
fold); data-estimated above.

**folding** — absorbing the embedding (or unembedding) into an adjacent circuit's weights so
the circuit becomes a vocab-indexed table. Exact only where the circuit's input is
token-determined (layer 0; the embedding path at any depth).

**gate** — an exactness check that a reduction/harness reproduces the live model to fp
tolerance before any experiment runs on it (tier0_gate.py: ~1e-15; stream decompositions:
stream-sum ≡ residual, pair-sum ≡ score).

**gauge** — a transformation that changes representation but not function (branch sign flips,
per-query positive scaling under row normalization, the unit-RMS shell for QK-normed factors).
Reductions must respect the gauge or pay for it (raw vs unit-RMS cond-means: 3×).

**hub stream** — a stream consumed far beyond the usual window; attn5's output is THE hub in
bilin18 (persists in top-pair energies through L16).

**live window / $W$** — in windowed-D, the number of most recent layers whose streams stay
live in a read; everything older is tabled. Cost roughly halves per +1 of $W$.

**pattern** — the attention weight matrix. In this model zoo there is NO softmax: bilin18's
pattern is an unnormalized product of two branch scores; sqrd12's is squared-and-row-normalized.

**rank-$k$ (live coefficients)** — replacing an object by its token mean + projections onto
$k$ fixed directions with coefficients computed by the live model. A structural factorization
claim, not a compute reduction.

**read** — one consumer's view of the residual: the QK read (selection inputs), the v read
(carriage inputs), the MLP read. Windowed-D patches *reads*, never the residual itself.

**selection** — where attention looks (the QK circuit / pattern). Tolerates hard classes and
0th-order context nearly everywhere; the exceptions are two heads (H5 match, H7 gain).

**self / cross / pair blocks** — the layer-0 bilinear-MLP decomposition of the hidden by
input streams: $e{\odot}e$, $e{\odot}a$, $a{\odot}a$.

**streams / residual identity** — the exact decomposition of the residual into the embedding
path + every lower layer's attn-out and mlp-out (2ℓ+1 tensors at layer ℓ). Exact because the
$\lambda$-mix is linear and both RMSNorms are per-position scalars.

**superadditive** — composed cost exceeding the sum of marginals; the default on this model
family (up to 10×). The lone additive exception: qk+v windowing.

**T=512** — audit sequence length, frozen because bilin18's CE degrades beyond ~512.

**trust region (assignments)** — iterative refinement protocol: small damped move batches,
held-out audit each step, revert-and-halve on worsening. Mandatory: mass moves under
first-order scores reliably backfire.

**vq-$k$** — vector quantization with $k$ atoms; for QK factors the partition is shared
across $[\hat q | \hat k]$ so both factors of a head-branch use the same token classes.

**windowed-D / windowed code propagation** — the flagship architecture (results/11): every
read = exact embedding path + cond-mean tables for streams older than $W$ + live recent
streams. Bounds error-chain depth at $W$; composes where score-space tabling walled.

**zero control** — auditing with a component silenced (scores zeroed / stream removed);
establishes whether the component is load-bearing before interpreting its reduction cost.

**0th / first-order in context** — 0th-order: a table indexed by the current token only.
First-order: live context weights × classed content (e.g., live pattern × classed OV).
The tier-3 lesson: content lookups need at least first order; selection is ~0th-order.

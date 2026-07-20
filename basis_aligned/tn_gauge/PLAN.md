# Tensor-network gauge sweep: toy verification and the problems I see

Logan asked me to try the joint/iterative interaction-sparsity optimization (the
TN-pure "gauge sweep" version), starting on a toy to make sure the code works and
nothing blows up, and to flag problems. I did the toy first. It changed my view of
the plan in one load-bearing way, and confirmed three of the concerns exactly.

Toy: `runs_lm/block2-seed0` = `[attn, mlp, attn, mlp]`, d=128, 4 heads,
d_head=32, hidden=512, vocab 1024, RMSNorm, lerp residual. Same bilinear family as
bilin18 (bilinear attention + bilinear gated MLP `D((Lx)⊙(Rx))`), fully polynomial.
Script: `toy_gauge_probe.py`. Results: `toy_gauge_probe.json`.

## What the toy verified (each is exact arithmetic, max|Δlogit| reported)

| Check | Claim | Result |
|---|---|---|
| A | A global orthogonal rotation of the residual basis is an exact gauge (RMSNorm is rotation-equivariant) | **EXACT** (Δlogit 8e-5) |
| B | Pinning the embedding basis (rank = d) forces that rotation to identity | embed rank **128/128** → residual interior gauge DOFs = **0** |
| C1 | Attention OV head-subspace carries a full O(d_head) private gauge | **EXACT** (Δlogit 4e-5) |
| C2 | Attention QK head-subspace rotation is *not* free — RoPE breaks it | **BROKEN** (Δlogit 18.7) |
| C3 | The MLP hidden basis is pinned by the elementwise `⊙`; only permutation+scaling is free | rotation **BREAKS** (39.6); perm+scale **EXACT** (5e-5) |
| D | An L1-minimizing OV gauge optimizes and does not blow up | mean\|VT\| 0.85→0.82, orth-err 2e-4, **CE unchanged** |
| E | The weight-only cross-layer interaction DAG is sparse ("only some layers matter") | **NOT on this toy** — uniform, spread 1.1× |

## The headline problem: a residual stream is one shared bond, not a chain

The DMRG/LIB picture treats each inter-layer wire as a private bond you can
independently re-gauge, sweeping between two fixed boundaries. A transformer's
residual stream is **not** that. It is a single vector space carried through by the
skip connection; every layer reads and writes the *same* running basis. So:

- There is exactly **one** global residual gauge (Check A), not one per layer.
- The embedding writes directly into that basis and the unembedding reads directly
  from it. If both are "meaningful by fiat" (pinned), and the embedding has rank d
  (Check B: it does, 128/128), then the **only** residual rotation that keeps the
  embedding fixed is the identity. **The two boundaries do not bracket a free
  interior — they pin the entire trunk.**

Consequence: for the shared residual bond there is *nothing to sweep and no
middle-basis SAE to train*. The recursion the proposal worried about doesn't just
terminate at the boundaries — for the residual basis it is empty. That is good
news (no fixed point to chase, no deep-layer SAE) and it directly answers Logan's
"I'm concerned about SAEs on layers > 0": for the residual stream you don't need
one, and shouldn't introduce one, to keep the model a tensor network.

## What *is* free, and it's better-posed than a sweep

The real gauge freedoms are **per-layer private bonds**, and because the residual
basis is fixed they are mutually **independent** — embarrassingly parallel, not a
DMRG chain:

1. **Attention OV** (Check C1): full O(d_head) per head. Rotate value rows, unrotate
   the corresponding `o` input columns. This is exactly the layer-0 OV dictionary
   work generalized to any layer, and it is where "rotate to sparsify the
   interaction core" genuinely applies. Check D shows it optimizes cleanly.
2. **Attention QK** (Check C2): **constrained** — a free head rotation is broken by
   RoPE. QK lives in a RoPE-pinned basis; its natural anchor is the input side, not
   a free rotation. This is the structural reason the backward pass "misses QK"
   (below): QK is not on any output-linear path and not freely rotatable.
3. **MLP hidden** (Check C3): the elementwise `⊙` **pins the basis**. You cannot
   rotate-to-sparsify the MLP interaction core — it is *already* in its privileged
   basis (the hidden units). The only gauge is permutation + scaling. So the MLP
   "interaction sparsity" problem is **not** a rotation problem; it is a
   *which-hidden-units-interact* problem (masking/clustering over fixed units), a
   different objective than the OV rotation. This is a real correction to the plan:
   the L1-on-a-rotated-core move works for OV, not for the MLP.

So the honest restructuring: **not** "sweep a chain of residual bases between two
boundaries," but "independently gauge each layer's private OV subspace (rotation),
QK subspace (RoPE-constrained, input-anchored), and MLP hidden units (permutation
+ mask, no rotation), with the residual trunk fixed by the boundaries." No sweep,
no oscillation risk, no deep-layer SAE, and it stays exactly a tensor network.

## The backward-pass confusion, resolved

Folding the unembedding backward propagates along **linear/value** paths — OV
outputs, MLP outputs, the residual identity. QK scores enter the output only as
data-dependent *pattern coefficients*, never on a linear path, so a linear backward
pass assigns QK nothing. Check C2 is the same fact from the other side: QK has no
free output-facing gauge. **This is not a bug; it is the asymmetry.** The typed
resolution: content/value bases (OV, MLP, unembedding-facing) anchor **backward**
from the unembedding; selection bases (QK read-sides) anchor **forward** from the
embedding via the propagated layer-0 alphabet. Going backward *through* a bilinear
MLP is available and cheap in CP coordinates (contract the output leg against the
anchored direction → an input-side symmetric pencil), but it widens per layer and
must be truncated+reclustered — the mirror of the forward blowup.

## The blowup, and what the toy flags about it

The degree-2ᴸ explosion only happens if you expand to embedding-monomials. Measure
interactions **at the bonds** (small contraction cores) instead, and degree never
appears. But Check E is a caution: the naive **weight-only** composition-norm DAG
between layers is *uniform* on this toy (every writer feeds every reader ~equally,
spread 1.1×), so weight magnitudes alone do **not** reveal "only some layers
matter." That matches the proposal's own Tier-1.5 caveat: magnitude-scored
contraction misranks. The fix is exactly Logan's instinct — score the DAG with the
**data-weighted measure** propagated from the layer-0 QK reduction (attention-
weighted co-occurrence over the small alphabet), not raw weight norms. So "learn
which layers interact" is real but needs the propagated context, not weights alone.

## Concrete recipe (revised from the toy)

1. Fix embedding and unembedding bases (meaningful). The residual trunk is then
   fixed — no residual gauge to solve.
2. **Per layer, independently and in parallel:**
   - OV heads: L1/varimax rotation in O(d_head) to sparsify the token→value core
     (Check D primitive). TN-pure.
   - QK heads: do **not** rotate; anchor forward from the embedding/alphabet, score
     by pattern faithfulness (RoPE-respecting).
   - MLP: permutation + mask over the fixed hidden units; cluster interacting units;
     no rotation.
3. Cross-layer DAG: score with the **propagated alphabet measure**, not weight
   norms (Check E). Threshold to the sparse layer-interaction graph.
4. Only if a specific bond's sparsity floor stays high (rotation can't undo genuine
   superposition) consider the minimal non-TN step: a linear dictionary with
   input-dependent *support* (still a tensor + a mask field), never a nonlinear SAE.

The model stays a tensor network throughout. Next: port the OV-rotation primitive
and the propagated-measure DAG onto bilin18 layers > 0 and see how the OV sparsity
floor grows with depth (LIB's honest prediction: difficulty grows with distance
from the boundaries).
